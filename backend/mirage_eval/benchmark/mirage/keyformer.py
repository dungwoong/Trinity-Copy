import argparse

import torch

import mirage as mi
from mirage.kernel import KNGraph

HIDDEN_DIM = 4096
SEQ_LEN = 16
CONTEXT_LEN = 1024
NUM_HEADS = 32
HEAD_DIM = 128


def projection():
    """
    Builds and superoptimizes a kernel graph for combined QKV projection
    without prenormalization.

    This kernel implements a simple linear projection for Q, K, V
    without any normalization, following the keyformer_wo_prenorm IR structure.

    Returns:
        KNGraph: The superoptimized kernel graph for QKV projections.
    """
    _graph = mi.new_kernel_graph()

    # Input tensor and combined weight matrix
    x = _graph.new_input(dims=(SEQ_LEN, HIDDEN_DIM), dtype=mi.float16)
    w = _graph.new_input(dims=(HIDDEN_DIM, HIDDEN_DIM * 3), dtype=mi.float16)

    # Combined QKV projection without normalization
    o = _graph.matmul(x, w)

    # Mark output
    _graph.mark_output(o)

    return _graph.superoptimize(config="mlp")


def standard_attention():
    """
    Builds and superoptimizes a kernel graph for standard attention.

    This kernel implements the standard attention mechanism:
    Attention(Q,K,V) = softmax(Q @ K^T) @ V

    Returns:
        KNGraph: The superoptimized kernel graph for standard attention.
    """
    _graph = mi.new_kernel_graph()

    # Inputs
    q = _graph.new_input(dims=(NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=mi.float16)
    k_cache = _graph.new_input(
        dims=(NUM_HEADS, HEAD_DIM, CONTEXT_LEN), dtype=mi.float16
    )
    v_cache = _graph.new_input(
        dims=(NUM_HEADS, CONTEXT_LEN, HEAD_DIM), dtype=mi.float16
    )

    # C = Q @ K^T
    c = _graph.matmul(q, k_cache)  # (NUM_HEADS, SEQ_LEN, CONTEXT_LEN)

    # Standard softmax
    c_exp = _graph.exp(c)
    c_sum = _graph.reduction(c_exp, 2)  # (NUM_HEADS, SEQ_LEN)
    c_div = _graph.div(c_exp, c_sum)  # (NUM_HEADS, SEQ_LEN, CONTEXT_LEN)

    # Standard attention output
    o = _graph.matmul(c_div, v_cache)  # (NUM_HEADS, SEQ_LEN, HEAD_DIM)

    # Mark output
    _graph.mark_output(o)
    _graph.mark_output(c)

    return _graph.superoptimize(config="attention")


def run_cycle(
    kernels: list[KNGraph],
    x: torch.Tensor,
    w: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    noise: torch.Tensor,
    tau: float,
    stream: torch.cuda.Stream,
):
    # Combined QKV projection without RMSNorm
    projection_kernel = kernels[0]
    X_qkv: torch.Tensor = projection_kernel(inputs=[x, w], stream=stream)[0]  # type: ignore

    # Split QKV
    x_q = X_qkv[:, :HIDDEN_DIM]
    x_k = X_qkv[:, HIDDEN_DIM : HIDDEN_DIM * 2]
    x_v = X_qkv[:, HIDDEN_DIM * 2 :]

    # Reshape for multi-head attention
    x_q = x_q.reshape(SEQ_LEN, NUM_HEADS, HEAD_DIM)
    x_k = x_k.reshape(SEQ_LEN, NUM_HEADS, HEAD_DIM)
    x_v = x_v.reshape(SEQ_LEN, NUM_HEADS, HEAD_DIM)

    # Permute to (NUM_HEADS, SEQ_LEN, HEAD_DIM)
    q = x_q.permute(1, 0, 2)
    k = x_k.permute(1, 0, 2)
    v = x_v.permute(1, 0, 2)

    # Update caches
    k_cache[:, -SEQ_LEN:, ...] = k
    v_cache[:, -SEQ_LEN:, ...] = v

    # Transpose K for Q @ K^T
    k_transposed = k_cache.permute(0, 2, 1)  # (NUM_HEADS, HEAD_DIM, CONTEXT_LEN)

    standard_attention_kernel = kernels[1]
    attention_output, c = standard_attention_kernel(
        inputs=[q, k_transposed, v_cache], stream=stream
    )   # type: ignore

    c_perturb = (c + noise) / tau

    # Perturbed attention
    c_div_perturb = torch.softmax(c_perturb, dim=-1)
    c_out = torch.sum(c_div_perturb, dim=1)

    # Reshape attention output
    attention_output = attention_output.permute(
        1, 0, 2
    )  # (SEQ_LEN, NUM_HEADS, HEAD_DIM)
    attention_output = attention_output.reshape(SEQ_LEN, HIDDEN_DIM)

    return attention_output, c_out


def main():
    parser = argparse.ArgumentParser(
        description="Keyformer Attention Benchmark Without Prenorm"
    )
    parser.add_argument(
        "--warmup", type=int, default=16, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--profile", type=int, default=1000, help="Number of profiling iterations"
    )
    args = parser.parse_args()

    print("Running Keyformer attention benchmark without prenorm with:")
    print(f"  Warmup iterations: {args.warmup}")
    print(f"  Profile iterations: {args.profile}")
    print("  CUDA graphs: Enabled")
    print()

    # Initialize tensors
    X = torch.randn(
        SEQ_LEN, HIDDEN_DIM, dtype=torch.float16, device=torch.device("cuda:0")
    )

    # Combined weight matrix for QKV
    Wqkv = torch.randn(
        HIDDEN_DIM, HIDDEN_DIM * 3, dtype=torch.float16, device=torch.device("cuda:0")
    )

    # Initialize caches
    k_cache_init = torch.randn(
        NUM_HEADS,
        CONTEXT_LEN,
        HEAD_DIM,
        dtype=torch.float16,
        device=torch.device("cuda:0"),
    )
    v_cache_init = torch.randn(
        NUM_HEADS,
        CONTEXT_LEN,
        HEAD_DIM,
        dtype=torch.float16,
        device=torch.device("cuda:0"),
    )

    # Noise tensor for perturbation
    noise = (
        torch.randn(
            NUM_HEADS,
            SEQ_LEN,
            CONTEXT_LEN,
            dtype=torch.float16,
            device=torch.device("cuda:0"),
        )
        * 0.01
    )  # Small noise
    
    tau = 0.5

    # Create kernel graphs
    print("Creating and optimizing kernel graphs...")
    projection_kernel = projection()
    standard_kernel = standard_attention()
    kernels = [projection_kernel, standard_kernel]
    print("Kernel optimization complete.")

    # Create stream for all operations
    capture_stream = torch.cuda.Stream()

    # Warmup phase
    print(f"Running {args.warmup} warmup iterations...")
    with torch.cuda.stream(capture_stream):
        for _ in range(args.warmup):
            k_cache = k_cache_init.clone()
            v_cache = v_cache_init.clone()
            run_cycle(
                kernels, X, Wqkv, k_cache, v_cache, noise, tau, capture_stream
            )
        capture_stream.synchronize()
    print("Warmup complete.")

    print("Capturing CUDA graph...")
    cuda_graph = torch.cuda.CUDAGraph()

    # Prepare tensors for graph capture
    k_cache_graph = k_cache_init.clone()
    v_cache_graph = v_cache_init.clone()
    X_graph = X.clone()
    noise_graph = noise.clone()

    # Capture the graph
    with torch.cuda.graph(cuda_graph, stream=capture_stream):
        run_cycle(
            kernels,
            X_graph,
            Wqkv,
            k_cache_graph,
            v_cache_graph,
            noise_graph,
            tau,
            capture_stream,
        )

    print(f"Running {args.profile} test iterations with timing...")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    iteration_stream = torch.cuda.Stream()

    with torch.cuda.stream(iteration_stream):
        start_event.record(iteration_stream)

        for _ in range(args.profile):
            run_cycle(kernels, X, Wqkv, k_cache, v_cache, noise, tau, iteration_stream)

        end_event.record(iteration_stream)
        end_event.synchronize()

    print("Profiling complete.\n")
    mean_time = start_event.elapsed_time(end_event)

    # Print results
    print("=" * 60)
    print("BENCHMARK RESULTS - Keyformer Without Prenorm")
    print("=" * 60)
    print("Configuration:")
    print(f"  Hidden dim: {HIDDEN_DIM}")
    print(f"  Sequence length: {SEQ_LEN}")
    print(f"  Context length: {CONTEXT_LEN}")
    print(f"  Number of heads: {NUM_HEADS}")
    print(f"  Head dimension: {HEAD_DIM}")
    print("  Features: Perturbed Attention (no prenorm)")
    print("  CUDA graphs: Enabled")
    print("\nTiming Statistics (ms):")
    print(f"  Mean:   {mean_time:.4f}")
    print("=" * 60)

    return mean_time


if __name__ == "__main__":
    main()
