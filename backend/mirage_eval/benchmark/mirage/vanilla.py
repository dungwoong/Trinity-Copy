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
    Builds and superoptimizes a kernel graph for a linear projection operation.

    This kernel is used for QKV (Query, Key, Value) projection in attention mechanisms.

    The function creates a Mirage kernel graph that takes an input tensor `x` of shape
    (SEQ_LEN, HIDDEN_DIM) and a weight matrix `w` of shape (HIDDEN_DIM, HIDDEN_DIM),
    performs matrix multiplication, and marks the result as the output. The graph is
    then superoptimized with the "mlp" configuration.

    Returns:
        KNGraph: The superoptimized kernel graph for the projection operation.
    """
    _graph = mi.new_kernel_graph()

    x = _graph.new_input(dims=(SEQ_LEN, HIDDEN_DIM), dtype=mi.float16)
    w = _graph.new_input(dims=(HIDDEN_DIM, HIDDEN_DIM * 3), dtype=mi.float16)
    o = _graph.matmul(x, w)

    _graph.mark_output(o)
    return _graph.superoptimize(config="mlp")


def attention():
    """
    Builds and superoptimizes a kernel graph for multi-head attention operation.

    This kernel implements the scaled dot-product attention mechanism used in
    vanilla Transformers. It takes query and concatenated key/value cache tensors,
    performs attention computation, and returns the attention output.

    The attention follows the formula: Attention(Q,K,V) = softmax(QK^T)V

    Returns:
        KNGraph: The superoptimized kernel graph for the attention operation.
    """
    _graph = mi.new_kernel_graph()

    # Input: Q already permuted to (NUM_HEADS, SEQ_LEN, HEAD_DIM)
    q = _graph.new_input(dims=(NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=mi.float16)

    # K_cache and V_cache are already concatenated outside the graph
    k_cache = _graph.new_input(
        dims=(NUM_HEADS, HEAD_DIM, CONTEXT_LEN), dtype=mi.float16
    )
    v_cache = _graph.new_input(
        dims=(NUM_HEADS, CONTEXT_LEN, HEAD_DIM), dtype=mi.float16
    )

    # C = Q @ K^T
    c = _graph.matmul(q, k_cache)  # (NUM_HEADS, SEQ_LEN, CONTEXT_LEN)

    # Manual softmax implementation
    # Note: Mirage doesn't have built-in softmax, so we decompose it
    # C_exp = exp(C)
    c_exp = _graph.exp(c)

    # C_sum = reduce_sum(C_exp, axis=2)
    # This reduces the last dimension, resulting in (NUM_HEADS, SEQ_LEN)
    c_sum = _graph.reduction(c_exp, 2)  # (NUM_HEADS, SEQ_LEN)

    # For proper broadcasting, we need to reshape c_sum to (NUM_HEADS, SEQ_LEN, 1)
    # In Mirage, broadcasting happens automatically during div operation
    # C_div = C_exp / C_sum (with broadcasting)
    c_div = _graph.div(c_exp, c_sum)  # (NUM_HEADS, SEQ_LEN, CONTEXT_LEN)

    # O = C_div @ V
    o = _graph.matmul(c_div, v_cache)  # (NUM_HEADS, SEQ_LEN, HEAD_DIM)

    # Output is still (NUM_HEADS, SEQ_LEN, HEAD_DIM)
    # Reshaping to (SEQ_LEN, HIDDEN_DIM) will be done outside the graph
    _graph.mark_output(o)
    return _graph.superoptimize(config="attention")


def run_cycle(
    kernels: list[KNGraph],
    x: torch.Tensor,
    w: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    stream: torch.cuda.Stream,
):
    # Projection
    projection_kernel = kernels[0]

    X_qkv: torch.Tensor = projection_kernel(inputs=[x, w], stream=stream)[0]  # type: ignore

    x_q = X_qkv[:, :HIDDEN_DIM]
    x_k = X_qkv[:, HIDDEN_DIM : HIDDEN_DIM * 2]
    x_v = X_qkv[:, HIDDEN_DIM * 2 :]

    # Reshape projected tensors for multi-head attention
    x_q = x_q.reshape(SEQ_LEN, NUM_HEADS, HEAD_DIM)
    x_k = x_k.reshape(SEQ_LEN, NUM_HEADS, HEAD_DIM)
    x_v = x_v.reshape(SEQ_LEN, NUM_HEADS, HEAD_DIM)

    # Permute to (NUM_HEADS, SEQ_LEN, HEAD_DIM)
    q = x_q.permute(1, 0, 2)
    k = x_k.permute(1, 0, 2)
    v = x_v.permute(1, 0, 2)

    k_cache[:, -SEQ_LEN:, ...] = k
    v_cache[:, -SEQ_LEN:, ...] = v

    # Transpose K for Q @ K^T
    k_transposed = k_cache.permute(
        0, 2, 1
    )  # (NUM_HEADS, HEAD_DIM, CONTEXT_LEN + SEQ_LEN)

    # Attention kernel expects: q, k_transposed, v_concat
    attention_kernel = kernels[1]
    attention_output = attention_kernel(
        inputs=[q, k_transposed, v_cache], stream=stream
    )[0]  # type: ignore

    # Permute back to (SEQ_LEN, NUM_HEADS, HEAD_DIM)
    # and reshape to (SEQ_LEN, HIDDEN_DIM)
    attention_output = attention_output.permute(
        1, 0, 2
    )  # (SEQ_LEN, NUM_HEADS, HEAD_DIM)
    attention_output = attention_output.reshape(SEQ_LEN, HIDDEN_DIM)

    return attention_output


def main():
    parser = argparse.ArgumentParser(description="Vanilla Attention Benchmark")
    parser.add_argument(
        "--warmup", type=int, default=16, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--profile", type=int, default=1000, help="Number of profiling iterations"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed timing information"
    )
    parser.add_argument(
        "--backend", type=str, default="cuda", help="Backend to use (cuda/nki)"
    )
    parser.add_argument(
        "--save-codes", action="store_true", help="Save generated kernel codes"
    )
    args = parser.parse_args()

    print("Running vanilla attention benchmark with:")
    print(f"  Warmup iterations: {args.warmup}")
    print(f"  Profile iterations: {args.profile}")
    print("  CUDA graphs: Enabled")
    print(f"  Backend: {args.backend}")
    print()

    # Initialize tensors
    X = torch.randn(
        SEQ_LEN, HIDDEN_DIM, dtype=torch.float16, device=torch.device("cuda:0")
    )

    Wqkv = torch.randn(
        HIDDEN_DIM, HIDDEN_DIM * 3, dtype=torch.float16, device=torch.device("cuda:0")
    )

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

    # Create kernel graphs
    print("Creating and optimizing kernel graphs...")
    projection_kernel = projection()
    attention_kernel = attention()
    kernels = [projection_kernel, attention_kernel]
    print("Kernel optimization complete.")

    # Create stream for all operations
    capture_stream = torch.cuda.Stream()

    # Warmup phase
    print(f"Running {args.warmup} warmup iterations...")
    with torch.cuda.stream(capture_stream):
        for _ in range(args.warmup):
            k_cache = k_cache_init.clone()
            v_cache = v_cache_init.clone()
            run_cycle(kernels, X, Wqkv, k_cache, v_cache, capture_stream)
        capture_stream.synchronize()
    print("Warmup complete.")

    print("Capturing CUDA graph...")
    cuda_graph = torch.cuda.CUDAGraph()

    # Prepare tensors for graph capture
    k_cache_graph = k_cache_init.clone()
    v_cache_graph = v_cache_init.clone()
    X_graph = X.clone()

    # Capture the graph
    with torch.cuda.graph(cuda_graph, stream=capture_stream):
        run_cycle(kernels, X_graph, Wqkv, k_cache_graph, v_cache_graph, capture_stream)

    print(f"Running {args.profile} test iterations with timing...")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    iteration_stream = torch.cuda.Stream()

    with torch.cuda.stream(iteration_stream):
        start_event.record(iteration_stream)

        for _ in range(args.profile):
            run_cycle(kernels, X, Wqkv, k_cache, v_cache, iteration_stream)

        end_event.record(iteration_stream)
        end_event.synchronize()

    print("Profiling complete.\n")
    mean_time = start_event.elapsed_time(end_event)

    # Print results
    print("=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print("Configuration:")
    print(f"  Hidden dim: {HIDDEN_DIM}")
    print(f"  Sequence length: {SEQ_LEN}")
    print(f"  Context length: {CONTEXT_LEN}")
    print(f"  Number of heads: {NUM_HEADS}")
    print(f"  Head dimension: {HEAD_DIM}")
    print("  CUDA graphs: Enabled")
    print("\nTiming Statistics (ms):")
    print(f"  Mean:   {mean_time:.4f}")
    print("=" * 60)

    return mean_time


if __name__ == "__main__":
    main()
