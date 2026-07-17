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


def qk_norm_matmul():
    """
    Builds and superoptimizes a kernel graph for Q normalization and matmul with K.

    This kernel normalizes the query tensor using RMS normalization and then
    computes the attention scores by multiplying with the key cache.

    Returns:
        KNGraph: The superoptimized kernel graph for Q normalization and K matmul.
    """
    _graph = mi.new_kernel_graph()

    # Input: Q already permuted to (NUM_HEADS, SEQ_LEN, HEAD_DIM)
    q = _graph.new_input(dims=(NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=mi.float16)

    # K_cache will store normalized keys: (NUM_HEADS, HEAD_DIM, CONTEXT_LEN)
    k_cache = _graph.new_input(
        dims=(NUM_HEADS, HEAD_DIM, CONTEXT_LEN), dtype=mi.float16
    )

    # Q normalization using built-in RMS norm
    # RMS norm computes: x / sqrt(mean(x^2))
    q_normalized = _graph.rms_norm(q, normalized_shape=(HEAD_DIM,))

    # C = Q_norm @ K_norm^T
    c = _graph.matmul(q_normalized, k_cache)  # (NUM_HEADS, SEQ_LEN, CONTEXT_LEN)

    _graph.mark_output(c)
    return _graph.superoptimize(config="attention")


def softmax_v_matmul():
    """
    Builds and superoptimizes a kernel graph for softmax and matmul with V.

    This kernel takes attention scores, applies softmax, and multiplies
    with the value cache to produce the final attention output.

    Returns:
        KNGraph: The superoptimized kernel graph for softmax and V matmul.
    """
    _graph = mi.new_kernel_graph()

    # Attention scores from qk_norm_matmul
    c = _graph.new_input(dims=(NUM_HEADS, SEQ_LEN, CONTEXT_LEN), dtype=mi.float16)

    # V_cache: (NUM_HEADS, CONTEXT_LEN, HEAD_DIM)
    v_cache = _graph.new_input(
        dims=(NUM_HEADS, CONTEXT_LEN, HEAD_DIM), dtype=mi.float16
    )

    # Manual softmax implementation
    # C_exp = exp(C)
    c_exp = _graph.exp(c)

    # C_sum = reduce_sum(C_exp, axis=2)
    c_sum = _graph.reduction(c_exp, 2)  # (NUM_HEADS, SEQ_LEN)

    # C_div = C_exp / C_sum (with broadcasting)
    c_div = _graph.div(c_exp, c_sum)  # (NUM_HEADS, SEQ_LEN, CONTEXT_LEN)

    # O = C_div @ V
    o = _graph.matmul(c_div, v_cache)  # (NUM_HEADS, SEQ_LEN, HEAD_DIM)

    # Output is still (NUM_HEADS, SEQ_LEN, HEAD_DIM)
    _graph.mark_output(o)
    return _graph.superoptimize(config="attention")


def normalize_keys(k: torch.Tensor) -> torch.Tensor:
    """
    Normalize keys using RMS normalization.

    Args:
        k: Keys tensor of shape (NUM_HEADS, SEQ_LEN, HEAD_DIM)

    Returns:
        Normalized keys tensor of the same shape
    """
    # RMS normalization: k / sqrt(mean(k^2))
    # Compute mean of squared values along HEAD_DIM dimension
    k_squared = k * k
    k_mean_squared = k_squared.mean(dim=2, keepdim=True)
    k_rms = torch.sqrt(k_mean_squared + 1e-6)  # Add epsilon for numerical stability
    k_normalized = k / k_rms

    return k_normalized


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

    # Normalize K before caching (QK normalization)
    k_normalized = normalize_keys(k)

    # Update caches
    # k_cache stores normalized keys
    k_cache[:, -SEQ_LEN:, ...] = k_normalized
    v_cache[:, -SEQ_LEN:, ...] = v

    # Transpose K for Q @ K^T
    k_transposed = k_cache.permute(
        0, 2, 1
    )  # (NUM_HEADS, HEAD_DIM, CONTEXT_LEN + SEQ_LEN)

    # QK norm and matmul
    qk_norm_kernel = kernels[1]
    attention_scores = qk_norm_kernel(inputs=[q, k_transposed], stream=stream)[0]  # type: ignore

    # Softmax and V matmul
    softmax_v_kernel = kernels[2]
    attention_output = softmax_v_kernel(
        inputs=[attention_scores, v_cache], stream=stream
    )[0]  # type: ignore

    # Permute back to (SEQ_LEN, NUM_HEADS, HEAD_DIM)
    # and reshape to (SEQ_LEN, HIDDEN_DIM)
    attention_output = attention_output.permute(
        1, 0, 2
    )  # (SEQ_LEN, NUM_HEADS, HEAD_DIM)
    attention_output = attention_output.reshape(SEQ_LEN, HIDDEN_DIM)

    return attention_output


def main():
    parser = argparse.ArgumentParser(description="QKNorm Attention Benchmark")
    parser.add_argument(
        "--warmup", type=int, default=16, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--profile", type=int, default=1000, help="Number of profiling iterations"
    )
    args = parser.parse_args()

    print("Running QK-normalized attention without prenorm benchmark with:")
    print(f"  Warmup iterations: {args.warmup}")
    print(f"  Profile iterations: {args.profile}")
    print("  CUDA graphs: Enabled")
    print()

    X = torch.randn(
        SEQ_LEN, HIDDEN_DIM, dtype=torch.float16, device=torch.device("cuda:0")
    )
    Wqkv = torch.randn(
        HIDDEN_DIM, HIDDEN_DIM * 3, dtype=torch.float16, device=torch.device("cuda:0")
    )

    # Initialize k_cache with normalized keys
    k_cache_init = torch.randn(
        NUM_HEADS,
        CONTEXT_LEN,
        HEAD_DIM,
        dtype=torch.float16,
        device=torch.device("cuda:0"),
    )
    # Normalize the initial k_cache
    k_cache = normalize_keys(k_cache_init)

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
    qk_norm_kernel = qk_norm_matmul()
    softmax_v_kernel = softmax_v_matmul()
    kernels = [projection_kernel, qk_norm_kernel, softmax_v_kernel]
    print("Kernel optimization complete.")

    capture_stream = torch.cuda.Stream()

    print(f"Running {args.warmup} warmup iterations...")
    with torch.cuda.stream(capture_stream):
        for _ in range(args.warmup):
            k_cache = k_cache_init.clone()
            v_cache = v_cache_init.clone()
            k_cache = normalize_keys(k_cache)
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
        k_cache_graph_norm = normalize_keys(k_cache_graph)
        run_cycle(
            kernels, X_graph, Wqkv, k_cache_graph_norm, v_cache_graph, capture_stream
        )

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
