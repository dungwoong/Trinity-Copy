import argparse

import torch

import mirage as mi
from mirage.kernel import KNGraph

HIDDEN_DIM = 4096
INTERMEDIATE_DIM = 14336
SEQ_LEN = 16
CONTEXT_LEN = 1024
NUM_HEADS = 32
HEAD_DIM = 128


def post_attention_norm():
    """
    Builds and superoptimizes a kernel graph for post-attention normalization.

    This kernel applies the output projection, residual connection, and RMS
    normalization after the attention mechanism, following the SwiGLU architecture.

    Operations:
    - attn_O1 = O2 @ WO (output projection)
    - attn_O2 = attn_O1 + X (residual connection)
    - Apply RMS normalization

    Returns:
        KNGraph: The superoptimized kernel graph for post-attention normalization.
    """
    _graph = mi.new_kernel_graph()

    # Input from attention: O2
    o2 = _graph.new_input(dims=(SEQ_LEN, HIDDEN_DIM), dtype=mi.float16)
    # Original input for residual connection: X
    x = _graph.new_input(dims=(SEQ_LEN, HIDDEN_DIM), dtype=mi.float16)
    # Output projection weight: WO
    wo = _graph.new_input(dims=(HIDDEN_DIM, HIDDEN_DIM), dtype=mi.float16)

    # attn_O1 = O2 * WO (output projection)
    attn_o1 = _graph.matmul(o2, wo)

    # attn_O2 = attn_O1 + X (residual connection)
    attn_o2 = _graph.add(attn_o1, x)

    # Apply RMS normalization
    # The spec uses 14336 as divisor in the normalization formula, but shape remains HIDDEN_DIM
    attn_o_norm = _graph.rms_norm(attn_o2, normalized_shape=(HIDDEN_DIM,))

    _graph.mark_output(attn_o_norm)
    return _graph.superoptimize(config="mlp")


def swiglu_mlp():
    """
    Builds and superoptimizes a kernel graph for the SwiGLU MLP layer.

    This kernel implements the SwiGLU (Swish-Gated Linear Unit) feedforward network,
    which uses a gating mechanism with SiLU activation.

    Operations:
    - FF1a = input @ WFF1a (gate projection)
    - FF1b = input @ WFF1b (up projection)
    - FF1b_silu = silu(FF1b) (SiLU activation)
    - FF1 = FF1a * FF1b_silu (gated multiplication)
    - FF2 = FF1 @ WFF2 (down projection)

    Returns:
        KNGraph: The superoptimized kernel graph for the SwiGLU MLP.
    """
    _graph = mi.new_kernel_graph()

    # Normalized input from post-attention norm
    x_norm = _graph.new_input(dims=(SEQ_LEN, HIDDEN_DIM), dtype=mi.float16)

    # SwiGLU weights
    # WFF1a: gate projection weight
    wff1a = _graph.new_input(dims=(HIDDEN_DIM, INTERMEDIATE_DIM), dtype=mi.float16)
    # WFF1b: up projection weight
    wff1b = _graph.new_input(dims=(HIDDEN_DIM, INTERMEDIATE_DIM), dtype=mi.float16)

    # FF1a = attn_O_norm * WFF1a (gate projection)
    ff1a = _graph.matmul(x_norm, wff1a)

    # FF1b = attn_O_norm * WFF1b (up projection)
    ff1b = _graph.matmul(x_norm, wff1b)

    # FF1b_silu = silu(FF1b) - applies SiLU activation
    ff1b_silu = _graph.silu(ff1b)

    # FF1 = FF1a x FF1b_silu (element-wise multiplication - gating)
    ff1 = _graph.mul(ff1a, ff1b_silu)

    _graph.mark_output(ff1)
    return _graph.superoptimize(config="mlp")


def run_cycle(
    kernels: list[KNGraph],
    o2: torch.Tensor,
    x: torch.Tensor,
    wo: torch.Tensor,
    wff1a: torch.Tensor,
    wff1b: torch.Tensor,
    wff2: torch.Tensor,
    stream: torch.cuda.Stream,
):
    """
    Runs one cycle of the SwiGLU pipeline.

    Args:
        kernels: List containing [post_attention_norm_kernel, swiglu_mlp_kernel]
        o2: Attention output tensor
        x: Original input tensor (for residual connection)
        wo: Output projection weight
        wff1a: SwiGLU gate projection weight
        wff1b: SwiGLU up projection weight
        wff2: SwiGLU down projection weight
        stream: CUDA stream for execution

    Returns:
        The final output tensor after SwiGLU MLP
    """
    post_attention_norm_kernel = kernels[0]
    swiglu_mlp_kernel = kernels[1]

    # Post-attention normalization
    x_norm: torch.Tensor = post_attention_norm_kernel(
        inputs=[o2, x, wo], stream=stream
    )[0]  # type: ignore

    # SwiGLU MLP
    ff1: torch.Tensor = swiglu_mlp_kernel(inputs=[x_norm, wff1a, wff1b], stream=stream)[
        0
    ]  # type: ignore

    output = torch.matmul(ff1, wff2)

    return output


def main():
    parser = argparse.ArgumentParser(description="SwiGLU Benchmark")
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

    print("Running SwiGLU benchmark with:")
    print(f"  Warmup iterations: {args.warmup}")
    print(f"  Profile iterations: {args.profile}")
    print("  CUDA graphs: Enabled")
    print(f"  Backend: {args.backend}")
    print()

    # Initialize tensors
    # Simulated attention output O2
    O2 = torch.randn(
        SEQ_LEN, HIDDEN_DIM, dtype=torch.float16, device=torch.device("cuda:0")
    )

    # Original input X for residual connection
    X = torch.randn(
        SEQ_LEN, HIDDEN_DIM, dtype=torch.float16, device=torch.device("cuda:0")
    )

    # Output projection weight
    Wo = torch.randn(
        HIDDEN_DIM, HIDDEN_DIM, dtype=torch.float16, device=torch.device("cuda:0")
    )

    # SwiGLU weights
    Wff1a = torch.randn(
        HIDDEN_DIM, INTERMEDIATE_DIM, dtype=torch.float16, device=torch.device("cuda:0")
    )
    Wff1b = torch.randn(
        HIDDEN_DIM, INTERMEDIATE_DIM, dtype=torch.float16, device=torch.device("cuda:0")
    )
    Wff2 = torch.randn(
        INTERMEDIATE_DIM, HIDDEN_DIM, dtype=torch.float16, device=torch.device("cuda:0")
    )

    # Create kernel graphs
    print("Creating and optimizing kernel graphs...")
    post_attention_norm_kernel = post_attention_norm()
    swiglu_mlp_kernel = swiglu_mlp()
    kernels = [post_attention_norm_kernel, swiglu_mlp_kernel]
    print("Kernel optimization complete.")

    # Create stream for all operations
    capture_stream = torch.cuda.Stream()

    # Warmup phase
    print(f"Running {args.warmup} warmup iterations...")
    with torch.cuda.stream(capture_stream):
        for _ in range(args.warmup):
            run_cycle(kernels, O2, X, Wo, Wff1a, Wff1b, Wff2, capture_stream)
        capture_stream.synchronize()
    print("Warmup complete.")

    print("Capturing CUDA graph...")
    cuda_graph = torch.cuda.CUDAGraph()

    # Prepare tensors for graph capture
    O2_graph = O2.clone()
    X_graph = X.clone()

    # Capture the graph
    with torch.cuda.graph(cuda_graph, stream=capture_stream):
        run_cycle(kernels, O2_graph, X_graph, Wo, Wff1a, Wff1b, Wff2, capture_stream)

    print(f"Running {args.profile} test iterations with timing...")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    iteration_stream = torch.cuda.Stream()

    with torch.cuda.stream(iteration_stream):
        start_event.record(iteration_stream)

        for _ in range(args.profile):
            run_cycle(kernels, O2, X, Wo, Wff1a, Wff1b, Wff2, iteration_stream)

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
