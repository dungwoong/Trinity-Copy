# TileSat Mirage Benchmark Project

TileSat is a project focused on benchmarking and optimizing attention variations using the Mirage library for kernel superoptimization.
The main benchmark code lives in `benchmark/` and the Mirage library is located in `src/mirage/`.

## Build & Commands

### Python Environment

- Install dependencies: `uv sync`
- Run Python scripts: `PATH=/usr/local/cuda/bin:$PATH CUDA_VISIBLE_DEVICES=0 uv run python benchmark/mirage/<script>.py`
- Run Mirage benchmarks: `PATH=/usr/local/cuda/bin:$PATH CUDA_VISIBLE_DEVICES=0 uv run python src/mirage/benchmark/<script>.py`

### Common Benchmark Parameters

- `--file`: Checkpoint file for optimization results
- `--backend`: Backend to use (cuda/nki)
- `--warmup`: Number of warmup iterations (default: 16)
- `--profile`: Number of profiling iterations (default: 1000)
- `--save_codes`: Save generated kernel codes (default: False)

### Linting

**IMPORTANT: Always run linting after modifying code files to ensure code quality and consistency.**

- markdown: `markdownlint --fix <file>`
- python: `ruff format <file> && ruff check --fix <file>`

## Code Style

- Python: Follow PEP 8 with double quotes for strings
- Line length: 100 characters
- Use type hints where appropriate
- Document functions with docstrings
- Prefer functional programming patterns
- Use descriptive variable names
- Follow existing patterns in the codebase

## Testing

- Run benchmark tests: `uv run python benchmark/mirage/<test_file>.py`
- Profile kernel performance: Use `--profile` flag with benchmark scripts
- Verify CUDA availability before running benchmarks
- Check GPU memory usage during benchmarking

## Architecture

### Mirage Components

- **Kernel Graphs**: Define computation DAGs for optimization
- **Superoptimization**: Automatic kernel optimization using search
- **Backends**: CUDA and NKI (Neuron Kernel Interface) support
- **Attention Variations**:
  - Multi-Head Attention (MHA)
  - Multi-Query Attention (MQA)
  - Group Query Attention (GQA)
  - QK-Normalized GQA
  - RMSNorm variations

### Project Structure

- `benchmark/mirage/`: Custom benchmark implementations
- `src/mirage/`: Mirage library source code
  - `src/mirage/benchmark/`: Mirage library benchmarks
  - `src/mirage/demo/`: Demonstration scripts
  - `src/mirage/cpp_examples/`: C++ implementation examples
  - `src/mirage/python/mirage/`: Python bindings

### Creating New Benchmarks

1. Define kernel graph using `mi.new_kernel_graph()`
2. Add input tensors with appropriate dimensions
3. Build computation graph (matmul, softmax, etc.)
4. Mark outputs with `graph.mark_output()`
5. Superoptimize with appropriate config
6. Profile with CUDA events for accurate timing

## Common Patterns

### Kernel Graph Creation

```python
graph = mi.new_kernel_graph()
input_tensor = graph.new_input(dims=(...), dtype=mi.float16)
# Build computation
output = graph.operation(input_tensor)
graph.mark_output(output)
optimized = graph.superoptimize(config="attention")
```

### Benchmarking Pattern

```python
# Warmup
for _ in range(warmup_iters):
    optimized_graph(inputs=input_tensors)

# Profile
torch.cuda.synchronize()
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
starter.record()
for _ in range(profile_iters):
    optimized_graph(inputs=input_tensors)
ender.record()
torch.cuda.synchronize()
elapsed_time = starter.elapsed_time(ender)
```

## Environment Variables

- `CUDA_VISIBLE_DEVICES`: Select GPU device
- `MIRAGE_CACHE_DIR`: Directory for caching optimized kernels
- `MIRAGE_LOG_LEVEL`: Logging verbosity

## Troubleshooting

- Ensure CUDA toolkit is properly installed
- Check GPU compute capability compatibility
- Verify sufficient GPU memory for batch sizes
- Clear cache if optimization results seem stale
- Check for CUDA OOM errors and reduce batch size if needed
