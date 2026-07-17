from codegen.convert_module import convert_ir_to_triton
import argparse, torch, importlib.util, sys, json
import time as time_module
import matplotlib.pyplot as plt
import numpy as np


def plot_bar_chart(data, title, output_path, best_color='#DAA520'):
    """Plot latency bar chart."""
    labels = list(data.keys())
    times = list(data.values())
    best_idx = np.argmin(times)

    fig, ax = plt.subplots(figsize=(5, 4))

    colors = ['#D3D3D3'] * len(labels)
    colors[best_idx] = best_color

    ax.bar(labels, times, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('High performance kernels', fontsize=11)
    ax.set_ylabel('Latency (us)', fontsize=11)

    y_min = min(times) * 0.95
    y_max = max(times) * 1.05

    ax.set_ylim(y_min, y_max)

    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {output_path}\n")


def get_gpu_name():
    """Get GPU name from CUDA device."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"


def run_single_benchmark(target, case, model, device_id, model_configs):
    """Run a single benchmark and return the latency in microseconds."""
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    dtype = torch.float16

    output_file = f"figure67/{target}/{target}_{case}.py"
    module_name = f"{target}_{case}_{model}"

    config = model_configs[model]
    M = config['M']
    D = config['D']
    N = config['N']
    P = config['P'] - M
    H = config['H']
    N4 = config['N4']

    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # --------------- Init for Attention ---------------------
    std = 0.01
    X = torch.randn((M, N), device=device, dtype=dtype) * std
    WQ = torch.randn((N, N), device=device, dtype=dtype) * std
    WK = torch.randn((N, N), device=device, dtype=dtype) * std
    WV = torch.randn((N, N), device=device, dtype=dtype) * std
    K_cache = torch.randn((H, P+M, D), device=device, dtype=dtype) * std
    V_cache = torch.randn((H, P+M, D), device=device, dtype=dtype) * std
    O2 = torch.zeros((M, N), device=device, dtype=dtype) * std

    # --------------- Additional init for RoCo ---------------------
    C_exp = torch.zeros((H, M, P+M), device=device, dtype=dtype)
    C_out1 = torch.zeros((H, P+M), device=device, dtype=dtype) * std
    C_out2 = torch.zeros((H, P+M), device=device, dtype=dtype) * std

    # --------------- Additional init for KeyFormer ---------------------
    C = torch.zeros((H, M, P+M), device=device, dtype=dtype)
    C_exp_perturb = torch.zeros((H, M, P+M), device=device, dtype=dtype)
    C_out = torch.zeros((H, P+M), device=device, dtype=dtype)
    noise = torch.randn((H, M, P+M), device=device, dtype=dtype) * std

    # --------------- Init for FFN ---------------------
    std = 0.001
    if target == "ffn":
        O2 = O2
    else:
        O2 = torch.randn(M, N, dtype=dtype, device=device) * std
    FF1 = torch.zeros(M, N4, dtype=dtype, device=device)
    FF2 = torch.zeros(M, N4, dtype=dtype, device=device)
    WFF1a = torch.randn(N, N4, dtype=dtype, device=device) * std
    WFF1b = torch.randn(N, N4, dtype=dtype, device=device) * std
    WFF2 = torch.randn(N4, N, dtype=dtype, device=device) * std
    WO = torch.randn(N, N, dtype=dtype, device=device) * std
    attn_O2 = torch.zeros(M, N, dtype=dtype, device=device)

    ITER = 1000

    # --------------- Load and run Trinity ---------------------
    spec = importlib.util.spec_from_file_location(module_name, output_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    forward = getattr(module, "forward")

    tensor_params = getattr(module, 'TENSOR_PARAMS')
    block_params = getattr(module, 'BLOCK_PARAMS')
    tensors = {
        'X': X, 'WQ': WQ, 'WK': WK, 'WV': WV,
        'K_cache': K_cache, 'V_cache': V_cache,
        'O2': O2, 'C': C, 'C_exp': C_exp, 'noise': noise,
        'C_exp_perturb': C_exp_perturb,
        'C_out': C_out, 'C_out1': C_out1, 'C_out2': C_out2,
        'WO': WO, 'attn_O2': attn_O2,
        'WFF1a': WFF1a, 'WFF1b': WFF1b,
        'FF1': FF1, 'FF2': FF2, 'WFF2': WFF2,
    }
    blocks = {
        'block_k': 0, 'block_n': 0, 'block_p': 0
    }
    args = []
    for param in tensor_params:
        if param in tensors:
            args.append(tensors[param])
        else:
            raise ValueError(f"Unknown tensor parameter: {param}")
    for param in block_params:
        if param in blocks:
            args.append(blocks[param])
        else:
            raise ValueError(f"Unknown block parameter: {param}")

    # Warmup
    for _ in range(10):
        forward(*args)

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(ITER):
        forward(*args)
    end_event.record()
    torch.cuda.synchronize()

    latency_ms = start_event.elapsed_time(end_event) / ITER
    latency_us = latency_ms * 1000  # Convert to microseconds

    return latency_us


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=str, default="llama", help="Input model type")
    parser.add_argument("--d", type=int, default=0, help="Device number")
    parser.add_argument("--trials", type=int, default=3, help="Number of trials per case")
    args = parser.parse_args()

    model = args.m
    device_id = args.d
    num_trials = args.trials

    # Get GPU name from device
    gpu_name = get_gpu_name()
    print(f"Detected GPU: {gpu_name}")

    # Load model config
    with open('./model_configs.json', 'r') as f:
        model_configs = json.load(f)

    if model not in model_configs:
        raise ValueError(f"Unknown model: {model}. Available: {list(model_configs.keys())}")

    # Define cases
    keyformer_cases = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    roco_cases = ['A', 'B', 'C', 'D', 'E']

    results = {
        'keyformer': {},
        'roco': {}
    }

    # Run Keyformer benchmarks
    print("=" * 60)
    print("Running Keyformer benchmarks...")
    print("=" * 60)
    for case in keyformer_cases:
        times = []
        for trial in range(num_trials):
            print(f"[Keyformer {case}] Trial {trial + 1}/{num_trials}...", end=" ")
            latency = run_single_benchmark('keyformer', case, model, device_id, model_configs)
            times.append(latency)
            print(f"{latency:.2f} us")
            time_module.sleep(2)  # Brief pause between trials

        avg_time = sum(times) / len(times)
        results['keyformer'][case] = avg_time
        print(f"[Keyformer {case}] Average: {avg_time:.2f} us\n")
    plot_bar_chart(results['keyformer'], 'KeyFormer', './figure/figure6_keyformer.png', '#DAA520')
    # Run RoCo benchmarks
    print("=" * 60)
    print("Running RoCo benchmarks...")
    print("=" * 60)
    for case in roco_cases:
        times = []
        for trial in range(num_trials):
            print(f"[RoCo {case}] Trial {trial + 1}/{num_trials}...", end=" ")
            latency = run_single_benchmark('roco', case, model, device_id, model_configs)
            times.append(latency)
            print(f"{latency:.2f} us")
            time_module.sleep(2)  # Brief pause between trials

        avg_time = sum(times) / len(times)
        results['roco'][case] = avg_time
        print(f"[RoCo {case}] Average: {avg_time:.2f} us\n")
    plot_bar_chart(results['roco'], 'RoCo', './figure/figure7_roco.png', '#4A90D9')
    
    # Print summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nKeyformer Results (us):")
    for case, time in results['keyformer'].items():
        best_marker = " <-- Best" if time == min(results['keyformer'].values()) else ""
        print(f"  {case}: {time:.2f}{best_marker}")

    print("\nRoCo Results (us):")
    for case, time in results['roco'].items():
        best_marker = " <-- Best" if time == min(results['roco'].values()) else ""
        print(f"  {case}: {time:.2f}{best_marker}")

    # Save results to JSON
    output_json = './figure/figure67_results.json'
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_json}")

    print("\nDone!")


if __name__ == "__main__":
    main()