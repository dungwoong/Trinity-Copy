from codegen.convert_module import convert_ir_to_triton
import argparse, torch, importlib.util, sys, json, os
import time as time_module
import matplotlib.pyplot as plt
import numpy as np

# Baseline colors (matching the paper figure)
BASELINE_COLORS = {
    'FlashInfer': '#E69F00',      # orange
    'Pytorch': '#56B4E9',          # light blue
    'FlashTensor': '#87CEEB',      # sky blue
    'TorchInductor': '#C0C0C0',    # light gray
    'TensorRT': '#F0E442',         # yellow
    'Trinity': '#90EE90',          # light green
}

BASELINES = ['FlashInfer', 'Pytorch', 'FlashTensor', 'TorchInductor', 'TensorRT', 'Trinity']
BENCHMARKS = ['vanilla', 'prenorm', 'qknorm', 'keyformer', 'roco', 'ffn']
BENCHMARK_LABELS = ['Vanilla', 'Pre-Norm', 'QK-Norm', 'KeyFormer', 'RoCo', 'SwiGLU - FFN']
MODELS = ['llama', 'falcon']
MODEL_LABELS = ['h=32,d=128', 'h=71,d=64']

# Case numbers for each model/benchmark combination (default)
CASE_NUMBERS = {
    'llama': {'vanilla': 1562, 'prenorm': 579, 'qknorm': 3690, 'keyformer': 2430, 'roco': 1620, 'ffn': 2248},
    'falcon': {'vanilla': 1562, 'prenorm': 579, 'qknorm': 1350, 'keyformer': 1799, 'roco': 4775, 'ffn': 2248},
}


def run_single_eval(model, target, case_num, device_id, baselines_to_run=None):
    """
    Run evaluation for a single benchmark configuration.
    Returns dict of {baseline_name: latency_ms}
    """
    from codegen.convert_module import convert_ir_to_triton

    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    dtype = torch.float16

    # Load model config
    with open('./model_configs.json', 'r') as f:
        model_configs = json.load(f)

    config = model_configs[model]
    M = config['M']
    D = config['D']
    N = config['N']
    P = config['P'] - M
    H = config['H']
    N4 = config['N4']

    constants = {'M': M, 'D': D, 'N': N, 'P': P, 'H': H, 'N4': N4}

    tensor_shapes = {
        'X': ('M', 'N'), 'X2': ('M',), 'X_norm': ('M', 'N'),
        'WQ': ('N', 'N'), 'WK': ('N', 'N'), 'WV': ('N', 'N'),
        'Q1': ('M', 'N'), 'K1': ('M', 'N'), 'V1': ('M', 'N'),
        'Q2': ('M', 'H', 'D'), 'K2': ('M', 'H', 'D'), 'V2': ('M', 'H', 'D'),
        'K_cache': ('H', 'P+M', 'D'), 'V_cache': ('H', 'P+M', 'D'),
        'Q': ('H', 'M', 'D'), 'K': ('H', 'M', 'D'), 'V': ('H', 'M', 'D'),
        'O': ('H', 'M', 'D'), 'O1': ('M', 'H', 'D'), 'O2': ('M', 'N'),
        'C': ('H', 'M', 'P+M'), 'C_exp': ('H', 'M', 'P+M'), 'C_div': ('H', 'M', 'P+M'),
        'C_sum': ('H', 'M'), 'noise': ('H', 'M', 'P+M'), 'C_perturb': ('H', 'M', 'P+M'),
        'C_exp_perturb': ('H', 'M', 'P+M'), 'C_sum_perturb': ('H', 'M'),
        'C_div_perturb': ('H', 'M', 'P+M'), 'C_out': ('H', 'P+M'),
        'C_out1': ('H', 'P+M'), 'C_out2': ('H', 'P+M'),
        'Q_norm': ('H', 'M', 'D'), 'K_norm': ('H', 'M', 'D'),
        'WO': ('N', 'N'), 'attn_O1': ('M', 'N'), 'attn_O2': ('M', 'N'),
        'attn_O3': ('M'), 'attn_O_norm': ('M', 'N'),
        'WFF1a': ('N', 'N4'), 'WFF1b': ('N', 'N4'),
        'FF1a': ('M', 'N4'), 'FF1b': ('M', 'N4'), 'FF1b_silu': ('M', 'N4'),
        'FF1': ('M', 'N4'), 'FF2': ('M', 'N'), 'WFF2': ('N4', 'N'),
    }

    # Set random seed
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Initialize tensors
    std = 0.01
    X = torch.randn((M, N), device=device, dtype=dtype) * std
    WQ = torch.randn((N, N), device=device, dtype=dtype) * std
    WK = torch.randn((N, N), device=device, dtype=dtype) * std
    WV = torch.randn((N, N), device=device, dtype=dtype) * std
    K_cache = torch.randn((H, P+M, D), device=device, dtype=dtype) * std
    V_cache = torch.randn((H, P+M, D), device=device, dtype=dtype) * std
    K_cache_flashinfer = K_cache.clone().transpose(0, 1).contiguous()
    V_cache_flashinfer = V_cache.clone().transpose(0, 1).contiguous()
    O2 = torch.zeros((M, N), device=device, dtype=dtype)
    C_exp = torch.zeros((H, M, P+M), device=device, dtype=torch.float32)
    C_out1 = torch.zeros((H, P+M), device=device, dtype=dtype)
    C_out2 = torch.zeros((H, P+M), device=device, dtype=dtype)
    C = torch.zeros((H, M, P+M), device=device, dtype=dtype)
    C_exp_perturb = torch.zeros((H, M, P+M), device=device, dtype=torch.float32)
    C_out = torch.zeros((H, P+M), device=device, dtype=dtype)
    noise = torch.randn((H, M, P+M), device=device, dtype=dtype) * std

    std = 0.001
    if target != "ffn":
        O2 = torch.randn(M, N, dtype=dtype, device=device) * std
    FF1 = torch.zeros(M, N4, dtype=dtype, device=device)
    FF2 = torch.zeros(M, N4, dtype=dtype, device=device)
    WFF1a = torch.randn(N, N4, dtype=dtype, device=device) * std
    WFF1b = torch.randn(N, N4, dtype=dtype, device=device) * std
    WFF2 = torch.randn(N4, N, dtype=dtype, device=device) * std
    WO = torch.randn(N, N, dtype=dtype, device=device) * std
    attn_O2 = torch.zeros(M, N, dtype=dtype, device=device)

    ITER = 1000
    results = {}

    case_file = f"./results/{target}/{target}_{model}_case{case_num}.txt"
    output_file = f"./results/{target}/{target}_{model}_benchmark{case_num}.py"
    module_name = f"{target}_{model}_best"

    # Load baselines
    trt, ti, fi, ft = None, None, None, None
    try:
        match target:
            case "vanilla":
                from baselines import Vanilla, TensorRT_Vanilla, FlashInfer_Vanilla
                trt = TensorRT_Vanilla(M, N, D, H, K_cache.clone(), V_cache.clone(), P, WQ, WK, WV, device, dtype)
                ti = Vanilla(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV, device, dtype)
                fi = FlashInfer_Vanilla(M, N, D, P, K_cache_flashinfer.clone(), V_cache_flashinfer.clone(), WQ, WK, WV, device, dtype)
                from flashtensor.h100_vanilla import bench_vanilla
                ft = bench_vanilla
            case "prenorm":
                from baselines import PreNorm, TensorRT_PreNorm
                trt = TensorRT_PreNorm(M, N, D, H, K_cache.clone(), V_cache.clone(), P, WQ, WK, WV, device, dtype)
                ti = PreNorm(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV, device, dtype)
                from flashtensor.h100_prenorm import bench_prenorm
                ft = bench_prenorm
            case "keyformer":
                from baselines import KeyFormer, TensorRT_KeyFormer
                trt = TensorRT_KeyFormer(M, N, D, H, K_cache.clone(), V_cache.clone(), P, noise, WQ, WK, WV, device, dtype)
                ti = KeyFormer(M, N, D, P, noise, K_cache.clone(), V_cache.clone(), WQ, WK, WV, device, dtype)
                from flashtensor.h100_kf import bench_kf
                ft = bench_kf
            case "qknorm":
                from baselines import QKNorm, TensorRT_QKNorm
                trt = TensorRT_QKNorm(M, N, D, H, K_cache.clone(), V_cache.clone(), P, WQ, WK, WV, device, dtype)
                ti = QKNorm(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV, device, dtype)
                from flashtensor.h100_qknorm import bench_qknorm
                ft = bench_qknorm
            case "roco":
                from baselines import RoCo, TensorRT_RoCo
                trt = TensorRT_RoCo(M, N, D, H, K_cache.clone(), V_cache.clone(), P, WQ, WK, WV, device, dtype)
                ti = RoCo(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV, device, dtype)
                from flashtensor.h100_roco import bench_roco
                ft = bench_roco
            case "ffn":
                from baselines import FFN, TensorRT_FFN
                trt = TensorRT_FFN(M, N, N4, WO=WO, WFF1a=WFF1a, WFF1b=WFF1b, WFF2=WFF2, device=device, dtype=dtype)
                ti = FFN(M, N, N4, WO=WO, WFF1a=WFF1a, WFF1b=WFF1b, WFF2=WFF2, device=device, dtype=dtype)
    except Exception as e:
        print(f"  Warning: Failed to load baselines: {e}")

    # Run Trinity
    if baselines_to_run is None or 'Trinity' in baselines_to_run:
        try:
            with open(case_file, "r") as f:
                ir = f.read().strip()
            triton_code = convert_ir_to_triton(ir, tensor_shapes, constants)
            with open(output_file, "w") as f:
                f.write(triton_code)

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
            blocks = {'block_k': 0, 'block_n': 0, 'block_p': 0}
            args = []
            for param in tensor_params:
                if param in tensors:
                    args.append(tensors[param])
            for param in block_params:
                if param in blocks:
                    args.append(blocks[param])

            for _ in range(10):
                forward(*args)

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(ITER):
                forward(*args)
            end_event.record()
            torch.cuda.synchronize()
            results['Trinity'] = start_event.elapsed_time(end_event) / ITER
        except Exception as e:
            print(f"  Trinity failed: {e}")
            results['Trinity'] = None

    # Run other baselines
    inputs = (O2, X) if target == "ffn" else (X,)

    # TensorRT
    if trt and (baselines_to_run is None or 'TensorRT' in baselines_to_run):
        try:
            trt.half()
            with torch.no_grad():
                for _ in range(10):
                    trt(*inputs)
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                for _ in range(ITER):
                    trt(*inputs)
                end_event.record()
                torch.cuda.synchronize()
                results['TensorRT'] = start_event.elapsed_time(end_event) / ITER
        except Exception as e:
            print(f"  TensorRT failed: {e}")
            results['TensorRT'] = None

    # Pytorch Eager
    if ti and (baselines_to_run is None or 'Pytorch' in baselines_to_run):
        try:
            ti = ti.eval()
            with torch.no_grad():
                for _ in range(10):
                    ti(*inputs)
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                for _ in range(ITER):
                    ti(*inputs)
                end_event.record()
                torch.cuda.synchronize()
                results['Pytorch'] = start_event.elapsed_time(end_event) / ITER
        except Exception as e:
            print(f"  Pytorch failed: {e}")
            results['Pytorch'] = None

    # TorchInductor
    if ti and (baselines_to_run is None or 'TorchInductor' in baselines_to_run):
        try:
            mode = "max-autotune-no-cudagraphs"
            compiled_model = torch.compile(ti, backend="inductor", mode=mode, fullgraph=True)
            for _ in range(10):
                compiled_model(*inputs)
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(ITER):
                compiled_model(*inputs)
            end_event.record()
            torch.cuda.synchronize()
            results['TorchInductor'] = start_event.elapsed_time(end_event) / ITER
        except Exception as e:
            print(f"  TorchInductor failed: {e}")
            results['TorchInductor'] = None

    # FlashInfer
    if fi and (baselines_to_run is None or 'FlashInfer' in baselines_to_run):
        try:
            fi.half()
            fi = fi.eval()
            with torch.no_grad():
                for _ in range(10):
                    fi(X)
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                for _ in range(ITER):
                    fi(X)
                end_event.record()
                torch.cuda.synchronize()
                results['FlashInfer'] = start_event.elapsed_time(end_event) / ITER
        except Exception as e:
            print(f"  FlashInfer failed: {e}")
            results['FlashInfer'] = None

    # FlashTensor
    if ft and (baselines_to_run is None or 'FlashTensor' in baselines_to_run):
        try:
            results['FlashTensor'] = ft(model, M, N, P, D, H, device, dtype)
        except Exception as e:
            print(f"  FlashTensor failed: {e}")
            results['FlashTensor'] = None

    return results


def plot_figure5(all_results, output_path='./figure/figure5.png'):
    """
    Plot Figure 5: Normalized inference latency across various benchmarks.
    """
    fig, axes = plt.subplots(1, 6, figsize=(18, 4))

    bar_width = 0.08

    for col, (bench, bench_label) in enumerate(zip(BENCHMARKS, BENCHMARK_LABELS)):
        ax = axes[col]

        for model_idx, model in enumerate(MODELS):
            # Get Trinity time for normalization
            data = all_results.get(bench, {}).get(model, {})
            trinity_time = data.get('Trinity')

            if trinity_time is None or trinity_time == 0:
                trinity_time = 1  # Avoid division by zero

            # Calculate normalized latencies
            normalized = []
            for baseline in BASELINES:
                time = data.get(baseline)
                if time is not None:
                    normalized.append(time / trinity_time)
                else:
                    normalized.append(0)  # Missing data

            # Plot bars for this model
            x_positions = np.arange(len(BASELINES))
            offset = (model_idx - 0.5) * (len(BASELINES) * bar_width + 0.1)
            x = x_positions * bar_width + offset + model_idx * 0.5

            colors = [BASELINE_COLORS[b] for b in BASELINES]
            ax.bar(x, normalized, bar_width * 0.9, color=colors, edgecolor='black', linewidth=0.3)

            # Mark missing baselines with 'X'
            for i, (val, baseline) in enumerate(zip(normalized, BASELINES)):
                if val == 0:
                    ax.text(x[i], 0.1, 'X', ha='center', va='bottom', fontsize=6, color='red')
                elif val > 3:
                    ax.text(x[i], 2.9, f'{val:.2f}', ha='center', va='bottom', fontsize=5, color='#56B4E9')

        # Set axis properties
        ax.set_ylim(0, 3)
        ax.set_xlim(-0.3, 1.3)
        ax.axhline(y=1, color='gray', linestyle='--', linewidth=0.5)

        ax.set_xlabel(f'{MODEL_LABELS[0]}    {MODEL_LABELS[1]}\n{bench_label}', fontsize=9)
        if col == 0:
            ax.set_ylabel('Normalized Latency', fontsize=10)

        ax.set_xticks([])
        ax.tick_params(axis='y', labelsize=8)

    # Legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=BASELINE_COLORS[b], edgecolor='black', linewidth=0.3, label=b)
                       for b in BASELINES]
    legend_elements[-1] = plt.Rectangle((0,0),1,1, facecolor=BASELINE_COLORS['Trinity'], edgecolor='black', linewidth=0.3, label='Trinity (Ours)')

    fig.legend(handles=legend_elements, loc='upper center', ncol=8, fontsize=9,
               bbox_to_anchor=(0.5, 1.05), frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure 5 saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=0, help="Device number")
    parser.add_argument("--trials", type=int, default=3, help="Number of trials")
    parser.add_argument("--load", type=str, default=None, help="Load existing results from JSON")
    args = parser.parse_args()

    device_id = args.d
    num_trials = args.trials

    os.makedirs('./figure', exist_ok=True)

    # Load existing results or initialize
    if args.load and os.path.exists(args.load):
        with open(args.load, 'r') as f:
            all_results = json.load(f)
        print(f"Loaded existing results from {args.load}")
    else:
        all_results = {}

    print("Running benchmarks...")
    print("=" * 60)

    for bench in BENCHMARKS:
        if bench not in all_results:
            all_results[bench] = {}

        for model in MODELS:
            case_num = CASE_NUMBERS[model][bench]
            print(f"\n{bench.upper()} - {model.upper()} (case {case_num})")

            trial_results = {b: [] for b in BASELINES}

            for trial in range(num_trials):
                print(f"  Trial {trial + 1}/{num_trials}...", end=" ")
                results = run_single_eval(model, bench, case_num, device_id)

                for baseline in BASELINES:
                    if results.get(baseline) is not None:
                        trial_results[baseline].append(results[baseline])

                print("done")
                time_module.sleep(2)

            # Average results
            avg_results = {}
            for baseline in BASELINES:
                if trial_results[baseline]:
                    avg_results[baseline] = sum(trial_results[baseline]) / len(trial_results[baseline])
                else:
                    avg_results[baseline] = None

            all_results[bench][model] = avg_results

            # Print summary for this config
            print(f"  Results:")
            for baseline, time in avg_results.items():
                if time is not None:
                    print(f"    {baseline}: {time:.4f} ms")

    # Save results
    output_json = './figure/figure5_results.json'
    with open(output_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_json}")

    # Generate plot
    plot_figure5(all_results, './figure/figure5.png')

    print("\nDone!")


if __name__ == "__main__":
    main()
