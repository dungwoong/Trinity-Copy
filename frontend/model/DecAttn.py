import torch
import torch.nn as nn
import ir.AST as T
from core.from_tir import build_primfunc_nodes
from core.from_rir import build_main_func
from core.sequantialize import sequentialize_main_func
from core.to_ir import calls_to_ir, filter_identity_and_apply_alias
from utils.io_utils import format_primfunc_nodes, format_main_func, export_main_func
from utils.ir_utils import inline_elementwise_op_calls, inline_shape_op_calls, bind_main_func_calls, normalize_main_func_axes
from utils.tir_utils import to_relax, to_tir
from utils.test_utils import validate_main_func_errors
from utils.pipeline import export_model_ir

class Vanilla(nn.Module):
    def __init__(self, M, N, D, P, cache_K, cache_V, W_q=None, W_k=None, W_v=None, device=None, dtype=None):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.P = P
        self.H = N // D
        self.device = device
        self.dtype = dtype

        self.q_proj = nn.Linear(N, N, bias=False)
        self.k_proj = nn.Linear(N, N, bias=False)
        self.v_proj = nn.Linear(N, N, bias=False)

        # cache는 buffer로 등록
        self.register_buffer('cache_K', cache_K.to(device))
        self.register_buffer('cache_V', cache_V.to(device))
    
    def forward(self, X):
        # X shape: (M, N) where M=16 (seq), N=4096 (hidden)
        # Project Q, K, V separately
        q = self.q_proj(X)
        k = self.k_proj(X)
        v = self.v_proj(X)
    
        # Reshape to multi-head
        q = q.view(self.M, self.H, self.D)  # (M, H, D)
        k = k.view(self.M, self.H, self.D)  # (M, H, D)
        v = v.view(self.M, self.H, self.D)  # (M, H, D)

        # Transpose to (H, M, D) for cache update
        k = k.transpose(0, 1)  # (H, M, D)
        v = v.transpose(0, 1)  # (H, M, D)

        # Update cache - using slicing to avoid in-place operation issues
        # cache_K_new = self.cache_K.clone()
        # cache_V_new = self.cache_V.clone()
        self.cache_K[:, self.P:self.P+self.M, :] = k
        self.cache_V[:, self.P:self.P+self.M, :] = v
        cache_K_new = self.cache_K
        cache_V_new = self.cache_V

        # Transpose q to (H, M, D)
        q = q.transpose(0, 1)  # (H, M, D)

        # Attention scores: (H, M, D) @ (H, D, P+M) -> (H, M, P+M)
        scores = torch.matmul(q, cache_K_new.transpose(1, 2))
        
        # Softmax - using torch.softmax for TVM compatibility
        weights = torch.softmax(scores, dim=-1)
        
        # Apply attention: (H, M, P+M) @ (H, P+M, D) -> (H, M, D)
        output = torch.matmul(weights, cache_V_new)
        
        # Transpose back and reshape: (H, M, D) -> (M, H, D) -> (M, N)
        output = output.transpose(0, 1)  # (M, H, D)
        output = output.contiguous().view(self.M, self.H * self.D)

        return output

def build_model_and_inputs():
    import os

    M = 16
    N = 4096
    D = 128
    H = 32
    P = 1008

    device = torch.device('cpu')  # TVM export는 CPU에서 더 안정적
    dtype = torch.float32  # float16은 export 시 문제가 있을 수 있음

    X = torch.randn((M, N), device=device, dtype=dtype)
    WQ = torch.randn((N, N), device=device, dtype=dtype)
    WK = torch.randn((N, N), device=device, dtype=dtype)
    WV = torch.randn((N, N), device=device, dtype=dtype)
    K_cache = torch.randn((H, P+M, D), device=device, dtype=dtype)
    V_cache = torch.randn((H, P+M, D), device=device, dtype=dtype)

    model = Vanilla(M, N, D, P, K_cache, V_cache, WQ, WK, WV, device, dtype)

    # Relax IR로 변환
    example_inputs = X
    return {
        "model": model,
        "example_inputs": example_inputs,
        "inline_shape_op": True,
        "inline_elementwise_op": True,
        "remove_short_loop_threshold": 16,
        "decompose_nested_op_ratio": 0.0,
    }
if __name__ == "__main__":
    cfg = build_model_and_inputs()
    export_model_ir(
        cfg["model"],
        cfg["example_inputs"],
        inline_shape_op=cfg.get("inline_shape_op", True),
        inline_elementwise_op=cfg.get("inline_elementwise_op", True),
        remove_short_loop_threshold=cfg.get("remove_short_loop_threshold", 64),
        decompose_nested_op_ratio=cfg.get("decompose_nested_op_ratio", 0.0),
    )

    # parsed_funcs = parse_tir_module(tir_mod)
    # json_output_path = f"{output_dir}/dec_attn_tir_parsed.json"
    # parsed_output_path = f"{output_dir}/dec_attn_tir_parsed.txt"
    # save_pretty_tir(parsed_funcs, parsed_output_path)
