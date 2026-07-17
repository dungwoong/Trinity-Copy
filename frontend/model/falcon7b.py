import os
import torch
import torch.nn as nn
import ir.AST as T
from utils.pipeline import export_model_ir


class RMSAttn(nn.Module):
    def __init__(self, M, H, D, P, cache_K, cache_V, device=None, dtype=None):
        super().__init__()
        self.M = M
        self.H = H
        self.D = D
        self.P = P
        self.N = H * D
        self.device = device
        self.dtype = dtype

        self.q_proj = nn.Linear(self.N, self.N, bias=False)
        self.k_proj = nn.Linear(self.N, self.N, bias=False)
        self.v_proj = nn.Linear(self.N, self.N, bias=False)

        self.register_buffer("cache_K", cache_K.to(device))
        self.register_buffer("cache_V", cache_V.to(device))

    def forward(self, X):
        x2 = (X * X).sum(dim=1)
        x_norm = X / torch.sqrt(x2 / self.N).unsqueeze(1)

        q1 = self.q_proj(x_norm)
        k1 = self.k_proj(x_norm)
        v1 = self.v_proj(x_norm)

        q2 = q1.view(self.M, self.H, self.D)
        k2 = k1.view(self.M, self.H, self.D)
        v2 = v1.view(self.M, self.H, self.D)

        q = q2.permute(1, 0, 2)
        k = k2.permute(1, 0, 2)
        v = v2.permute(1, 0, 2)

        cache_k = torch.cat([self.cache_K, k], dim=1)
        cache_v = torch.cat([self.cache_V, v], dim=1)

        c = torch.matmul(q, cache_k.permute(0, 2, 1))
        c_exp = torch.exp(c)
        c_sum = c_exp.sum(dim=2)
        c_div = c_exp / c_sum.unsqueeze(-1)
        o = torch.matmul(c_div, cache_v)

        o1 = o.permute(1, 0, 2)
        o2 = o1.contiguous().view(self.M, self.N)
        return o2


def build_model_and_inputs():
    device = torch.device("cpu")
    dtype = torch.float32

    # Keep small defaults for runtime
    M = 16
    H = 71
    D = 64
    P = 528

    N = H * D
    X = torch.randn((M, N), device=device, dtype=dtype)
    K_cache = torch.randn((H, P, D), device=device, dtype=dtype)
    V_cache = torch.randn((H, P, D), device=device, dtype=dtype)

    model = RMSAttn(M, H, D, P, K_cache, V_cache, device=device, dtype=dtype)

    print("Model created. Converting to Relax IR...")
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
