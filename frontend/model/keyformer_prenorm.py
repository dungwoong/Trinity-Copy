import os
import torch
import torch.nn as nn
import ir.AST as T
from utils.pipeline import export_model_ir

from core.from_tir import build_primfunc_nodes
from core.from_rir import build_main_func
from core.sequantialize import sequentialize_main_func
from core.to_ir import calls_to_ir, filter_identity_and_apply_alias
from utils.io_utils import format_primfunc_nodes, format_main_func, export_main_func
from utils.ir_utils import inline_elementwise_op_calls, inline_shape_op_calls, bind_main_func_calls, normalize_main_func_axes
from utils.tir_utils import to_relax, to_tir
from utils.test_utils import validate_main_func_errors


class KeyformerPreNormAttn(nn.Module):
    def __init__(self, M, H, D, P, cache_K, cache_V, tau, noise, device=None, dtype=None):
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
        self.register_buffer("tau", tau.to(device))
        self.register_buffer("noise", noise.to(device))

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

        k_cache = torch.cat([self.cache_K, k], dim=1)
        v_cache = torch.cat([self.cache_V, v], dim=1)

        c = torch.matmul(q, k_cache.permute(0, 2, 1))
        c_perturb = (c + self.noise) / self.tau

        c_exp = torch.exp(c)
        c_exp_perturb = torch.exp(c_perturb)

        c_sum = c_exp.sum(dim=2)
        c_sum_perturb = c_exp_perturb.sum(dim=2)

        c_div = c_exp / c_sum.unsqueeze(-1)
        c_div_perturb = c_exp_perturb / c_sum_perturb.unsqueeze(-1)
        c_out = c_div_perturb.sum(dim=1)

        o = torch.matmul(c_div, v_cache)
        o1 = o.permute(1, 0, 2)
        o2 = o1.contiguous().view(self.M, self.N)
        return o2, c_out


def build_model_and_inputs():
    device = torch.device("cpu")
    dtype = torch.float32

    M = 16
    H = 32
    D = 128
    P = 1024

    N = H * D
    X = torch.randn((M, N), device=device, dtype=dtype)
    K_cache = torch.randn((H, P, D), device=device, dtype=dtype)
    V_cache = torch.randn((H, P, D), device=device, dtype=dtype)
    tau = torch.tensor(1.0, device=device, dtype=dtype)
    noise = torch.randn((H, M, P + M), device=device, dtype=dtype)

    model = KeyformerPreNormAttn(
        M, H, D, P, K_cache, V_cache, tau, noise, device=device, dtype=dtype
    )

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
