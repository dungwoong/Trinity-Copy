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


class RMSAttnGQA(nn.Module):
    def __init__(self, B, QH, KVH, D, P, cache_K, cache_V, device=None, dtype=None):
        super().__init__()
        self.B = B
        self.QH = QH
        self.KVH = KVH
        self.D = D
        self.P = P
        self.N = QH * D
        self.KN = KVH * D
        self.device = device
        self.dtype = dtype

        self.q_proj = nn.Linear(self.N, self.N, bias=False)
        self.k_proj = nn.Linear(self.N, self.KN, bias=False)
        self.v_proj = nn.Linear(self.N, self.KN, bias=False)

        self.register_buffer("cache_K", cache_K.to(device))
        self.register_buffer("cache_V", cache_V.to(device))

    def forward(self, X):
        x2 = (X * X).sum(dim=1)
        x_norm = X / torch.sqrt(x2 / self.N).unsqueeze(1)

        q1 = self.q_proj(x_norm)
        k1 = self.k_proj(x_norm)
        v1 = self.v_proj(x_norm)

        q2 = q1.view(self.B, 1, self.QH, self.D)
        k2 = k1.view(self.B, 1, self.KVH, self.D)
        v2 = v1.view(self.B, 1, self.KVH, self.D)

        q = q2.permute(0, 2, 1, 3)
        k_cache = torch.cat([self.cache_K, k2.permute(0, 1, 2, 3)], dim=1)
        v_cache = torch.cat([self.cache_V, v2.permute(0, 1, 2, 3)], dim=1)

        if self.QH != self.KVH:
            repeat = self.QH // self.KVH
            k_cache = k_cache.repeat_interleave(repeat, dim=2)
            v_cache = v_cache.repeat_interleave(repeat, dim=2)

        c = torch.matmul(q, k_cache.permute(0, 2, 3, 1))
        c_exp = torch.exp(c)
        c_sum = c_exp.sum(dim=3)
        c_div = c_exp / c_sum.unsqueeze(3)
        o = torch.matmul(c_div, v_cache.permute(0, 2, 1, 3))

        o1 = o.permute(0, 2, 1, 3)
        o2 = o1.contiguous().view(self.B, self.N)
        return o2


def build_model_and_inputs():
    device = torch.device("cpu")
    dtype = torch.float32

    B = 1
    QH = 32
    KVH = 8
    D = 128
    P = 528

    N = QH * D
    X = torch.randn((B, N), device=device, dtype=dtype)
    K_cache = torch.randn((B, P, KVH, D), device=device, dtype=dtype)
    V_cache = torch.randn((B, P, KVH, D), device=device, dtype=dtype)

    model = RMSAttnGQA(B, QH, KVH, D, P, K_cache, V_cache, device=device, dtype=dtype)

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
