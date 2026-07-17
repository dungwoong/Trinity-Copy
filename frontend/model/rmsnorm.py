import os
import torch
import torch.nn as nn
import ir.AST as T
from utils.pipeline import export_model_ir

from core.from_tir import build_primfunc_nodes
from core.from_rir import build_main_func
from core.sequantialize import sequentialize_main_func
from utils.io_utils import export_main_func
from utils.ir_utils import inline_elementwise_op_calls, inline_shape_op_calls, bind_main_func_calls, normalize_main_func_axes
from utils.tir_utils import to_relax, to_tir
from utils.test_utils import validate_main_func_errors


class RMSNormQKV(nn.Module):
    def __init__(self, M, H, D, device=None, dtype=None):
        super().__init__()
        self.M = M
        self.H = H
        self.D = D
        self.N = H * D
        self.device = device
        self.dtype = dtype

        self.q_proj = nn.Linear(self.N, self.N, bias=False)
        self.k_proj = nn.Linear(self.N, self.N, bias=False)
        self.v_proj = nn.Linear(self.N, self.N, bias=False)

    def forward(self, X):
        x2 = (X * X).sum(dim=1)
        x_norm = X / torch.sqrt(x2 / self.N).unsqueeze(1)

        q1 = self.q_proj(x_norm)
        k1 = self.k_proj(x_norm)
        v1 = self.v_proj(x_norm)
        return q1, k1, v1


def build_model_and_inputs():
    device = torch.device("cpu")
    dtype = torch.float32

    M = 16
    H = 71
    D = 64

    N = H * D
    X = torch.randn((M, N), device=device, dtype=dtype)

    model = RMSNormQKV(M, H, D, device=device, dtype=dtype)

    print("Model created. Converting to Relax IR...")
    example_inputs = X
    return {
        "model": model,
        "example_inputs": example_inputs,
        "inline_shape_op": True,
        "inline_elementwise_op": True,
        "remove_short_loop_threshold": 16,
        "decompose_nested_op_ratio": 1,
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
