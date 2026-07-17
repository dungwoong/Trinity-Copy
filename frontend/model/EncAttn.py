import torch
import torch.nn as nn
import tvm
import ir.AST as T
from core.from_tir import build_primfunc_nodes
from core.from_rir import build_main_func
from core.sequantialize import sequentialize_main_func
from core.to_ir import calls_to_ir, filter_identity_and_apply_alias
from utils.io_utils import format_primfunc_nodes, format_main_func, export_main_func
from utils.ir_utils import inline_elementwise_op_calls, inline_shape_op_calls, bind_main_func_calls, normalize_main_func_axes
from utils.tir_utils import to_tir, to_relax
from utils.test_utils import validate_main_func_errors
from utils.pipeline import export_model_ir

class SimpleAttention(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        batch, seq, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch, seq, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq, self.heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1))
        attn = torch.softmax(scores, dim=-1)

        output = torch.matmul(attn, v)
        return output

def build_model_and_inputs():
    import os

    analyzer = tvm.arith.Analyzer()

    # 모델 설정
    d_model = 64
    heads = 4
    batch = 1
    seq = 16

    # 모델 생성
    model = SimpleAttention(d_model, heads)
    example_input = torch.randn(batch, seq, d_model)

    # Relax IR로 변환
    example_inputs = example_input
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
    # print_tir_ir(parsed_funcs)
    # parsed_output_path = f"{output_dir}/enc_attn_tir_parsed.txt"
    # save_pretty_tir(parsed_funcs, parsed_output_path)
    
    # ------------- For DEBUG ------------- #
    # x = func.body.block.body.body.body.body.block
    # simplified = analyzer.canonical_simplify(x)
    # y = simplified
    # print("\nvalue: ", y, "\ntype: ", y.__tvm_ffi_type_info__.type_cls if x is not isinstance(x, list) else None, "\n\n")
    # print(dir(y), "\n\n")
    # print("value: ", x, "\ntype: ", x.__tvm_ffi_type_info__.type_cls if x is not isinstance(x, list) else None, "\n\n")
    # print(dir(x))
