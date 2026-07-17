import torch
import torch.nn as nn

from utils.pipeline import export_model_ir


class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, r: int = 4, alpha: float = 8.0):
        super().__init__()
        self.base = nn.Linear(in_features, out_features, bias=False)
        self.lora_a = nn.Linear(in_features, r, bias=False)
        self.lora_b = nn.Linear(r, out_features, bias=False)
        self.scale = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(x)) * self.scale


class MoE(nn.Module):
    def __init__(self, hidden: int, experts: int = 4, top_k: int = 2):
        super().__init__()
        self.experts = experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden, experts, bias=True)
        self.expert_fc1 = nn.ModuleList([nn.Linear(hidden, hidden * 2, bias=True) for _ in range(experts)])
        self.expert_fc2 = nn.ModuleList([nn.Linear(hidden * 2, hidden, bias=True) for _ in range(experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.gate(x)
        topk_vals, topk_idx = torch.topk(logits, self.top_k, dim=-1)
        topk_weights = torch.softmax(topk_vals, dim=-1)
        mask = torch.nn.functional.one_hot(topk_idx, num_classes=self.experts)
        mask = mask.to(dtype=topk_weights.dtype)
        weights = (mask * topk_weights.unsqueeze(-1)).sum(dim=-2)

        outputs = []
        for i in range(self.experts):
            h = torch.relu(self.expert_fc1[i](x))
            outputs.append(self.expert_fc2[i](h))
        stacked = torch.stack(outputs, dim=-2)
        weights = weights.unsqueeze(-1)
        return (stacked * weights).sum(dim=-2)


class ParallelBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, head_dim: int):
        super().__init__()
        self.hidden = hidden
        self.heads = heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        self.norm = nn.LayerNorm(hidden)
        self.q_proj = nn.Linear(hidden, heads * head_dim, bias=True)
        self.k_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.out_proj = LoRALinear(heads * head_dim, hidden, r=4, alpha=8.0)

        self.moe = MoE(hidden, experts=4, top_k=2)
        self.res_scale = 0.7

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        bsz, seq_len, _ = x_norm.shape
        q = self.q_proj(x_norm).view(bsz, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(bsz, seq_len, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(bsz, seq_len, self.heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v)
        attn = attn.transpose(1, 2).contiguous().view(bsz, seq_len, self.heads * self.head_dim)
        attn_out = self.out_proj(attn)

        moe_out = self.moe(x_norm)
        return x + self.res_scale * attn_out + (1.0 - self.res_scale) * moe_out


def build_model_and_inputs():
    seq_len = 8
    hidden = 128
    heads = 4
    head_dim = 32

    device = torch.device("cpu")
    dtype = torch.float32

    model = ParallelBlock(hidden=hidden, heads=heads, head_dim=head_dim).to(
        device=device, dtype=dtype
    )
    x = torch.randn((1, seq_len, hidden), device=device, dtype=dtype)
    return {
        "model": model,
        "example_inputs": x,
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
