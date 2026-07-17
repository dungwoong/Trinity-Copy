import math
import torch
import torch.nn as nn

from utils.pipeline import export_model_ir


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight


def _build_alibi_slopes(n_heads: int) -> torch.Tensor:
    def get_slopes(power: int) -> list[float]:
        start = 2 ** (-2 ** (-(math.log2(power) - 3)))
        ratio = start
        return [start * ratio**i for i in range(power)]

    if math.log2(n_heads).is_integer():
        slopes = get_slopes(n_heads)
    else:
        closest = 2 ** math.floor(math.log2(n_heads))
        slopes = get_slopes(closest)
        extra = get_slopes(2 * closest)
        slopes.extend(extra[0::2][: n_heads - closest])
    return torch.tensor(slopes, dtype=torch.float32)


class AlibiEncoderBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, head_dim: int):
        super().__init__()
        self.hidden = hidden
        self.heads = heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        self.norm1 = RMSNorm(hidden)
        self.q_proj = nn.Linear(hidden, heads * head_dim, bias=True)
        self.k_proj = nn.Linear(hidden, heads * head_dim, bias=True)
        self.v_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.out_proj = nn.Linear(heads * head_dim, hidden, bias=False)

        self.norm2 = RMSNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 4, bias=True),
            nn.GELU(),
            nn.Linear(hidden * 4, hidden, bias=True),
        )
        self.res_scale = 0.5

        slopes = _build_alibi_slopes(heads)
        self.register_buffer("alibi_slopes", slopes)

    def _alibi_bias(self, seq_len: int, device, dtype) -> torch.Tensor:
        pos = torch.arange(seq_len, device=device)
        bias = (pos[None, :] - pos[:, None]).abs().to(dtype)
        slopes = self.alibi_slopes.to(device=device, dtype=dtype)[:, None, None]
        return -slopes * bias[None, :, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        x_norm = self.norm1(x)
        q = self.q_proj(x_norm).view(bsz, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(bsz, seq_len, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(bsz, seq_len, self.heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        scores = scores + self._alibi_bias(seq_len, x.device, scores.dtype)[None, :, :, :]
        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v)
        attn = attn.transpose(1, 2).contiguous().view(bsz, seq_len, self.heads * self.head_dim)
        x = x + self.out_proj(attn)

        x_norm = self.norm2(x)
        x = x + self.res_scale * self.ffn(x_norm)
        return x


def build_model_and_inputs():
    seq_len = 16
    hidden = 256
    heads = 4
    head_dim = 64

    device = torch.device("cpu")
    dtype = torch.float32

    model = AlibiEncoderBlock(hidden=hidden, heads=heads, head_dim=head_dim).to(
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
