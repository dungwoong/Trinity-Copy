import torch
import torch.nn as nn

from utils.pipeline import export_model_ir


class GQARotaryDecoder(nn.Module):
    def __init__(
        self,
        hidden: int,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        max_len: int,
        past_len: int,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.hidden = hidden
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.max_len = max_len
        self.past_len = past_len
        self.scale = head_dim**-0.5
        self.repeat_kv = q_heads // kv_heads

        qkv_dim = (q_heads + 2 * kv_heads) * head_dim
        self.qkv_proj = nn.Linear(hidden, qkv_dim, bias=False)
        self.out_proj = nn.Linear(q_heads * head_dim, hidden, bias=False)

        cache_shape = (1, kv_heads, max_len, head_dim)
        self.register_buffer("cache_k", torch.zeros(cache_shape, device=device, dtype=dtype))
        self.register_buffer("cache_v", torch.zeros(cache_shape, device=device, dtype=dtype))

    def _apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, T, D)
        bsz, heads, seq_len, dim = x.shape
        half = dim // 2
        positions = torch.arange(seq_len, device=x.device, dtype=x.dtype)
        inv_freq = torch.arange(0, half, device=x.device, dtype=x.dtype) / half
        angles = positions[:, None] * inv_freq[None, :]
        sin = torch.sin(angles)[None, None, :, :]
        cos = torch.cos(angles)[None, None, :, :]

        x_even = x[..., :half]
        x_odd = x[..., half:]
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        return torch.cat([out_even, out_odd], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)
        q_size = self.q_heads * self.head_dim
        kv_size = self.kv_heads * self.head_dim
        q, k, v = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seq_len, self.q_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.kv_heads, self.head_dim).transpose(1, 2)

        q = self._apply_rope(q)
        k = self._apply_rope(k)

        self.cache_k[:, :, self.past_len : self.past_len + seq_len, :] = k
        self.cache_v[:, :, self.past_len : self.past_len + seq_len, :] = v

        key = self.cache_k[:, :, : self.past_len + seq_len, :]
        value = self.cache_v[:, :, : self.past_len + seq_len, :]

        if self.repeat_kv > 1:
            key = key.repeat_interleave(self.repeat_kv, dim=1)
            value = value.repeat_interleave(self.repeat_kv, dim=1)

        scores = torch.matmul(q, key.transpose(-1, -2)) * self.scale
        key_len = key.shape[-2]
        q_pos = torch.arange(seq_len, device=x.device)
        k_pos = torch.arange(key_len, device=x.device)
        mask = k_pos[None, :] <= (q_pos[:, None] + self.past_len)
        mask = mask.to(dtype=scores.dtype)
        scores = scores + (1.0 - mask)[None, None, :, :] * (-1e4)

        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, value)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.q_heads * self.head_dim)
        return self.out_proj(out)


def build_model_and_inputs():
    seq_len = 16
    hidden = 256
    q_heads = 4
    kv_heads = 1
    head_dim = 64
    past_len = 8
    max_len = past_len + seq_len

    device = torch.device("cpu")
    dtype = torch.float32

    model = GQARotaryDecoder(
        hidden=hidden,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        max_len=max_len,
        past_len=past_len,
        device=device,
        dtype=dtype,
    ).to(device=device, dtype=dtype)

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
