import torch
import torch.nn as nn

from utils.pipeline import export_model_ir


class SwiGLU(nn.Module):
    def __init__(self, hidden: int, ff_mult: int = 4):
        super().__init__()
        inner = hidden * ff_mult
        self.w1 = nn.Linear(hidden, inner, bias=True)
        self.w2 = nn.Linear(hidden, inner, bias=True)
        self.w3 = nn.Linear(inner, hidden, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(self.w1(x) * torch.sigmoid(self.w2(x)))


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden: int, heads: int, head_dim: int, causal: bool):
        super().__init__()
        self.hidden = hidden
        self.heads = heads
        self.head_dim = head_dim
        self.causal = causal
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.out_proj = nn.Linear(heads * head_dim, hidden, bias=False)

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        bsz, q_len, _ = query.shape
        k_len = key.shape[1]

        q = self._reshape(self.q_proj(query))
        k = self._reshape(self.k_proj(key))
        v = self._reshape(self.v_proj(value))

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if self.causal:
            q_pos = torch.arange(q_len, device=query.device)
            k_pos = torch.arange(k_len, device=query.device)
            mask = k_pos[None, :] <= q_pos[:, None]
            mask = mask.to(dtype=scores.dtype)
            scores = scores + (1.0 - mask)[None, None, :, :] * (-1e4)

        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(bsz, q_len, self.heads * self.head_dim)
        return self.out_proj(out)


class Encoder(nn.Module):
    def __init__(self, hidden: int, heads: int, head_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.attn = MultiHeadAttention(hidden, heads, head_dim, causal=False)
        self.ffn = SwiGLU(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm(x), self.norm(x), self.norm(x))
        x = x + self.ffn(self.norm(x))
        return x


class Decoder(nn.Module):
    def __init__(self, hidden: int, heads: int, head_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.self_attn = MultiHeadAttention(hidden, heads, head_dim, causal=True)
        self.norm2 = nn.LayerNorm(hidden)
        self.cross_attn = MultiHeadAttention(hidden, heads, head_dim, causal=False)
        self.norm3 = nn.LayerNorm(hidden)
        self.ffn = SwiGLU(hidden)

    def forward(self, x: torch.Tensor, enc: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), self.norm2(enc), self.norm2(enc))
        x = x + self.ffn(self.norm3(x))
        return x


class EncoderDecoder(nn.Module):
    def __init__(self, hidden: int, heads: int, head_dim: int):
        super().__init__()
        self.encoder = Encoder(hidden, heads, head_dim)
        self.decoder = Decoder(hidden, heads, head_dim)

    def forward(self, enc_in: torch.Tensor, dec_in: torch.Tensor) -> torch.Tensor:
        enc = self.encoder(enc_in)
        return self.decoder(dec_in, enc)


def build_model_and_inputs():
    seq_len = 16
    hidden = 256
    heads = 4
    head_dim = 64

    device = torch.device("cpu")
    dtype = torch.float32

    model = EncoderDecoder(hidden=hidden, heads=heads, head_dim=head_dim).to(
        device=device, dtype=dtype
    )
    enc_in = torch.randn((1, seq_len, hidden), device=device, dtype=dtype)
    dec_in = torch.randn((1, seq_len, hidden), device=device, dtype=dtype)
    return {
        "model": model,
        "example_inputs": (enc_in, dec_in),
        "inline_shape_op": True,
        "inline_elementwise_op": True,
        "remove_short_loop_threshold": 24,
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
