import torch
import torch.nn as nn
from torch.export import export
import tvm
from tvm import tir
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program
from utils.pipeline import export_model_ir

import ir.AST as T
from core.from_tir import build_primfunc_nodes
from core.from_rir import build_main_func
from core.sequantialize import sequentialize_main_func
from core.to_ir import calls_to_ir, filter_identity_and_apply_alias
from utils.io_utils import format_primfunc_nodes, format_main_func, export_main_func
from utils.ir_utils import inline_elementwise_op_calls, inline_shape_op_calls, bind_main_func_calls, normalize_main_func_axes
from utils.tir_utils import to_tir, to_relax
from utils.test_utils import validate_main_func_errors


class TransformerBlock(nn.Module):
    def __init__(self, M, N, D, P, cache_K, cache_V, device=None, dtype=None):
        super().__init__()
        self.M = M  # Current Sequence Length (16)
        self.N = N  # Hidden Size (4096)
        self.D = D  # Head Dimension (128)
        self.P = P  # Past Context Length
        self.H = N // D # Number of Heads (32)
        self.device = device
        self.dtype = dtype

        # --- 1. Attention Part ---
        self.ln1 = nn.LayerNorm(N, elementwise_affine=True) # Pre-Norm for Attention
        self.q_proj = nn.Linear(N, N, bias=False)
        self.k_proj = nn.Linear(N, N, bias=False)
        self.v_proj = nn.Linear(N, N, bias=False)
        self.o_proj = nn.Linear(N, N, bias=False) # Output Projection

        # --- 2. FFN Part ---
        # 보통 Hidden Size의 4배를 사용합니다.
        self.ffn_hidden_dim = 4 * N 
        self.ln2 = nn.LayerNorm(N, elementwise_affine=True) # Pre-Norm for FFN
        
        # 일반적인 MLP 구조: Up -> Activation -> Down
        self.w_up = nn.Linear(N, self.ffn_hidden_dim, bias=False)
        self.act = nn.GELU() # Activation Function
        self.w_down = nn.Linear(self.ffn_hidden_dim, N, bias=False)

        # Cache Buffer 등록
        self.register_buffer('cache_K', cache_K.to(device))
        self.register_buffer('cache_V', cache_V.to(device))

    def forward(self, X):
        """
        X shape: (M, N)
        """
        # === 1. Multi-Head Attention Block ===
        residual = X
        
        # 1-1. Pre-LayerNorm
        X_norm = self.ln1(X) 

        # 1-2. Q, K, V Projection
        q = self.q_proj(X_norm)
        k = self.k_proj(X_norm)
        v = self.v_proj(X_norm)

        # 1-3. Reshape for Multi-Head
        q = q.view(self.M, self.H, self.D)  # (M, H, D)
        k = k.view(self.M, self.H, self.D)  # (M, H, D)
        v = v.view(self.M, self.H, self.D)  # (M, H, D)

        # 1-4. KV Cache Update
        # Transpose to (H, M, D) for cache storage
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        # Update cache (In-place update logic)
        # Slicing을 사용하여 in-place 업데이트를 수행합니다.
        self.cache_K[:, self.P : self.P + self.M, :] = k
        self.cache_V[:, self.P : self.P + self.M, :] = v
        
        # 어텐션 연산에는 업데이트된 전체 캐시를 사용
        cache_K_curr = self.cache_K[:, : self.P + self.M, :] 
        cache_V_curr = self.cache_V[:, : self.P + self.M, :]

        # 1-5. Attention Calculation
        q = q.transpose(0, 1)  # (H, M, D)
        
        # Score: (H, M, D) @ (H, D, P+M) -> (H, M, P+M)
        scores = torch.matmul(q, cache_K_curr.transpose(1, 2))
        scores = scores / (self.D ** 0.5) # Scale
        
        weights = torch.softmax(scores, dim=-1)
        
        # Context: (H, M, P+M) @ (H, P+M, D) -> (H, M, D)
        attn_out = torch.matmul(weights, cache_V_curr)
        
        # Restore shape: (H, M, D) -> (M, H, D) -> (M, N)
        attn_out = attn_out.transpose(0, 1).contiguous().view(self.M, self.N)
        
        # 1-6. Output Projection
        attn_out = self.o_proj(attn_out)

        # 1-7. Residual Connection
        X = residual + attn_out


        # === 2. Feed-Forward Network (FFN) Block ===
        residual = X
        
        # 2-1. Pre-LayerNorm
        X_norm = self.ln2(X)
        
        # 2-2. Up Projection
        ffn_out = self.w_up(X_norm)
        
        # 2-3. Activation
        ffn_out = self.act(ffn_out)
        
        # 2-4. Down Projection
        ffn_out = self.w_down(ffn_out)
        
        # 2-5. Residual Connection
        X = residual + ffn_out

        return X


def build_model_and_inputs():
    import os

    # 파라미터 설정
    M = 16      # Sequence Length (Chunk)
    N = 4096    # Hidden Dimension
    D = 128     # Head Dimension
    H = 32      # Number of Heads (4096 / 128 = 32)
    P = 1008    # Past Context Length (KV Cache에 이미 있는 길이)

    device = torch.device('cpu')  # TVM export는 CPU에서 안정적
    dtype = torch.float32         # Export 호환성을 위해 float32 권장

    # 더미 데이터 생성
    X = torch.randn((M, N), device=device, dtype=dtype)

    # KV Cache 초기화 (전체 크기: P + M + 여유분)
    # 여기서는 정확히 P+M 만큼의 공간을 가정하거나, 
    # forward에서 슬라이싱을 하므로 충분히 큰 버퍼를 줍니다.
    # 코드상 self.cache_K[:, self.P:self.P+self.M, :]에 쓰므로 최소 P+M 크기 필요
    K_cache = torch.randn((H, P + M, D), device=device, dtype=dtype)
    V_cache = torch.randn((H, P + M, D), device=device, dtype=dtype)

    # 모델 인스턴스 생성 (LayerNorm, FFN 포함)
    model = TransformerBlock(M, N, D, P, K_cache, V_cache, device=device, dtype=dtype)

    print(f"Model created. Converting to Relax IR...")
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
