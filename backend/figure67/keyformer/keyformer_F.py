import triton
import triton.language as tl
import torch

@triton.autotune(
    configs = [
        triton.Config({'BLOCK_K': 32, 'BLOCK_P': 32}),
        triton.Config({'BLOCK_K': 32, 'BLOCK_P': 64}),
        triton.Config({'BLOCK_K': 32, 'BLOCK_P': 128}),
        triton.Config({'BLOCK_K': 64, 'BLOCK_P': 32}),
        triton.Config({'BLOCK_K': 64, 'BLOCK_P': 64}),
        triton.Config({'BLOCK_K': 64, 'BLOCK_P': 128}),
        triton.Config({'BLOCK_K': 128, 'BLOCK_P': 32}),
        triton.Config({'BLOCK_K': 128, 'BLOCK_P': 64}),
        triton.Config({'BLOCK_K': 128, 'BLOCK_P': 128})
    ], key=[]
)
@triton.jit
def kernel_0(
    C_ptr,
    C_stride0: tl.constexpr,
    C_stride1: tl.constexpr,
    C_stride2: tl.constexpr,
    C_exp_ptr,
    C_exp_stride0: tl.constexpr,
    C_exp_stride1: tl.constexpr,
    C_exp_stride2: tl.constexpr,
    C_exp_perturb_ptr,
    C_exp_perturb_stride0: tl.constexpr,
    C_exp_perturb_stride1: tl.constexpr,
    C_exp_perturb_stride2: tl.constexpr,
    C_out_ptr,
    C_out_stride0: tl.constexpr,
    C_out_stride1: tl.constexpr,
    K_cache_ptr,
    K_cache_stride0: tl.constexpr,
    K_cache_stride1: tl.constexpr,
    K_cache_stride2: tl.constexpr,
    O2_ptr,
    O2_stride0: tl.constexpr,
    O2_stride1: tl.constexpr,
    V_cache_ptr,
    V_cache_stride0: tl.constexpr,
    V_cache_stride1: tl.constexpr,
    V_cache_stride2: tl.constexpr,
    WK_ptr,
    WK_stride0: tl.constexpr,
    WK_stride1: tl.constexpr,
    WQ_ptr,
    WQ_stride0: tl.constexpr,
    WQ_stride1: tl.constexpr,
    WV_ptr,
    WV_stride0: tl.constexpr,
    WV_stride1: tl.constexpr,
    X_ptr,
    X_stride0: tl.constexpr,
    X_stride1: tl.constexpr,
    noise_ptr,
    noise_stride0: tl.constexpr,
    noise_stride1: tl.constexpr,
    noise_stride2: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_P: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    N4: tl.constexpr,
    P: tl.constexpr
):
    # Allocate intermediate tensors
    C_perturb = tl.zeros((1, M, BLOCK_P), dtype=tl.float16)
    C_sum = tl.zeros((1, M), dtype=tl.float16)
    C_sum_perturb = tl.zeros((1, M), dtype=tl.float16)
    K1 = tl.zeros((M, BLOCK_N), dtype=tl.float16)
    O = tl.zeros((1, M, D), dtype=tl.float16)
    Q1 = tl.zeros((M, BLOCK_N), dtype=tl.float16)
    V1 = tl.zeros((M, BLOCK_N), dtype=tl.float16)

    # Parallel loop n from 0 to Q1_dim1 with tile size BLOCK_N
    # Executed across grid dimension 0
    n = 0 + tl.program_id(0) * BLOCK_N
    
    # Sequential loop k from 0 to 4096 with tile size BLOCK_K
    for k in range(0, 4096, BLOCK_K):
        offset_0 = (tl.arange(0, 16))[:, None] * X_stride0 + (k + tl.arange(0, BLOCK_K))[None, :] * X_stride1
        k_indices = k + tl.arange(0, BLOCK_K)
        mask_0 = (k_indices < N)[None, :]
        temp_0 = tl.load(X_ptr + offset_0, mask=mask_0, other=0.0)
        offset_1 = (k + tl.arange(0, BLOCK_K))[:, None] * WQ_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * WQ_stride1
        n_indices = n + tl.arange(0, BLOCK_N)
        mask_1 = (k_indices < N)[:, None] & (n_indices < N)[None, :]
        temp_1 = tl.load(WQ_ptr + offset_1, mask=mask_1, other=0.0)
        Q1 = ((1 * Q1).to(tl.float16) + tl.dot(temp_0, temp_1).to(tl.float16)).to(tl.float16)
        offset_2 = (k + tl.arange(0, BLOCK_K))[:, None] * WK_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * WK_stride1
        mask_2 = (k_indices < N)[:, None] & (n_indices < N)[None, :]
        temp_2 = tl.load(WK_ptr + offset_2, mask=mask_2, other=0.0)
        K1 = ((1 * K1).to(tl.float16) + tl.dot(temp_0, temp_2).to(tl.float16)).to(tl.float16)
        offset_3 = (k + tl.arange(0, BLOCK_K))[:, None] * WV_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * WV_stride1
        mask_3 = (k_indices < N)[:, None] & (n_indices < N)[None, :]
        temp_3 = tl.load(WV_ptr + offset_3, mask=mask_3, other=0.0)
        V1 = ((1 * V1).to(tl.float16) + tl.dot(temp_0, temp_3).to(tl.float16)).to(tl.float16)
    temp_4 = tl.expand_dims(Q1, 1)
    Q = tl.permute(temp_4, (1, 0, 2))
    temp_5 = tl.expand_dims(K1, 1)
    K = tl.permute(temp_5, (1, 0, 2))
    temp_6 = tl.expand_dims(V1, 1)
    V = tl.permute(temp_6, (1, 0, 2))
    offset_4 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * K_cache_stride0 + (1008 + tl.arange(0, 16))[None, :, None] * K_cache_stride1 + (tl.arange(0, 128))[None, None, :] * K_cache_stride2
    tl.store(K_cache_ptr + offset_4, K)
    offset_5 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * V_cache_stride0 + (1008 + tl.arange(0, 16))[None, :, None] * V_cache_stride1 + (tl.arange(0, 128))[None, None, :] * V_cache_stride2
    tl.store(V_cache_ptr + offset_5, V)
    # Sequential loop p from 0 to 1024 with tile size BLOCK_P
    for p in range(0, 1024, BLOCK_P):
        offset_6 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * K_cache_stride0 + (p + tl.arange(0, BLOCK_P))[None, :, None] * K_cache_stride1 + (tl.arange(0, 128))[None, None, :] * K_cache_stride2
        p_indices = p + tl.arange(0, BLOCK_P)
        mask_4 = (p_indices < P+M)[None, :, None]
        temp_7 = tl.load(K_cache_ptr + offset_6, mask=mask_4, other=0.0)
        temp_8 = tl.permute(temp_7, (0, 2, 1))
        offset_7 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * C_stride0 + (tl.arange(0, 16))[None, :, None] * C_stride1 + (p + tl.arange(0, BLOCK_P))[None, None, :] * C_stride2
        mask_5 = (p_indices < P+M)[None, None, :]
        tl.store(C_ptr + offset_7, tl.dot(Q, temp_8).to(tl.float16), mask=mask_5)
    # Sequential loop p from 0 to 1024 with tile size BLOCK_P
    for p in range(0, 1024, BLOCK_P):
        offset_8 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * C_stride0 + (tl.arange(0, 16))[None, :, None] * C_stride1 + (p + tl.arange(0, BLOCK_P))[None, None, :] * C_stride2
        p_indices = p + tl.arange(0, BLOCK_P)
        mask_6 = (p_indices < P+M)[None, None, :]
        temp_9 = tl.load(C_ptr + offset_8, mask=mask_6, other=0.0)
        offset_9 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * noise_stride0 + (tl.arange(0, 16))[None, :, None] * noise_stride1 + (p + tl.arange(0, BLOCK_P))[None, None, :] * noise_stride2
        mask_7 = (p_indices < P+M)[None, None, :]
        temp_10 = tl.load(noise_ptr + offset_9, mask=mask_7, other=0.0)
        C_perturb = ((temp_9 + temp_10).to(tl.float16) / 1.5).to(tl.float16)
        offset_10 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * C_exp_stride0 + (tl.arange(0, 16))[None, :, None] * C_exp_stride1 + (p + tl.arange(0, BLOCK_P))[None, None, :] * C_exp_stride2
        mask_8 = (p_indices < P+M)[None, None, :]
        tl.store(C_exp_ptr + offset_10, tl.exp(temp_9.to(tl.float32)).to(tl.float16), mask=mask_8)
        offset_11 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * C_exp_perturb_stride0 + (tl.arange(0, 16))[None, :, None] * C_exp_perturb_stride1 + (p + tl.arange(0, BLOCK_P))[None, None, :] * C_exp_perturb_stride2
        mask_9 = (p_indices < P+M)[None, None, :]
        tl.store(C_exp_perturb_ptr + offset_11, tl.exp(C_perturb.to(tl.float32)).to(tl.float16), mask=mask_9)
        C_sum = ((1 * C_sum).to(tl.float16) + tl.sum(tl.exp(temp_9.to(tl.float32)).to(tl.float16), axis=2, dtype=tl.float16)).to(tl.float16)
        C_sum_perturb = ((1 * C_sum_perturb).to(tl.float16) + tl.sum(tl.exp(C_perturb.to(tl.float32)).to(tl.float16), axis=2, dtype=tl.float16)).to(tl.float16)
    # Skipped empty sloop with dummy body
    # Sequential loop p from 0 to 1024 with tile size BLOCK_P
    for p in range(0, 1024, BLOCK_P):
        offset_12 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * C_exp_stride0 + (tl.arange(0, 16))[None, :, None] * C_exp_stride1 + (p + tl.arange(0, BLOCK_P))[None, None, :] * C_exp_stride2
        p_indices = p + tl.arange(0, BLOCK_P)
        mask_10 = (p_indices < P+M)[None, None, :]
        temp_11 = tl.load(C_exp_ptr + offset_12, mask=mask_10, other=0.0)
        offset_13 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * V_cache_stride0 + (p + tl.arange(0, BLOCK_P))[None, :, None] * V_cache_stride1 + (tl.arange(0, 128))[None, None, :] * V_cache_stride2
        mask_11 = (p_indices < P+M)[None, :, None]
        temp_12 = tl.load(V_cache_ptr + offset_13, mask=mask_11, other=0.0)
        O = (tl.dot(temp_11, temp_12).to(tl.float16) + (1 * O).to(tl.float16)).to(tl.float16)
        offset_14 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * C_exp_perturb_stride0 + (tl.arange(0, 16))[None, :, None] * C_exp_perturb_stride1 + (p + tl.arange(0, BLOCK_P))[None, None, :] * C_exp_perturb_stride2
        mask_12 = (p_indices < P+M)[None, None, :]
        temp_13 = tl.load(C_exp_perturb_ptr + offset_14, mask=mask_12, other=0.0)
        offset_15 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None] * C_out_stride0 + (p + tl.arange(0, BLOCK_P))[None, :] * C_out_stride1
        mask_13 = (p_indices < P+M)[None, :]
        tl.store(C_out_ptr + offset_15, tl.sum((temp_13 / C_sum_perturb[:, :, None]).to(tl.float16), axis=1, dtype=tl.float16), mask=mask_13)
    O = (O / C_sum[:, :, None]).to(tl.float16)
    temp_14 = tl.permute(O, (1, 0, 2))
    offset_16 = (tl.arange(0, 16))[:, None] * O2_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * O2_stride1
    n_indices = n + tl.arange(0, BLOCK_N)
    mask_14 = (n_indices < N)[None, :]
    tl.store(O2_ptr + offset_16, tl.reshape(temp_14, (M, D)), mask=mask_14)


# Metadata for benchmark.py
TENSOR_PARAMS = ['C', 'C_exp', 'C_exp_perturb', 'C_out', 'K_cache', 'O2', 'V_cache', 'WK', 'WQ', 'WV', 'X', 'noise']
BLOCK_PARAMS = ['block_k', 'block_p']

def forward(C, C_exp, C_exp_perturb, C_out, K_cache, O2, V_cache, WK, WQ, WV, X, noise, block_k=16, block_p=16):
    """
    Wrapper function that executes all kernels sequentially.
    """
    kernel_0[((4096 - 0 + 128 - 1) // 128,)](
        C,
        C.stride(0),
        C.stride(1),
        C.stride(2),
        C_exp,
        C_exp.stride(0),
        C_exp.stride(1),
        C_exp.stride(2),
        C_exp_perturb,
        C_exp_perturb.stride(0),
        C_exp_perturb.stride(1),
        C_exp_perturb.stride(2),
        C_out,
        C_out.stride(0),
        C_out.stride(1),
        K_cache,
        K_cache.stride(0),
        K_cache.stride(1),
        K_cache.stride(2),
        O2,
        O2.stride(0),
        O2.stride(1),
        V_cache,
        V_cache.stride(0),
        V_cache.stride(1),
        V_cache.stride(2),
        WK,
        WK.stride(0),
        WK.stride(1),
        WQ,
        WQ.stride(0),
        WQ.stride(1),
        WV,
        WV.stride(0),
        WV.stride(1),
        X,
        X.stride(0),
        X.stride(1),
        noise,
        noise.stride(0),
        noise.stride(1),
        noise.stride(2),
        # BLOCK_K, BLOCK_P are provided by autotune,
        BLOCK_N=128,
        # BLOCK_K is automatically set by autotune,
        # BLOCK_P is automatically set by autotune,
        D=128,
        H=32,
        M=16,
        N=4096,
        N4=16384,
        P=1008
    )

    # Return output tensors if needed
    # This depends on your specific use case
    pass
