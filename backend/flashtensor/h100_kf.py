import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Callable, Any, Optional, Tuple

def bench_kf(model, M, N, P, D, H, device, dtype):
    x = torch.randn(1, M, N, dtype=dtype, device=device)
    wq = torch.randn(1, N, N, dtype=dtype, device=device)
    wk = torch.randn(1, N, N, dtype=dtype, device=device)
    wv = torch.randn(1, N, N, dtype=dtype, device=device)
    wqkv = torch.cat([wq, wk, wv], dim=2).to(device=device, dtype=dtype)
    k_cache = torch.randn((1, H, P+M, D), dtype=dtype, device=device)
    v_cache = torch.randn((1, H, P+M, D), dtype=dtype, device=device)

    out_p1 = torch.empty((1, H, M, D), dtype=dtype, device=device)
    out_p9 = torch.empty((1, H, M, 1), dtype=torch.float32, device=device)
    out_p7 = torch.empty((1, H, M), dtype=torch.float32, device=device)

    if model == "llama":
      attn = KeyFormer_llama
    else:
      attn = KeyFormer_falcon

    for _ in range(10):
        kf_kernel(x, wqkv, k_cache, v_cache, M, N, P, D, H, attn, out_p1, out_p9, out_p7)
    torch.cuda.synchronize()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)

    s.record()
    for _ in range(1000):
      kf_kernel(x, wqkv, k_cache, v_cache, M, N, P, D, H, attn, out_p1, out_p9, out_p7)
    e.record()
    torch.cuda.synchronize()

    avg = s.elapsed_time(e)/1000
    return avg

def kf_kernel(x, wqkv, k_cache, v_cache, M, N, P, D, H, attn, out_p1, out_p9, out_p7):
    qkv = torch.matmul(x, wqkv)
    q, k, v = torch.chunk(qkv, 3, dim=-1)

    q = q.view(1, M, H, D).transpose(1,2)
    k = k.view(1, M, H, D).transpose(1,2)
    v = v.view(1, M, H, D).transpose(1,2)

    k_cache[:, :, P:P+M, :] = k
    v_cache[:, :, P:P+M, :] = v

    # KeyFormer_llama/KeyFormer_falcon: (q, k, v) 순서로 호출
    attn(q, k_cache, v_cache, out_p1, out_p9, out_p7)

def KeyFormer_llama(q, k, v, out_p1, out_p9, out_p7):
    """KeyFormer llama: p1 + p9 + p7 순차 호출"""
    # g = torch.empty(1, 32, 1024, 1024, dtype=torch.float32, device=q.device)  # placeholder
    g = None

    out_p1 = KeyFormer_p1_llama(q, v, k, out_p1)
    denom_m = KeyFormer_p9_llama(q, k, g, out_p9)
    out_p7 = KeyFormer_p7_llama(q, k, g, denom_m, out_p7)
    return out_p1, denom_m, out_p7

def KeyFormer_falcon(q, k, v, out_p1, out_p9, out_p7):
    """KeyFormer falcon: p1 + p9 + p7 순차 호출"""
    # g = torch.empty(1, 71, 1024, 1024, dtype=torch.float32, device=q.device)  # placeholder
    g= None
    out_p1 = KeyFormer_p1_falcon(q, v, k, out_p1)
    denom_m = KeyFormer_p9_falcon(q, k, g, out_p9)
    out_p7 = KeyFormer_p7_falcon(q, k, g, denom_m, out_p7)
    return out_p1, denom_m, out_p7

def KeyFormer_p1_llama(q: torch.Tensor, v: torch.Tensor, k: torch.Tensor, out) -> torch.Tensor:
  # out: (1,16,32,128)
  # out = torch.empty_like(q)
  grid = (1, 1, 32)  # (B tiles, M tiles, H heads)
  KeyFormer_p1_kernel_llama[grid](
      q, v, k, out,
      CONST_HD=32*128,  # 4096
      BLOCK_D=64       # tl.constexpr
  )
  return out

def KeyFormer_p9_llama(q: torch.Tensor, k: torch.Tensor, g: torch.Tensor, out) -> torch.Tensor:
  # dev = q.device
  # out = torch.empty(1, 32, 16, 1, dtype=torch.float32, device=dev)  # (B,H,M,1)
  grid = (1, 1, 32)  # (B tiles, M tile(1), H)
  KeyFormer_p9_kernel_llama[grid](q, k, g, out)
  return out

def KeyFormer_p7_llama(q: torch.Tensor, k: torch.Tensor, g: torch.Tensor, denom_m: torch.Tensor, out) -> torch.Tensor:
  # dev = q.device
  # out = torch.empty(1, 32, 16, dtype=torch.float32, device=dev)  # (B,H,M) reduce over N
  grid = (1, 1, 32)
  KeyFormer_p7_kernel_llama[grid](q, k, g, denom_m, out)
  return out

@triton.jit
def KeyFormer_p1_kernel_llama(
  arg_q,     # (1,16,H,D)
  arg_v,     # (1,1024,H,D)
  arg_k,     # (1,1024,H,D)
  arg_out,   # (1,16,H,D)
  CONST_HD: tl.constexpr,   # H*D (4096 or 4544)
  BLOCK_D: tl.constexpr      # 128 or 64
):
  # fixed problem sizes
  CONST_M : tl.constexpr = 16
  CONST_N : tl.constexpr = 1024
  BLOCK_M : tl.constexpr = 16
  BLOCK_N : tl.constexpr = 128

  pid_b = tl.program_id(0)      # unused (B=1)
  pid_m = tl.program_id(1)      # unused (one M tile since M=16)
  pid_h = tl.program_id(2)      # 0..H-1

  # scale ~ 1/sqrt(D): 원 코드 값 사용
  const_scale = 1.275311e-01 if BLOCK_D == 128 else (1.0 / 8.0)  # 0.125

  # per-head offset (H axis contiguous by D)
  off_hD = pid_h * BLOCK_D

  # Q: (M,D) with strides (HD,1)
  block_ptr_q = tl.make_block_ptr(
    base=arg_q + off_hD,
    shape=(BLOCK_M, BLOCK_D),
    strides=(CONST_HD, 1),
    offsets=(0, 0),
    block_shape=(BLOCK_M, BLOCK_D),
    order=(1, 0),
  )
  q = tl.load(block_ptr_q).to(tl.float32) * const_scale
  q = q.to(tl.float16)  # (16,D)

  # K^T: (D,N) with strides (1,HD)
  block_ptr_kT = tl.make_block_ptr(
    base=arg_k + off_hD,
    shape=(BLOCK_D, CONST_N),
    strides=(1, CONST_HD),
    offsets=(0, 0),
    block_shape=(BLOCK_D, BLOCK_N),
    order=(0, 1),
  )

  # V: (N,D) with strides (HD,1)
  block_ptr_v = tl.make_block_ptr(
    base=arg_v + off_hD,
    shape=(CONST_N, BLOCK_D),
    strides=(CONST_HD, 1),
    offsets=(0, 0),
    block_shape=(BLOCK_N, BLOCK_D),
    order=(1, 0),
  )

  # OUT, DENOM
  block_ptr_out = tl.make_block_ptr(
    base=arg_out + off_hD,
    shape=(BLOCK_M, BLOCK_D),
    strides=(CONST_HD, 1),
    offsets=(0, 0),
    block_shape=(BLOCK_M, BLOCK_D),
    order=(1, 0),
  )

  acc_out   = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
  acc_denom = tl.zeros([BLOCK_M, 1],       dtype=tl.float32)

  # loop over N tiles
  for n0 in range(0, CONST_N, BLOCK_N):
    kT_tile = tl.load(block_ptr_kT)                 # (D,BN)
    scores  = tl.dot(q, kT_tile)                    # (M,BN)
    probs   = tl.math.exp2(scores)                  # (M,BN)
    acc_denom += tl.sum(probs, axis=1, keep_dims=True).to(tl.float32)
    v_tile  = tl.load(block_ptr_v)                  # (BN,D)
    acc_out += tl.dot(probs.to(tl.float16), v_tile) # (M,D)
    block_ptr_kT = tl.advance(block_ptr_kT, (0, BLOCK_N))
    block_ptr_v  = tl.advance(block_ptr_v,  (BLOCK_N, 0))

  out = (acc_out / acc_denom).to(tl.float16)
  tl.store(block_ptr_out, out)


@triton.autotune(configs=[triton.Config({}, num_warps=4), triton.Config({}, num_warps=8)], key=[])
@triton.jit
def KeyFormer_p9_kernel_llama(
  q,   # (1,16,32,128)
  k,   # (1,1024,32,128)
  g,   # (1,32,1024,1024)  # 안 써도 무방
  out  # (1,32,16,1)
):
  pid_h = tl.program_id(2)  # head 0..31
  const_scale = 1.0 / 11.3125  # 1/sqrt(128)

  # === base offset per head ===
  off_hD = pid_h * 128  # BLOCK_D = 128

  # Q tile (M,D) = (16,128), strides (HD=4096, 1)
  block_ptr_q = tl.make_block_ptr(
    base=q + off_hD,
    shape=(16, 128),
    strides=(4096, 1),   # H*D = 32*128
    offsets=(0, 0),
    block_shape=(16, 128),
    order=(1, 0),
  )
  q_tile = tl.load(block_ptr_q).to(tl.float32) * const_scale
  q_tile = q_tile.to(tl.float16)

  # out denom (M,1)
  block_ptr_out = tl.make_block_ptr(
    base=out + pid_h * 16,
    shape=(16, 1),
    strides=(1, 1),
    offsets=(0, 0),
    block_shape=(16, 1),
    order=(1, 0),
  )
  acc = tl.zeros([16, 1], dtype=tl.float32)

  # iterate N in chunks of 128
  for n0 in range(0, 1024, 128):
    # K^T chunk (D,Nb) with strides (1, HD=4096)
    block_ptr_kT = tl.make_block_ptr(
      base=k + off_hD + n0 * 4096,
      shape=(128, 128),
      strides=(1, 4096),
      offsets=(0, 0),
      block_shape=(128, 128),
      order=(0, 1),
    )
    k_chunk = tl.load(block_ptr_kT)  # (128,128)

    scores = tl.dot(q_tile, k_chunk)              # (16,128)
    e = tl.math.exp2(scores.to(tl.float32))       # (16,128)
    acc += tl.sum(e, axis=1, keep_dims=True).to(tl.float32)

  tl.store(block_ptr_out, acc)


@triton.autotune(configs=[triton.Config({}, num_warps=4), triton.Config({}, num_warps=8)], key=[])
@triton.jit
def KeyFormer_p7_kernel_llama(
  q,        # (1,16,32,128)
  k,        # (1,1024,32,128)
  g,        # (1,32,1024,1024)  # 필요 시 사용
  denom_m,  # (1,32,16,1)
  out_sum   # (1,32,16)
):
  pid_h = tl.program_id(2)  # head 0..31
  const_scale = 1.0 / 11.3125

  off_hD = pid_h * 128  # D=128

  # Q (M=16, D=128), strides (HD=4096,1)
  block_ptr_q = tl.make_block_ptr(
    base=q + off_hD,
    shape=(16, 128),
    strides=(4096, 1),
    offsets=(0, 0),
    block_shape=(16, 128),
    order=(1, 0),
  )
  q_tile = tl.load(block_ptr_q).to(tl.float32) * const_scale
  q_tile = q_tile.to(tl.float16)

  # denom_m (M,1)
  block_ptr_dm = tl.make_block_ptr(
    base=denom_m + pid_h * 16,
    shape=(16, 1),
    strides=(1, 1),
    offsets=(0, 0),
    block_shape=(16, 1),
    order=(1, 0),
  )
  dm = tl.load(block_ptr_dm)  # (16,1)

  # out (M)
  block_ptr_out = tl.make_block_ptr(
    base=out_sum + pid_h * 16,
    shape=(16,),
    strides=(1,),
    offsets=(0,),
    block_shape=(16,),
    order=(0,),
  )
  acc = tl.zeros([16], dtype=tl.float32)

  for n0 in range(0, 1024, 128):
    # K^T (D,Nb=128), strides (1, HD=4096)
    block_ptr_kT = tl.make_block_ptr(
      base=k + off_hD + n0 * 4096,
      shape=(128, 128),
      strides=(1, 4096),
      offsets=(0, 0),
      block_shape=(128, 128),
      order=(0, 1),
    )
    k_chunk = tl.load(block_ptr_kT)  # (128,128)

    scores = tl.dot(q_tile, k_chunk)      # (16,128)
    p = tl.math.exp2(scores.to(tl.float32)) / dm
    acc += tl.sum(p, axis=1, keep_dims=False).to(tl.float32)

  tl.store(block_ptr_out, acc)

# def bench_KeyFormer_llama():
#   t1 = bench_KeyFormer_p1_llama()
#   t2 = bench_KeyFormer_p9_llama()
#   t3 = bench_KeyFormer_p7_llama()

#   return t1+t2+t3

#====== falcon ======

def KeyFormer_p1_falcon(q: torch.Tensor, v: torch.Tensor, k: torch.Tensor, out) -> torch.Tensor:
  # out: (1,16,71,64)
  # dev = q.device
  # out = torch.empty_like(q)
  grid = (1, 1, 71)  # (B tiles, M tiles, H heads)
  # 커널 호출은 대괄호 인덱싱 형태로!
  KeyFormer_p1_kernel_falcon[grid](
      q, v, k, out,
      CONST_HD=71*64,   # 4544
      BLOCK_D=64        # tl.constexpr
  )
  return out

@triton.autotune(configs=[triton.Config({}, num_warps=4), triton.Config({}, num_warps=8)],
                 key=[])

@triton.jit
def KeyFormer_p1_kernel_falcon(
  arg_q,     # (1,16,H,D)
  arg_v,     # (1,1024,H,D)
  arg_k,     # (1,1024,H,D)
  arg_out,   # (1,16,H,D)
  CONST_HD: tl.constexpr,   # H*D (4096 or 4544)
  BLOCK_D: tl.constexpr      # 128 or 64
):
  # fixed problem sizes
  CONST_M : tl.constexpr = 16
  CONST_N : tl.constexpr = 1024
  BLOCK_M : tl.constexpr = 16
  BLOCK_N : tl.constexpr = 128

  pid_b = tl.program_id(0)      # unused (B=1)
  pid_m = tl.program_id(1)      # unused (one M tile since M=16)
  pid_h = tl.program_id(2)      # 0..H-1

  # scale ~ 1/sqrt(D): 원 코드 값 사용
  const_scale = 1.275311e-01 if BLOCK_D == 128 else (1.0 / 8.0)  # 0.125

  # per-head offset (H axis contiguous by D)
  off_hD = pid_h * BLOCK_D

  # Q: (M,D) with strides (HD,1)
  block_ptr_q = tl.make_block_ptr(
    base=arg_q + off_hD,
    shape=(BLOCK_M, BLOCK_D),
    strides=(CONST_HD, 1),
    offsets=(0, 0),
    block_shape=(BLOCK_M, BLOCK_D),
    order=(1, 0),
  )
  q = tl.load(block_ptr_q).to(tl.float32) * const_scale
  q = q.to(tl.float16)  # (16,D)

  # K^T: (D,N) with strides (1,HD)
  block_ptr_kT = tl.make_block_ptr(
    base=arg_k + off_hD,
    shape=(BLOCK_D, CONST_N),
    strides=(1, CONST_HD),
    offsets=(0, 0),
    block_shape=(BLOCK_D, BLOCK_N),
    order=(0, 1),
  )

  # V: (N,D) with strides (HD,1)
  block_ptr_v = tl.make_block_ptr(
    base=arg_v + off_hD,
    shape=(CONST_N, BLOCK_D),
    strides=(CONST_HD, 1),
    offsets=(0, 0),
    block_shape=(BLOCK_N, BLOCK_D),
    order=(1, 0),
  )

  # OUT, DENOM
  block_ptr_out = tl.make_block_ptr(
    base=arg_out + off_hD,
    shape=(BLOCK_M, BLOCK_D),
    strides=(CONST_HD, 1),
    offsets=(0, 0),
    block_shape=(BLOCK_M, BLOCK_D),
    order=(1, 0),
  )

  acc_out   = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
  acc_denom = tl.zeros([BLOCK_M, 1],       dtype=tl.float32)

  # loop over N tiles
  for n0 in range(0, CONST_N, BLOCK_N):
    kT_tile = tl.load(block_ptr_kT)                 # (D,BN)
    scores  = tl.dot(q, kT_tile)                    # (M,BN)
    probs   = tl.math.exp2(scores)                  # (M,BN)
    acc_denom += tl.sum(probs, axis=1, keep_dims=True).to(tl.float32)
    v_tile  = tl.load(block_ptr_v)                  # (BN,D)
    acc_out += tl.dot(probs.to(tl.float16), v_tile) # (M,D)
    block_ptr_kT = tl.advance(block_ptr_kT, (0, BLOCK_N))
    block_ptr_v  = tl.advance(block_ptr_v,  (BLOCK_N, 0))

  out = (acc_out / acc_denom).to(tl.float16)
  tl.store(block_ptr_out, out)


def KeyFormer_p9_falcon(q: torch.Tensor, k: torch.Tensor, g: torch.Tensor, out) -> torch.Tensor:
  # dev = q.device
  # out = torch.empty(1, 71, 16, 1, dtype=torch.float32, device=dev)
  grid = (1, 1, 71)
  KeyFormer_p9_kernel_falcon[grid](q, k, g, out)
  return out

@triton.autotune(configs=[triton.Config({}, num_warps=4), triton.Config({}, num_warps=8)], key=[])
@triton.jit
def KeyFormer_p9_kernel_falcon(
  q,   # (1,16,71,64)
  k,   # (1,1024,71,64)
  g,   # (1,71,1024,1024)
  out  # (1,71,16,1)
):
  pid_h = tl.program_id(2)  # head 0..70

  const_scale = 1.0 / 8.0  # 1/sqrt(64)
  CONST_HD = 71 * 64
  CONST_M  = 16
  CONST_N  = 1024
  BLOCK_D  = 64
  BLOCK_M  = 16
  BLOCK_N  = 128

  off_hD = pid_h * BLOCK_D

  # Q (M,D)
  block_ptr_q = tl.make_block_ptr(
    base=q + off_hD,
    shape=(16, 64),
    strides=(4544, 1),
    offsets=(0, 0),
    block_shape=(16, 64),
    order=(1, 0),
  )
  q_tile = tl.load(block_ptr_q).to(tl.float32) * const_scale
  q_tile = q_tile.to(tl.float16)

  # out denom (M,1)
  block_ptr_out = tl.make_block_ptr(
    base=out + pid_h * CONST_M,
    shape=(16, 1),
    strides=(1, 1),
    offsets=(0, 0),
    block_shape=(16, 1),
    order=(1, 0),
  )
  acc = tl.zeros([16, 1], dtype=tl.float32)

  for n0 in range(0, CONST_N, BLOCK_N):
    # K^T chunk (D,Nb)
    block_ptr_kT = tl.make_block_ptr(
      base=k + off_hD + n0 * CONST_HD,
      shape=(64, 128),
      strides=(1, 4544),
      offsets=(0, 0),
      block_shape=(64, 128),
      order=(0, 1),
    )
    k_chunk = tl.load(block_ptr_kT)
    k_scaled = (k_chunk.to(tl.float32) * const_scale).to(tl.float16)

    scores = tl.dot(q_tile, k_chunk)  # (16,128)
    e = tl.math.exp2(scores.to(tl.float32))
    acc += tl.sum(e, axis=1, keep_dims=True).to(tl.float32)

  tl.store(block_ptr_out, acc)

def KeyFormer_p7_falcon(q: torch.Tensor, k: torch.Tensor, g: torch.Tensor, denom_m: torch.Tensor, out) -> torch.Tensor:
  # dev = q.device
  # out = torch.empty(1, 71, 16, dtype=torch.float32, device=dev)
  grid = (1, 1, 71)
  KeyFormer_p7_kernel_falcon[grid](q, k, g, denom_m, out)
  return out

@triton.autotune(configs=[triton.Config({}, num_warps=4), triton.Config({}, num_warps=8)], key=[])
@triton.jit
def KeyFormer_p7_kernel_falcon(
  q,        # (1,16,71,64)
  k,        # (1,1024,71,64)
  g,        # (1,71,1024,1024)
  denom_m,  # (1,71,16,1)
  out_sum   # (1,71,16)
):
  pid_h = tl.program_id(2)  # 0..70

  const_scale = 1.0 / 8.0
  CONST_HD = 71 * 64
  CONST_M  = 16
  CONST_N  = 1024
  BLOCK_D  = 64
  BLOCK_M  = 16
  BLOCK_N  = 128

  off_hD = pid_h * BLOCK_D

  block_ptr_q = tl.make_block_ptr(
    base=q + off_hD,
    shape=(16, 64),
    strides=(4544, 1),
    offsets=(0, 0),
    block_shape=(16, 64),
    order=(1, 0),
  )
  q_tile = tl.load(block_ptr_q).to(tl.float32) * const_scale
  q_tile = q_tile.to(tl.float16)

  block_ptr_dm = tl.make_block_ptr(
    base=denom_m + pid_h * CONST_M,
    shape=(16, 1),
    strides=(1, 1),
    offsets=(0, 0),
    block_shape=(16, 1),
    order=(1, 0),
  )
  dm = tl.load(block_ptr_dm)  # (16,1)

  block_ptr_out = tl.make_block_ptr(
    base=out_sum + pid_h * CONST_M,
    shape=(16,),
    strides=(1,),
    offsets=(0,),
    block_shape=(16,),
    order=(0,),
  )
  acc = tl.zeros([16], dtype=tl.float32)

  for n0 in range(0, CONST_N, BLOCK_N):
    block_ptr_kT = tl.make_block_ptr(
      base=k + off_hD + n0 * 4544,
      shape=(64, 128),
      strides=(1, 4544),
      offsets=(0, 0),
      block_shape=(64, 128),
      order=(0, 1),
    )
    k_chunk = tl.load(block_ptr_kT)
    scores = tl.dot(q_tile, k_chunk)  # (16,128)
    p = tl.math.exp2(scores.to(tl.float32)) / dm
    acc += tl.sum(p, axis=1, keep_dims=False).to(tl.float32)

  tl.store(block_ptr_out, acc)

# def bench_KeyFormer_falcon():
#   t1 = bench_KeyFormer_p1_falcon()
#   t2 = bench_KeyFormer_p9_falcon()
#   t3 = bench_KeyFormer_p7_falcon()

#   return t1+t2+t3