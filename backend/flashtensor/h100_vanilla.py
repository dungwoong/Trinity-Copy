import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Callable, Any, Optional, Tuple

def bench_vanilla(model, M, N, P, D, H, device, dtype):
    x = torch.randn(1, M, N, dtype=dtype, device=device)
    wq = torch.randn(1, N, N, dtype=dtype, device=device)
    wk = torch.randn(1, N, N, dtype=dtype, device=device)
    wv = torch.randn(1, N, N, dtype=dtype, device=device)
    wqkv = torch.cat([wq, wk, wv], dim=2).to(device=device, dtype=dtype)
    k_cache = torch.randn((1, H, P+M, D), dtype=dtype, device=device)
    v_cache = torch.randn((1, H, P+M, D), dtype=dtype, device=device)

    denom = torch.empty(1, H, M, 1, dtype=torch.float32, device=device)
    out   = torch.empty(1, M, H, D, dtype=torch.float16, device=device)

    if model == "llama":
      attn = Attn_p3_llama
    else:
      attn = Attn_p3_falcon

    for _ in range(10):
        vanilla_kernel(x, wqkv, k_cache, v_cache, M, N, P, D, H, attn, denom, out)
    torch.cuda.synchronize()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)

    s.record()
    for _ in range(1000):
      vanilla_kernel(x, wqkv, k_cache, v_cache, M, N, P, D, H, attn, denom, out)
    e.record()
    torch.cuda.synchronize()

    avg = s.elapsed_time(e)/1000
    # print(avg)
    return avg

def vanilla_kernel(x, wqkv, k_cache, v_cache, M, N, P, D, H, attn, denom, out):
    # variance = x.pow(2).mean(-1, keepdim=True)
    # X_norm = x * torch.rsqrt(variance)

    qkv = torch.matmul(x, wqkv)
    q, k, v = torch.chunk(qkv, 3, dim=-1)

    q = q.view(1, M, H, D)
    k = k.view(1, M, H, D)
    v = v.view(1, M, H, D)

    q = q.transpose(1,2)
    k = k.transpose(1,2)
    v = v.transpose(1,2)

    k_cache[:, :, P:P+M, :] = k
    v_cache[:, :, P:P+M, :] = v

    attn(q, k_cache, v_cache, denom, out)

def Attn_p3_llama(arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor, denom, out) -> Tuple[torch.Tensor, torch.Tensor]:
  dev = arg0.device
  autotune_key = torch.cuda.get_device_capability(dev)[0]
  q = arg0
  v = arg1
  k = arg2
  # denom: (1,32,16,1), out: (1,16,32,128)
  # denom = torch.empty(1, 32, 16, 1, dtype=torch.float32, device=dev)
  # out   = torch.empty(1, 16, 32, 128, dtype=torch.float16, device=dev)
  grid = (1, 1, 64)  # (batch=1, M-tiles=1, heads=32)
  Attn_p3_kernel_llama[grid](q, v, k, denom, out, autotune_key)
  return denom, out

@triton.autotune(configs=[
  triton.Config({}, num_warps=4),
  triton.Config({}, num_warps=8),
], key=['autotune_key'])
@triton.jit
def Attn_p3_kernel_llama(
  arg_0,  # q (1,16,32,128)
  arg_1,  # v (1,1024,32,128)
  arg_2,  # k (1,1024,32,128)
  arg_3,  # denom (1,32,16,1)
  arg_4,  # out   (1,16,32,128)
  autotune_key,
):
  pid_b = tl.program_id(0)   # batch (unused, 1)
  pid_m = tl.program_id(1)   # M-tiles (unused, 1 tile)
  pid_h = tl.program_id(2)   # head 0..70

  # compile-time tile sizes
  BLOCK_M : tl.constexpr = 16   # q_len
  BLOCK_D : tl.constexpr = 128   # head_dim
  BLOCK_N : tl.constexpr = 64  # kv tile

  # problem sizes/strides
  const_scale = 1.275311e-01     # 원 코드 값 유지
  const_HD   : tl.constexpr = 4096   # 32 * 128
  const_M    : tl.constexpr = 16
  const_N    : tl.constexpr = 1024

  off_hD = pid_h * BLOCK_D  # head offset inside H*D slab

  # Q tile: (16,128) with strides (HD,1) over (M,D)
  block_ptr_q = tl.make_block_ptr(
    base=arg_0 + off_hD,
    shape=[BLOCK_M, BLOCK_D],
    strides=[const_HD, 1],
    offsets=[0, 0],
    block_shape=[BLOCK_M, BLOCK_D],
    order=[1, 0],
  )
  q = tl.load(block_ptr_q).to(tl.float32) * const_scale
  q = q.to(tl.float16)  # (16,128)

  # K^T tiles: view (D,N) with strides (1,HD), loop N by 128
  block_ptr_kT = tl.make_block_ptr(
    base=arg_2 + off_hD,
    shape=[BLOCK_D, const_N],
    strides=[1, const_HD],
    offsets=[0, 0],
    block_shape=[BLOCK_D, BLOCK_N],
    order=[0, 1],
  )

  # V tiles: (N,D) with strides (HD,1)
  block_ptr_v = tl.make_block_ptr(
    base=arg_1 + off_hD,
    shape=[const_N, BLOCK_D],
    strides=[const_HD, 1],
    offsets=[0, 0],
    block_shape=[BLOCK_N, BLOCK_D],
    order=[1, 0],
  )

  # denom/out pointers for this head
  block_ptr_denom = tl.make_block_ptr(
    base=arg_3 + pid_h * const_M,     # layout (1, H, M, 1) contiguous ⇒ H stride = M
    shape=[const_M, 1],
    strides=[1, 1],
    offsets=[0, 0],
    block_shape=[BLOCK_M, 1],
    order=[1, 0],
  )
  block_ptr_out = tl.make_block_ptr(
    base=arg_4 + off_hD,              # layout (1, M, H, D): H stride = D, M stride = H*D
    shape=[BLOCK_M, BLOCK_D],
    strides=[const_HD, 1],
    offsets=[0, 0],
    block_shape=[BLOCK_M, BLOCK_D],
    order=[1, 0],
  )

  acc_out   = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
  acc_denom = tl.zeros([BLOCK_M, 1],      dtype=tl.float32)

  # loop over N in BLOCK_N
  for n0 in range(0, const_N, BLOCK_N):
    kT_tile = tl.load(block_ptr_kT)                  # (128,128)
    scores  = tl.dot(q, kT_tile)                     # (16,128)
    probs   = tl.math.exp2(scores)                   # (16,128)
    acc_denom += tl.sum(probs, axis=1, keep_dims=True).to(tl.float32)

    v_tile  = tl.load(block_ptr_v)                   # (128,128)
    acc_out += tl.dot(probs.to(tl.float16), v_tile)  # (16,128)

    block_ptr_kT = tl.advance(block_ptr_kT, (0, BLOCK_N))
    block_ptr_v  = tl.advance(block_ptr_v,  (BLOCK_N, 0))

  out = (acc_out / acc_denom).to(tl.float16)
  tl.store(block_ptr_denom, acc_denom)
  tl.store(block_ptr_out,   out)

# def bench_Attn_llama():
#   t1 = bench_Attn_p3_llama()
#   return t1


def Attn_p3_falcon(arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor, denom, out) -> Tuple[torch.Tensor, torch.Tensor]:
  dev = arg0.device
  autotune_key = torch.cuda.get_device_capability(dev)[0]
  q = arg0
  v = arg1
  k = arg2
  # denom: (1,71,16,1), out: (1,16,71,64)
  # denom = torch.empty(1, 71, 16, 1, dtype=torch.float32, device=dev)
  # out   = torch.empty(1, 16, 71, 64, dtype=torch.float16, device=dev)
  grid = (1, 1, 71)  # (batch=1, M-tiles=1, heads=71)
  Attn_p3_kernel_falcon[grid](q, v, k, denom, out, autotune_key)
  return denom, out

@triton.autotune(configs=[
  triton.Config({}, num_warps=4),
  triton.Config({}, num_warps=8),
], key=['autotune_key'])
@triton.jit
def Attn_p3_kernel_falcon(
  arg_0,  # q (1,16,71,64)
  arg_1,  # v (1,1024,71,64)
  arg_2,  # k (1,1024,71,64)
  arg_3,  # denom (1,71,16,1)
  arg_4,  # out   (1,16,71,64)
  autotune_key,
):
  pid_b = tl.program_id(0)   # batch (unused, 1)
  pid_m = tl.program_id(1)   # M-tiles (unused, 1 tile)
  pid_h = tl.program_id(2)   # head 0..70

  # compile-time tile sizes
  BLOCK_M : tl.constexpr = 16   # q_len
  BLOCK_D : tl.constexpr = 64   # head_dim
  BLOCK_N : tl.constexpr = 128  # kv tile

  # problem sizes/strides
  const_scale = 1.275311e-01     # 원 코드 값 유지
  const_HD   : tl.constexpr = 4544   # 71 * 64
  const_M    : tl.constexpr = 16
  const_N    : tl.constexpr = 1024

  off_hD = pid_h * BLOCK_D  # head offset inside H*D slab

  # Q tile: (16,64) with strides (HD,1) over (M,D)
  block_ptr_q = tl.make_block_ptr(
    base=arg_0 + off_hD,
    shape=[BLOCK_M, BLOCK_D],
    strides=[const_HD, 1],
    offsets=[0, 0],
    block_shape=[BLOCK_M, BLOCK_D],
    order=[1, 0],
  )
  q = tl.load(block_ptr_q).to(tl.float32) * const_scale
  q = q.to(tl.float16)  # (16,64)

  # K^T tiles: view (D,N) with strides (1,HD), loop N by 128
  block_ptr_kT = tl.make_block_ptr(
    base=arg_2 + off_hD,
    shape=[BLOCK_D, const_N],
    strides=[1, const_HD],
    offsets=[0, 0],
    block_shape=[BLOCK_D, BLOCK_N],
    order=[0, 1],
  )

  # V tiles: (N,D) with strides (HD,1)
  block_ptr_v = tl.make_block_ptr(
    base=arg_1 + off_hD,
    shape=[const_N, BLOCK_D],
    strides=[const_HD, 1],
    offsets=[0, 0],
    block_shape=[BLOCK_N, BLOCK_D],
    order=[1, 0],
  )

  # denom/out pointers for this head
  block_ptr_denom = tl.make_block_ptr(
    base=arg_3 + pid_h * const_M,     # layout (1, H, M, 1) contiguous ⇒ H stride = M
    shape=[const_M, 1],
    strides=[1, 1],
    offsets=[0, 0],
    block_shape=[BLOCK_M, 1],
    order=[1, 0],
  )
  block_ptr_out = tl.make_block_ptr(
    base=arg_4 + off_hD,              # layout (1, M, H, D): H stride = D, M stride = H*D
    shape=[BLOCK_M, BLOCK_D],
    strides=[const_HD, 1],
    offsets=[0, 0],
    block_shape=[BLOCK_M, BLOCK_D],
    order=[1, 0],
  )

  acc_out   = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
  acc_denom = tl.zeros([BLOCK_M, 1],      dtype=tl.float32)

  # loop over N in BLOCK_N
  for n0 in range(0, const_N, BLOCK_N):
    kT_tile = tl.load(block_ptr_kT)                  # (64,128)
    scores  = tl.dot(q, kT_tile)                     # (16,128)
    probs   = tl.math.exp2(scores)                   # (16,128)
    acc_denom += tl.sum(probs, axis=1, keep_dims=True).to(tl.float32)

    v_tile  = tl.load(block_ptr_v)                   # (128,64)
    acc_out += tl.dot(probs.to(tl.float16), v_tile)  # (16,64)

    block_ptr_kT = tl.advance(block_ptr_kT, (0, BLOCK_N))
    block_ptr_v  = tl.advance(block_ptr_v,  (BLOCK_N, 0))

  out = (acc_out / acc_denom).to(tl.float16)
  tl.store(block_ptr_denom, acc_denom)
  tl.store(block_ptr_out,   out)

# def bench_Attn_falcon():
#   t1 = bench_Attn_p3_falcon()
  # return t1
