seq_len = 16
context_length = 1024
num_q_heads = 32
num_kv_heads = 32
head_dim = 128

attn_O1 = O2 * WO
attn_O2 = attn_O1 + X
attn_O3 = reduce_sum(sqr(attn_O2), 1)
attn_O_norm = attn_O2 / bcast(sqrt(attn_O3 / 14336), 1)

FF1a = attn_O_norm * WFF1a
FF1b = attn_O_norm * WFF1b
FF1b_silu = FF1b x sigmoid(FF1b)
FF1 = FF1a x FF1b_silu
FF2 = FF1 * WFF2
