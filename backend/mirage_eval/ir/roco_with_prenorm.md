seq_len = 16
context_length = 1024
num_q_heads = 32
num_kv_heads = 32
head_dim = 128

X2 = reduce_sum(sqr(X), 1)
X_norm = X / bcast(sqrt(X2 / 4096), 1)
Q1 = X_norm * WQ
K1 = X_norm * WK
V1 = X_norm * WV
Q2 = reshape(Q1, 16, 32, 128)
K2 = reshape(K1, 16, 32, 128)
V2 = reshape(V1, 16, 32, 128)
Q = permute(Q2, 1, 0, 2)
K = permute(K2, 1, 0, 2)
V = permute(V2, 1, 0, 2)
K_cache = concat(K_cache, K, 1)
V_cache = concat(V_cache, V, 1)

C = Q * permute(K_cache, 0, 2, 1)

C_exp = exp(C)
C_sum = reduce_sum(C_exp, 2)
C_div = C_exp / bcast(C_sum, 2)

C_out1 = reduce_sum(C_div, 1)
C_out2 = reduce_sum(sqr(C_div), 1)

O = C_div * V_cache
O1 = permute(O, 1, 0, 2)
O2 = reshape(O1, 16, 4096)

return O2, C_out1, C_out2
