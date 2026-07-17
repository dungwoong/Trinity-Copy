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

Q_norm = Q / bcast(sqrt(reduce_sum(sqr(Q), 2) / 128), 2)
K_norm = K / bcast(sqrt(reduce_sum(sqr(K), 2) / 128), 2)

K_cache = concat(K_cache, K_norm, 1)
V_cache = concat(V_cache, V, 1)
C = Q_norm * permute(K_cache, 0, 2, 1)
C_exp = exp(C)
C_sum = reduce_sum(C_exp, 2)
C_div = C_exp / bcast(C_sum, 2)
O = C_div * V_cache
O1 = permute(O, 1, 0, 2)
O2 = reshape(O1, 16, 4096)
