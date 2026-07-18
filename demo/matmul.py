import sys
from pathlib import Path
from triton.testing import do_bench

import torch
import torch.nn as nn

class Matmul(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-5

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        output = torch.matmul(a, b)
        return output

if __name__ == "__main__":
    import trinity

    M, N, K = 4096, 4096, 4096

    dtype = torch.float16
    A = torch.randn((M, K), dtype=dtype, device='cuda')
    B = torch.randn((K, N), dtype=dtype, device='cuda')
    C = torch.randn((M, N), dtype=dtype, device='cuda')

    model = Matmul()
    result = trinity.optimize(model, (A, B), basename="matmul", skip_frontend=False, verbose=True, backend_max_benchmarks=512)
    print(result.kernel)
    print(result.kernel_path)
    print(result.ir_expression)
    result.kernel(A, B, C)
    ref = A @ B
    
    # skip correctness check for now

    trinity_ms = do_bench(lambda: result.kernel(A, B, C))
    torch_ms = do_bench(lambda: model(A, B))
    print(f'{trinity_ms=}, {torch_ms=}')