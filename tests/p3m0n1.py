import torch

torch.set_float32_matmul_precision('medium')

from ops.p3m0n1 import gp_pga
from tests.baselines import gp_pga_torch
from tests.utils import run_correctness_test, run_benchmark


if __name__ == "__main__":
    assert torch.cuda.is_available()

    rep = 1000
    batch_size = 4096
    num_features = 512

    x = torch.randn(16, batch_size, num_features).cuda().contiguous()
    y = torch.randn(16, batch_size, num_features).cuda().contiguous()

    run_correctness_test(gp_pga, gp_pga_torch, {'x': x, 'y': y})
    run_benchmark(gp_pga, gp_pga_torch, (x, y), rep, verbose=True)
