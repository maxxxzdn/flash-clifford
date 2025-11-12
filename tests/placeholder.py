import torch

torch.set_float32_matmul_precision('medium')

from ops.placeholder import placeholder
from tests.baselines import placeholder_torch
from tests.utils import run_correctness_test, run_benchmark


if __name__ == "__main__":
    assert torch.cuda.is_available()

    rep = 1000
    batch_size = 4096
    num_features = 512

    x = torch.randn(16, batch_size, num_features).cuda().contiguous()
    y = torch.randn(16, batch_size, num_features).cuda().contiguous()

    run_correctness_test(placeholder, placeholder_torch, {'x': x, 'y': y})
    run_benchmark(placeholder, placeholder_torch, (x, y), rep, verbose=True)
