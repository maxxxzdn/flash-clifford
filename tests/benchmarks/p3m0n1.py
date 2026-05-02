import torch
torch.set_float32_matmul_precision('medium')
torch._dynamo.config.cache_size_limit = 512

from ops.p3m0n1 import gp_pga
from tests.baselines import gp_pga_torch
from tests.utils import plot_heatmap, print_results_table, run_sweep


def setup_benchmark(batch_size, num_features):
    x = torch.randn(16, batch_size, num_features).cuda().contiguous()
    y = torch.randn(16, batch_size, num_features).cuda().contiguous()
    return x, y


if __name__ == "__main__":
    assert torch.cuda.is_available()

    path = "tests/benchmarks/results/p3m0n1"

    results = run_sweep(
        gp_pga,
        gp_pga_torch,
        setup_benchmark,
        batch_sizes=[1024, 2048, 4096, 8192],
        num_features_list=[128, 256, 512, 1024],
        rep=200
    )

    print_results_table(results, "p3m0n1")

    plot_heatmap(results, 'speedup_fwd', 'Forward Pass Speedup: Triton vs PyTorch\nG(3,0,1)',
                 path + '/speedup/fwd.png')
    plot_heatmap(results, 'speedup_fwd_bwd', 'Forward + Backward Pass Speedup: Triton vs PyTorch\nG(3,0,1)',
                 path + '/speedup/fwd_bwd.png')
    plot_heatmap(results, 'mem_ratio_fwd', 'Forward Pass Memory Ratio: Triton / PyTorch\nG(3,0,1)',
                 path + '/memory/fwd.png', invert_cmap=True)
    plot_heatmap(results, 'mem_ratio_fwd_bwd', 'Forward + Backward Pass Memory Ratio: Triton / PyTorch\nG(3,0,1)',
                 path + '/memory/fwd_bwd.png', invert_cmap=True)
