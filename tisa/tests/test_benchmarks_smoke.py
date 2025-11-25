import pathlib

from tisa.benchmarks.run_benchmarks import run_benchmark


def test_benchmark_smoke(tmp_path):
    config = pathlib.Path("tisa/benchmarks/datasets.yaml")
    out_dir = tmp_path / "bench"
    result = run_benchmark(config, out_dir)
    for key in ("metrics", "summary", "manifest"):
        path = pathlib.Path(result[key])
        assert path.exists()
