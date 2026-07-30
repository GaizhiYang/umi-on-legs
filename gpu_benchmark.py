#!/usr/bin/env python3
"""Small, reproducible CUDA benchmark based on PyTorch.

Measures matrix-multiplication throughput and approximate device-memory copy
bandwidth.  This is a synthetic benchmark; results are most useful when the
same command and software environment are used to compare GPUs.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a CUDA GPU with PyTorch")
    parser.add_argument("--device", default="cuda:0", help="CUDA device, e.g. cuda:0")
    parser.add_argument(
        "--size",
        type=int,
        default=4096,
        help="square matrix size used for GEMM (default: 4096)",
    )
    parser.add_argument(
        "--copy-mb",
        type=int,
        default=512,
        help="amount of data copied per bandwidth iteration (default: 512 MiB)",
    )
    parser.add_argument("--warmup", type=int, default=10, help="warm-up iterations")
    parser.add_argument("--iterations", type=int, default=30, help="timed iterations")
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=("float32", "float16", "bfloat16"),
        default=("float32", "float16", "bfloat16"),
        help="GEMM data types to test",
    )
    parser.add_argument(
        "--no-tf32",
        action="store_true",
        help="disable TF32 for float32 matrix multiplication",
    )
    parser.add_argument("--json", type=Path, help="also write results to this JSON file")
    args = parser.parse_args()
    if args.size <= 0 or args.copy_mb <= 0:
        parser.error("--size and --copy-mb must be positive")
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("--warmup must be non-negative and --iterations must be positive")
    return args


def load_torch() -> Any:
    try:
        import torch
    except ImportError:
        print("错误：未安装 PyTorch。请安装支持 CUDA 的 PyTorch 后再运行。", file=sys.stderr)
        print("安装说明：https://pytorch.org/get-started/locally/", file=sys.stderr)
        raise SystemExit(2)
    return torch


def cuda_time_ms(torch: Any, operation: Callable[[], None], iterations: int) -> list[float]:
    """Time individual asynchronous CUDA operations with CUDA events."""
    timings = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        timings.append(float(start.elapsed_time(end)))
    return timings


def describe_times(times_ms: list[float]) -> dict[str, float]:
    ordered = sorted(times_ms)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return {
        "median_ms": statistics.median(ordered),
        "mean_ms": statistics.mean(ordered),
        "min_ms": min(ordered),
        "p95_ms": ordered[p95_index],
    }


def benchmark_gemm(
    torch: Any,
    device: Any,
    size: int,
    dtype_name: str,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    dtype = getattr(torch, dtype_name)
    element_bytes = torch.tensor([], dtype=dtype).element_size()
    required_bytes = 3 * size * size * element_bytes
    free_bytes, _ = torch.cuda.mem_get_info(device)
    if required_bytes > free_bytes * 0.85:
        return {
            "dtype": dtype_name,
            "status": "skipped",
            "reason": (
                f"estimated {required_bytes / 2**30:.2f} GiB required, "
                f"only {free_bytes / 2**30:.2f} GiB free"
            ),
        }

    try:
        a = torch.randn((size, size), device=device, dtype=dtype)
        b = torch.randn((size, size), device=device, dtype=dtype)
        out = torch.empty((size, size), device=device, dtype=dtype)
        operation = lambda: torch.mm(a, b, out=out)
        for _ in range(warmup):
            operation()
        torch.cuda.synchronize(device)
        times = cuda_time_ms(torch, operation, iterations)
        stats = describe_times(times)
        # Dense NxN GEMM performs approximately 2*N^3 floating-point ops.
        stats["tflops"] = 2 * size**3 / (stats["median_ms"] / 1000) / 1e12
        return {"dtype": dtype_name, "status": "ok", **stats}
    except RuntimeError as exc:
        return {"dtype": dtype_name, "status": "skipped", "reason": str(exc).splitlines()[0]}
    finally:
        for name in ("a", "b", "out"):
            if name in locals():
                del locals()[name]
        torch.cuda.empty_cache()


def benchmark_copy(
    torch: Any,
    device: Any,
    copy_mb: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    requested_bytes = copy_mb * 2**20
    free_bytes, _ = torch.cuda.mem_get_info(device)
    # Two buffers are needed; shrink automatically on low-memory GPUs.
    bytes_per_buffer = min(requested_bytes, int(free_bytes * 0.35))
    elements = bytes_per_buffer // 4
    bytes_per_buffer = elements * 4
    if elements == 0:
        return {"status": "skipped", "reason": "not enough free device memory"}

    try:
        source = torch.empty(elements, device=device, dtype=torch.float32)
        destination = torch.empty_like(source)
        operation = lambda: destination.copy_(source)
        for _ in range(warmup):
            operation()
        torch.cuda.synchronize(device)
        times = cuda_time_ms(torch, operation, iterations)
        stats = describe_times(times)
        # One read plus one write crosses the device memory interface.
        stats["bandwidth_gb_s"] = (2 * bytes_per_buffer) / (stats["median_ms"] / 1000) / 1e9
        return {
            "status": "ok",
            "buffer_mib": bytes_per_buffer / 2**20,
            **stats,
        }
    except RuntimeError as exc:
        return {"status": "skipped", "reason": str(exc).splitlines()[0]}
    finally:
        for name in ("source", "destination"):
            if name in locals():
                del locals()[name]
        torch.cuda.empty_cache()


def print_report(results: dict[str, Any]) -> None:
    system = results["system"]
    print(f"GPU:       {system['gpu_name']}")
    print(f"CUDA:      {system['cuda_version']} (PyTorch {system['torch_version']})")
    print(f"显存:      {system['total_memory_gib']:.2f} GiB")
    print(f"TF32:      {'开启' if system['tf32_enabled'] else '关闭'}")
    print(f"矩阵大小:  {results['settings']['matrix_size']} x {results['settings']['matrix_size']}")
    print()
    print(f"{'测试':<18} {'中位耗时':>12} {'性能':>16} {'P95 耗时':>12}")
    print("-" * 62)
    for item in results["gemm"]:
        label = f"GEMM {item['dtype']}"
        if item["status"] == "ok":
            print(
                f"{label:<18} {item['median_ms']:>9.3f} ms "
                f"{item['tflops']:>11.2f} TFLOPS {item['p95_ms']:>9.3f} ms"
            )
        else:
            print(f"{label:<18} 跳过：{item['reason']}")
    copy = results["memory_copy"]
    if copy["status"] == "ok":
        print(
            f"{'显存复制':<18} {copy['median_ms']:>9.3f} ms "
            f"{copy['bandwidth_gb_s']:>11.2f} GB/s {copy['p95_ms']:>9.3f} ms"
        )
    else:
        print(f"{'显存复制':<18} 跳过：{copy['reason']}")


def main() -> int:
    args = parse_args()
    torch = load_torch()
    if not torch.cuda.is_available():
        print("错误：PyTorch 未检测到可用的 CUDA GPU。", file=sys.stderr)
        print("请检查 NVIDIA 驱动，以及 PyTorch 是否为 CUDA 版本。", file=sys.stderr)
        return 2

    try:
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        properties = torch.cuda.get_device_properties(device)
    except (RuntimeError, ValueError) as exc:
        print(f"错误：无法使用设备 {args.device}: {exc}", file=sys.stderr)
        return 2

    torch.backends.cuda.matmul.allow_tf32 = not args.no_tf32
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if not args.no_tf32 else "highest")

    results: dict[str, Any] = {
        "system": {
            "gpu_name": properties.name,
            "device": str(device),
            "total_memory_gib": properties.total_memory / 2**30,
            "cuda_version": torch.version.cuda,
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "tf32_enabled": bool(torch.backends.cuda.matmul.allow_tf32),
        },
        "settings": {
            "matrix_size": args.size,
            "copy_mib": args.copy_mb,
            "warmup": args.warmup,
            "iterations": args.iterations,
        },
        "gemm": [],
    }

    print("正在预热并测试 GPU，请稍候……", flush=True)
    benchmark_started = time.monotonic()
    for dtype_name in args.dtypes:
        results["gemm"].append(
            benchmark_gemm(
                torch, device, args.size, dtype_name, args.warmup, args.iterations
            )
        )
    results["memory_copy"] = benchmark_copy(
        torch, device, args.copy_mb, args.warmup, args.iterations
    )
    results["elapsed_seconds"] = time.monotonic() - benchmark_started

    print_report(results)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
        print(f"\nJSON 结果已写入：{args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
