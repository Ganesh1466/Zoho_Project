"""Reproducible, metrics-backed INT4/FP16 OpenVINO model comparison."""

from __future__ import annotations

import argparse
import csv
import ctypes
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from localmind import MAX_NEW_TOKENS_LIMIT, LocalMindEngine
from performance_experiments.profile_local_api import profile_api


PROMPTS = {
    "explain_openvino": "In two short paragraphs, explain what OpenVINO is used for.",
    "explain_entc": "List three common ENTC engineering subjects and two career areas.",
}

CSV_FIELDS = [
    "variant",
    "model_path",
    "device",
    "prompt_id",
    "run_index",
    "warmup",
    "prompt_chars",
    "input_tokens",
    "output_tokens",
    "load_time_ms",
    "ttft_ms",
    "tpot_ms",
    "throughput_tokens_per_second",
    "generation_time_ms",
    "inference_time_ms",
    "model_size_bytes",
    "process_peak_working_set_bytes",
    "process_private_bytes",
    "output_text",
]


class _ProcessMemoryCountersEx(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def process_memory_bytes() -> tuple[int | None, int | None]:
    if sys.platform != "win32":
        return None, None
    try:
        psapi = ctypes.WinDLL("psapi.dll", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
        get_process = kernel32.GetCurrentProcess
        get_process.restype = ctypes.c_void_p
        get_memory = psapi.GetProcessMemoryInfo
        get_memory.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_ProcessMemoryCountersEx),
            ctypes.c_ulong,
        ]
        get_memory.restype = ctypes.c_int
        counters = _ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(counters)
        if not get_memory(get_process(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.PeakWorkingSetSize), int(counters.PrivateUsage)
    except (AttributeError, OSError):
        return None, None


def make_row(
    engine: LocalMindEngine,
    variant: str,
    prompt_id: str,
    prompt: str,
    run_index: int,
    warmup: bool,
    text: str,
    metrics: dict[str, int | float],
) -> dict[str, Any]:
    peak_working_set, private_bytes = process_memory_bytes()
    return {
        "variant": variant,
        "model_path": str(engine.model_path),
        "device": engine.device,
        "prompt_id": prompt_id,
        "run_index": run_index,
        "warmup": warmup,
        "prompt_chars": len(prompt),
        **metrics,
        "load_time_ms": engine.load_time_ms,
        "model_size_bytes": engine.model_size_bytes,
        "process_peak_working_set_bytes": peak_working_set,
        "process_private_bytes": private_bytes,
        "output_text": text.replace("\r", " ").replace("\n", " "),
    }


def run_variant(
    model_path: str,
    variant: str,
    device: str,
    prompts: dict[str, str],
    warmup_count: int,
    run_count: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    print(f"Loading {variant} model from {Path(model_path).resolve()}...")
    engine = LocalMindEngine(model_path, device)
    rows = []

    for prompt_id, prompt in prompts.items():
        for warmup_index in range(warmup_count):
            history = engine.new_history()
            engine.respond(history, prompt, max_new_tokens)
            print(f"{variant}: warm-up {prompt_id} {warmup_index + 1}/{warmup_count}")

        for run_index in range(1, run_count + 1):
            history = engine.new_history()
            started = time.perf_counter()
            text, metrics = engine.respond(history, prompt, max_new_tokens)
            wall_time_ms = (time.perf_counter() - started) * 1000
            row = make_row(
                engine, variant, prompt_id, prompt, run_index, False, text, metrics
            )
            row["generation_time_ms"] = metrics["generation_time_ms"]
            rows.append(row)
            print(
                f"{variant}/{prompt_id} run {run_index}: "
                f"{metrics['throughput_tokens_per_second']:.2f} tokens/s, "
                f"TTFT {metrics['ttft_ms']:.1f} ms, "
                f"wall {wall_time_ms:.1f} ms"
            )

    return rows


def print_summary(rows: list[dict[str, Any]]) -> None:
    for variant in sorted({str(row["variant"]) for row in rows}):
        selected = [row for row in rows if row["variant"] == variant]
        for key in ("ttft_ms", "tpot_ms", "throughput_tokens_per_second"):
            values = [float(row[key]) for row in selected]
            print(
                f"{variant} {key}: mean={statistics.mean(values):.3f}, "
                f"median={statistics.median(values):.3f}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--int4", help="Path to the INT4 OpenVINO model.")
    parser.add_argument(
        "--fp16",
        help="Path to a compatible FP16 OpenVINO model; omitted until one is prepared.",
    )
    parser.add_argument(
        "--api-url",
        help="Profile an already-running LocalMind web API instead of loading a model.",
    )
    parser.add_argument(
        "--api-output",
        default=str(
            Path(__file__).resolve().parent
            / "performance_experiments"
            / "python_api_profile.csv"
        ),
        help="CSV output for --api-url mode.",
    )
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "benchmarks" / "precision_comparison.csv"),
    )
    parser.add_argument(
        "--results-output",
        default=str(Path(__file__).resolve().parent / "benchmarks" / "results.csv"),
    )
    args = parser.parse_args()
    if args.runs < 1 or args.warmup < 0:
        parser.error("--runs must be at least 1 and --warmup cannot be negative.")
    if not 1 <= args.max_new_tokens <= MAX_NEW_TOKENS_LIMIT:
        parser.error(
            f"--max-new-tokens must be between 1 and {MAX_NEW_TOKENS_LIMIT}."
        )
    if args.api_url:
        if args.int4 or args.fp16:
            parser.error("--api-url cannot be combined with --int4 or --fp16.")
        return profile_api(
            args.api_url,
            args.runs,
            args.warmup,
            args.max_new_tokens,
            args.api_output,
        )
    if not args.int4:
        parser.error("--int4 is required unless --api-url is supplied.")
    if args.fp16 and not Path(args.fp16).is_dir():
        parser.error(f"FP16 model directory does not exist: {args.fp16}")
    if not Path(args.int4).is_dir():
        parser.error(f"INT4 model directory does not exist: {args.int4}")

    all_rows = run_variant(
        args.int4, "INT4", args.device, PROMPTS, args.warmup, args.runs,
        args.max_new_tokens,
    )
    if args.fp16:
        all_rows.extend(
            run_variant(
                args.fp16, "FP16", args.device, PROMPTS, args.warmup, args.runs,
                args.max_new_tokens,
            )
        )
    else:
        print("FP16 was not supplied; only the existing INT4 model will be measured.")

    for output_arg in (args.output, args.results_output):
        output = Path(output_arg).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Wrote {len(all_rows)} measured runs to {output}")
    print_summary(all_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
