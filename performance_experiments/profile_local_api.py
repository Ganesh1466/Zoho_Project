"""Profile the already-running LocalMind API without loading another model."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from localmind import MAX_NEW_TOKENS_LIMIT


CSV_FIELDS = [
    "run_type",
    "run_index",
    "model_name",
    "device",
    "recorded_model_load_ms",
    "prompt_chars",
    "input_tokens",
    "context_tokens",
    "recent_turns_in_context",
    "summarized_turns",
    "context_preparation_ms",
    "output_tokens",
    "output_limit_reached",
    "client_wall_ms",
    "openvino_ttft_ms",
    "openvino_tpot_ms",
    "openvino_throughput_tokens_per_second",
    "client_throughput_tokens_per_second",
    "openvino_generation_ms",
    "openvino_inference_ms",
    "api_delivery_overhead_ms",
    "non_generation_overhead_ms",
    "server_cpu_ms",
    "server_cpu_percent_of_logical_capacity",
    "server_peak_working_set_bytes",
    "server_private_bytes",
    "server_idle_working_set_bytes",
    "server_idle_private_bytes",
    "server_post_working_set_bytes",
    "server_post_private_bytes",
    "server_generation_peak_working_set_bytes",
    "server_generation_peak_private_bytes",
    "tokenization_ms",
    "max_new_tokens",
]


def get_json(url: str, payload: dict[str, Any] | None = None) -> tuple[dict[str, Any], float]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {} if data is None else {"Content-Type": "application/json"}
    request = Request(url, data=data, headers=headers)
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=300) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"LocalMind API request failed for {url}: {exc}") from exc
    return body, (time.perf_counter() - started) * 1000


def listener_pid(port: int) -> int | None:
    if os.name != "nt":
        return None
    command = (
        f"(Get-NetTCPConnection -State Listen -LocalPort {port} "
        "-ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty OwningProcess)"
    )
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        return int(result.stdout.strip())
    except ValueError:
        return None


def windows_process_counters(
    pid: int | None,
) -> tuple[float | None, int | None, int | None, int | None]:
    if os.name != "nt" or pid is None:
        return None, None, None, None

    import ctypes
    from ctypes import wintypes

    class FileTime(ctypes.Structure):
        _fields_ = [("low", wintypes.DWORD), ("high", wintypes.DWORD)]

    class MemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
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

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    open_process = kernel32.OpenProcess
    open_process.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    open_process.restype = wintypes.HANDLE
    handle = open_process(0x1000, False, pid)
    if not handle:
        return None, None, None, None
    try:
        creation, exit_time, kernel_time, user_time = FileTime(), FileTime(), FileTime(), FileTime()
        get_times = kernel32.GetProcessTimes
        get_times.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(FileTime),
            ctypes.POINTER(FileTime),
            ctypes.POINTER(FileTime),
            ctypes.POINTER(FileTime),
        ]
        get_times.restype = wintypes.BOOL
        cpu_ms = None
        if get_times(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel_time),
            ctypes.byref(user_time),
        ):
            kernel_ticks = (kernel_time.high << 32) | kernel_time.low
            user_ticks = (user_time.high << 32) | user_time.low
            cpu_ms = (kernel_ticks + user_ticks) / 10_000

        counters = MemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_memory = psapi.GetProcessMemoryInfo
        get_memory.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(MemoryCounters),
            wintypes.DWORD,
        ]
        get_memory.restype = wintypes.BOOL
        if not get_memory(handle, ctypes.byref(counters), counters.cb):
            return cpu_ms, None, None, None
        return (
            cpu_ms,
            int(counters.WorkingSetSize),
            int(counters.PeakWorkingSetSize),
            int(counters.PrivateUsage),
        )
    finally:
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [wintypes.HANDLE]
        close_handle.restype = wintypes.BOOL
        close_handle(handle)


def measure(
    base_url: str,
    model: dict[str, Any],
    pid: int | None,
    logical_cpus: int,
    run_type: str,
    run_index: int,
    session_id: str,
    prompt: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    payload = {
        "session_id": session_id,
        "message": prompt,
        "max_new_tokens": max_new_tokens,
    }
    cpu_before, working_set_before, _, private_before = windows_process_counters(pid)
    memory_samples = []
    if working_set_before is not None or private_before is not None:
        memory_samples.append((working_set_before, private_before))
    sample_stop = threading.Event()

    def sample_memory() -> None:
        while not sample_stop.wait(0.1):
            _, working_set, _, private = windows_process_counters(pid)
            memory_samples.append((working_set, private))

    sampler = None
    if pid is not None:
        sampler = threading.Thread(target=sample_memory, daemon=True)
        sampler.start()
    try:
        result, wall_ms = get_json(f"{base_url}/api/chat", payload)
    finally:
        sample_stop.set()
        if sampler is not None:
            sampler.join()
    cpu_after, working_set_after, peak_working_set, private_after = (
        windows_process_counters(pid)
    )
    memory_samples.append((working_set_after, private_after))
    sampled_working_sets = [
        working_set for working_set, _ in memory_samples if working_set is not None
    ]
    sampled_private = [
        private for _, private in memory_samples if private is not None
    ]
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError("The chat API response did not contain a metrics object.")
    required_metrics = {
        "generation_time_ms",
        "output_tokens",
        "throughput_tokens_per_second",
    }
    missing_metrics = required_metrics.difference(metrics)
    if missing_metrics:
        missing = ", ".join(sorted(missing_metrics))
        raise RuntimeError(f"The chat API metrics were missing required fields: {missing}.")
    invalid_metrics = [
        key
        for key in required_metrics
        if type(metrics[key]) not in (int, float)
    ]
    if invalid_metrics:
        invalid = ", ".join(sorted(invalid_metrics))
        raise RuntimeError(f"The chat API metrics contained invalid values: {invalid}.")

    cpu_ms = None if cpu_before is None or cpu_after is None else cpu_after - cpu_before
    cpu_percent = (
        None
        if cpu_ms is None or wall_ms <= 0
        else 100 * cpu_ms / (wall_ms * max(logical_cpus, 1))
    )
    generation_ms = float(metrics["generation_time_ms"])
    return {
        "run_type": run_type,
        "run_index": run_index,
        "model_name": model.get("model_name"),
        "device": model.get("device"),
        "recorded_model_load_ms": model.get("load_time_ms"),
        "prompt_chars": len(prompt),
        "input_tokens": metrics.get("input_tokens"),
        "context_tokens": metrics.get("context_tokens"),
        "recent_turns_in_context": metrics.get("recent_turns_in_context"),
        "summarized_turns": metrics.get("summarized_turns"),
        "context_preparation_ms": metrics.get("context_preparation_ms"),
        "output_tokens": metrics.get("output_tokens"),
        "output_limit_reached": metrics.get("output_limit_reached"),
        "client_wall_ms": round(wall_ms, 3),
        "openvino_ttft_ms": metrics.get("ttft_ms"),
        "openvino_tpot_ms": metrics.get("tpot_ms"),
        "openvino_throughput_tokens_per_second": metrics.get(
            "throughput_tokens_per_second"
        ),
        "client_throughput_tokens_per_second": (
            None
            if wall_ms <= 0
            else round(metrics["output_tokens"] / (wall_ms / 1000), 3)
        ),
        "openvino_generation_ms": generation_ms,
        "openvino_inference_ms": metrics.get("inference_time_ms"),
        "api_delivery_overhead_ms": round(wall_ms - generation_ms, 3),
        "non_generation_overhead_ms": round(
            wall_ms
            - generation_ms
            - float(metrics.get("context_preparation_ms", 0.0)),
            3,
        ),
        "server_cpu_ms": None if cpu_ms is None else round(cpu_ms, 3),
        "server_cpu_percent_of_logical_capacity": (
            None if cpu_percent is None else round(cpu_percent, 2)
        ),
        "server_peak_working_set_bytes": peak_working_set,
        "server_private_bytes": private_after,
        "server_idle_working_set_bytes": working_set_before,
        "server_idle_private_bytes": private_before,
        "server_post_working_set_bytes": working_set_after,
        "server_post_private_bytes": private_after,
        "server_generation_peak_working_set_bytes": (
            max(sampled_working_sets) if sampled_working_sets else None
        ),
        "server_generation_peak_private_bytes": (
            max(sampled_private) if sampled_private else None
        ),
        "tokenization_ms": "",
        "max_new_tokens": max_new_tokens,
    }


def profile_api(
    url: str,
    runs: int,
    warmup_count: int,
    max_new_tokens: int,
    output_path: str,
) -> int:
    if runs < 1 or warmup_count < 0:
        raise ValueError("runs must be at least 1 and warmup_count cannot be negative.")
    if not 1 <= max_new_tokens <= MAX_NEW_TOKENS_LIMIT:
        raise ValueError(
            f"max_new_tokens must be between 1 and {MAX_NEW_TOKENS_LIMIT}."
        )

    base_url = url.rstrip("/")
    model, _ = get_json(f"{base_url}/api/status")
    if not model.get("ready"):
        raise RuntimeError("The local model API is not ready.")
    pid = listener_pid(int(base_url.rsplit(":", 1)[1]))
    logical_cpus = os.cpu_count() or 1
    print(
        f"Profiling loaded {model.get('model_name')} on {model.get('device')}; "
        f"load time reported by server: {model.get('load_time_ms')} ms."
    )
    for warmup_index in range(warmup_count):
        print(f"Sending warm-up request {warmup_index + 1}/{warmup_count}.")
        get_json(
            f"{base_url}/api/chat",
            {
                "session_id": str(uuid.uuid4()),
                "message": "Reply briefly with ready.",
                "max_new_tokens": min(max_new_tokens, 8),
            },
        )

    rows = []
    short_prompt = "Explain OpenVINO briefly."
    for run_index in range(1, runs + 1):
        rows.append(
            measure(
                base_url,
                model,
                pid,
                logical_cpus,
                "warm_short_prompt",
                run_index,
                str(uuid.uuid4()),
                short_prompt,
                max_new_tokens,
            )
        )

    long_prompt = "Explain OpenVINO briefly. " * 35
    rows.append(
        measure(
            base_url,
            model,
            pid,
            logical_cpus,
            "long_prompt",
            1,
            str(uuid.uuid4()),
            long_prompt,
            max_new_tokens,
        )
    )

    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} profile records to {output}")
    for row in rows:
        print(
            f"{row['run_type']} #{row['run_index']}: "
            f"input={row['input_tokens']} tokens, "
            f"TTFT={row['openvino_ttft_ms']:.1f} ms, "
            f"OpenVINO={row['openvino_throughput_tokens_per_second']:.2f} tokens/s, "
            f"client={row['client_throughput_tokens_per_second']:.2f} tokens/s, "
            f"generation={row['openvino_generation_ms']:.1f} ms, "
            f"context prep={row['context_preparation_ms']} ms, "
            f"request overhead={row['non_generation_overhead_ms']:.1f} ms, "
            f"server CPU={row['server_cpu_percent_of_logical_capacity']}%"
        )
    throughputs = [
        float(row["openvino_throughput_tokens_per_second"])
        for row in rows
        if row["openvino_throughput_tokens_per_second"] is not None
    ]
    if throughputs:
        print(
            "OpenVINO throughput across measured API requests: "
            f"mean={statistics.mean(throughputs):.2f}, "
            f"median={statistics.median(throughputs):.2f} tokens/s"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--output",
        default=str(ROOT / "performance_experiments" / "python_api_profile.csv"),
    )
    args = parser.parse_args()
    if args.runs < 1 or args.warmup < 0:
        parser.error("--runs must be at least 1 and --warmup cannot be negative.")
    if not 1 <= args.max_new_tokens <= MAX_NEW_TOKENS_LIMIT:
        parser.error(
            f"--max-new-tokens must be between 1 and {MAX_NEW_TOKENS_LIMIT}."
        )
    return profile_api(
        args.url,
        args.runs,
        args.warmup,
        args.max_new_tokens,
        args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
