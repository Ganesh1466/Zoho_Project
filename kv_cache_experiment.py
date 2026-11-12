"""Measure OpenVINO GenAI inference across chat and prompt-length scenarios."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from benchmark import process_memory_bytes
from localmind import DEFAULT_MAX_NEW_TOKENS, LocalMindEngine


CSV_FIELDS = [
    "scenario",
    "turn_index",
    "conversation_turns_before_request",
    "device",
    "prompt_chars",
    "input_tokens",
    "output_tokens",
    "ttft_ms",
    "tpot_ms",
    "throughput_tokens_per_second",
    "generation_time_ms",
    "inference_time_ms",
    "process_peak_working_set_bytes",
    "process_private_bytes",
    "kv_cache_control",
    "kv_cache_reuse_metric",
]


def record(
    engine: LocalMindEngine,
    scenario: str,
    turn_index: int,
    prior_turns: int,
    prompt: str,
    metrics: dict[str, int | float],
) -> dict[str, Any]:
    peak_working_set, private_bytes = process_memory_bytes()
    return {
        "scenario": scenario,
        "turn_index": turn_index,
        "conversation_turns_before_request": prior_turns,
        "device": engine.device,
        "prompt_chars": len(prompt),
        **metrics,
        "process_peak_working_set_bytes": peak_working_set,
        "process_private_bytes": private_bytes,
        "kv_cache_control": "OpenVINO GenAI runtime-managed; no verified disable switch",
        "kv_cache_reuse_metric": "",
    }


def measure_conversation(
    engine: LocalMindEngine,
    scenario: str,
    prompts: list[str],
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    history = engine.new_history()
    rows = []
    for index, prompt in enumerate(prompts, start=1):
        prior_turns = (len(history.get_messages()) // 2)
        _, metrics = engine.respond(history, prompt, max_new_tokens)
        rows.append(record(engine, scenario, index, prior_turns, prompt, metrics))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=str(Path(__file__).resolve().parent / "qwen-3b-openvino-int4"),
    )
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "benchmarks" / "kv_cache_experiments.csv"),
    )
    args = parser.parse_args()
    if not Path(args.model).is_dir():
        parser.error(f"Model directory does not exist: {args.model}")

    engine = LocalMindEngine(args.model, args.device)
    print(f"Loaded model in {engine.load_time_ms:.1f} ms; running warm-up.")
    engine.respond(engine.new_history(), "Please answer briefly: ready?", args.max_new_tokens)

    scenarios = [
        ("single_turn", ["What is OpenVINO?"]),
        (
            "multi_turn_2",
            [
                "I am learning about local language models.",
                "What does local inference mean in that context?",
            ],
        ),
        (
            "multi_turn_4",
            [
                "I am building a local chatbot.",
                "It runs on CPU.",
                "The model is Qwen 2.5 3B INT4.",
                "Summarize the project details I gave you.",
            ],
        ),
        (
            "repeated_prefix_first_request",
            ["Please answer as a short numbered list. Explain CPU model inference."],
        ),
        (
            "repeated_prefix_second_request",
            ["Please answer as a short numbered list. Explain OpenVINO model loading."],
        ),
        (
            "longer_prompt",
            [
                "Summarize these repeated context statements in one short paragraph: "
                + ("OpenVINO runs the local INT4 model on CPU. " * 18)
            ],
        ),
    ]

    rows: list[dict[str, Any]] = []
    for scenario, prompts in scenarios:
        print(f"Measuring {scenario}...")
        rows.extend(measure_conversation(engine, scenario, prompts, args.max_new_tokens))

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} measured calls to {output}")
    for scenario in sorted({str(row["scenario"]) for row in rows}):
        selected = [row for row in rows if row["scenario"] == scenario]
        ttft = [float(row["ttft_ms"]) for row in selected]
        throughput = [float(row["throughput_tokens_per_second"]) for row in selected]
        print(
            f"{scenario}: median TTFT={statistics.median(ttft):.1f} ms; "
            f"median throughput={statistics.median(throughput):.2f} tokens/s"
        )
    if sys.platform != "win32":
        print("Process memory counters are unavailable on this platform.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
