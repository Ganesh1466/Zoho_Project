"""Run LocalMind as a local interactive terminal chatbot."""

from __future__ import annotations

import argparse
import sys

from localmind import (
    DEFAULT_MAX_NEW_TOKENS,
    MAX_NEW_TOKENS_LIMIT,
    LocalMindEngine,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Local Qwen chatbot powered by OpenVINO.")
    parser.add_argument("--model", help="Path to an OpenVINO GenAI model directory.")
    parser.add_argument("--device", default="CPU", help="OpenVINO device (default: CPU).")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Maximum response length, 1-{MAX_NEW_TOKENS_LIMIT}.",
    )
    args = parser.parse_args()

    try:
        engine = LocalMindEngine(args.model, args.device)
        history = engine.new_history()
    except Exception as exc:
        print(f"Could not start LocalMind: {exc}", file=sys.stderr)
        return 1

    print(f"LocalMind ready: {engine.model_path} [{engine.device}]")
    print("Commands: /help, /clear, /exit")

    while True:
        try:
            message = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0

        if not message:
            continue
        if message.lower() in {"/exit", "exit", "quit"}:
            print("Goodbye.")
            return 0
        if message.lower() == "/help":
            print("Enter a message to chat. /clear starts a new conversation; /exit quits.")
            continue
        if message.lower() == "/clear":
            history = engine.new_history()
            print("Conversation cleared.")
            continue

        try:
            response, metrics = engine.respond(history, message, args.max_new_tokens)
        except Exception as exc:
            print(f"Generation failed: {exc}", file=sys.stderr)
            continue

        print(f"\nLocalMind: {response}")
        print(
            "Metrics: "
            f"TTFT {metrics['ttft_ms']:.1f} ms; "
            f"TPOT {metrics['tpot_ms']:.1f} ms/token; "
            f"{metrics['throughput_tokens_per_second']:.2f} tokens/s"
        )
        if metrics["output_limit_reached"]:
            print(
                "Note: the response reached its output limit and may be incomplete. "
                "Increase --max-new-tokens or ask LocalMind to continue."
            )


if __name__ == "__main__":
    raise SystemExit(main())
