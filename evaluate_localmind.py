"""Run the repeatable LocalMind response and conversation-context evaluation."""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from localmind import DEFAULT_MAX_NEW_TOKENS, MAX_NEW_TOKENS_LIMIT


ROOT = Path(__file__).resolve().parent
CASES_PATH = ROOT / "tests" / "evaluation_cases.json"
RUBRIC_FIELDS = (
    "context_relevance",
    "factual_correctness",
    "completeness",
    "clarity",
    "hallucination",
    "conversation_continuity",
)


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=300) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = str(exc)
        exc.close()
        raise RuntimeError(
            f"LocalMind evaluation request failed for {url}: {detail}"
        ) from exc
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"LocalMind evaluation request failed for {url}: {exc}") from exc
    if not isinstance(result, dict):
        raise RuntimeError("The LocalMind API returned a non-object JSON response.")
    return result


def get_json(url: str) -> dict[str, Any]:
    try:
        with urlopen(url, timeout=10) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = str(exc)
        exc.close()
        raise RuntimeError(
            f"LocalMind evaluation request failed for {url}: {detail}"
        ) from exc
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"LocalMind evaluation request failed for {url}: {exc}") from exc
    if not isinstance(result, dict):
        raise RuntimeError("The LocalMind API returned a non-object JSON response.")
    return result


def run_evaluation(
    base_url: str,
    max_new_tokens: int,
    label: str,
) -> dict[str, Any]:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    results = []
    for case in cases:
        session_id = str(uuid.uuid4())
        turns = []
        for turn_index, turn in enumerate(case["turns"], start=1):
            result = post_json(
                f"{base_url.rstrip('/')}/api/chat",
                {
                    "session_id": session_id,
                    "message": turn["message"],
                    "max_new_tokens": max_new_tokens,
                },
            )
            metrics = result.get("metrics")
            if not isinstance(metrics, dict) or not isinstance(
                result.get("response"), str
            ):
                raise RuntimeError(
                    f"Malformed LocalMind response in case {case['id']}, "
                    f"turn {turn_index}."
                )
            metrics.setdefault(
                "output_limit_reached",
                metrics.get("output_tokens", 0) >= max_new_tokens,
            )
            turns.append(
                {
                    "turn": turn_index,
                    "user": turn["message"],
                    "expected_concepts": turn.get("expected_concepts", []),
                    "avoid": turn.get("avoid", []),
                    "assistant": result["response"],
                    "metrics": metrics,
                }
            )
        post_json(
            f"{base_url.rstrip('/')}/api/reset",
            {"session_id": session_id},
        )
        results.append(
            {
                "id": case["id"],
                "domain": case["domain"],
                "turns": turns,
                "manual_review_scores_1_to_5": {
                    field: None for field in RUBRIC_FIELDS
                },
            }
        )
        final_turn = turns[-1]
        print(
            f"{case['id']}: output={final_turn['metrics'].get('output_tokens')} "
            f"tokens, limit_reached={final_turn['metrics'].get('output_limit_reached')}, "
            f"response={final_turn['assistant'][:140].replace(chr(10), ' ')}"
        )

    return {
        "label": label,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url.rstrip("/"),
        "max_new_tokens": max_new_tokens,
        "rubric_scale": "1-5; complete manual review before claiming quality scores",
        "cases": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--label", default="candidate")
    parser.add_argument(
        "--output",
        default=str(ROOT / "performance_experiments" / "evaluation_results.json"),
    )
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= MAX_NEW_TOKENS_LIMIT:
        parser.error(
            f"--max-new-tokens must be between 1 and {MAX_NEW_TOKENS_LIMIT}."
        )

    status = get_json(f"{args.url.rstrip('/')}/api/status")
    if not status.get("ready"):
        raise RuntimeError("The LocalMind API is not ready.")
    results = run_evaluation(args.url, args.max_new_tokens, args.label)
    results["model"] = status.get("model_name")
    results["device"] = status.get("device")

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved evaluation results to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
