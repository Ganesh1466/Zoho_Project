"""Shared local OpenVINO GenAI inference engine for LocalMind."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "qwen-3b-openvino-int4"
DEFAULT_MAX_NEW_TOKENS = 128
MAX_NEW_TOKENS_LIMIT = 4096
MODEL_CONTEXT_TOKENS = 32768
INPUT_CONTEXT_TOKENS = 8192
CONTEXT_SAFETY_TOKENS = 128
MAX_RECENT_TURNS = 8
MAX_SUMMARY_ITEMS = 8

SYSTEM_PROMPT = """You are LocalMind, a general-purpose assistant. Answer the user's actual question directly and use prior turns when they clarify the topic, references, or intended meaning. Use wording and conversation history as domain cues: a branch, course, or degree usually refers to an academic subject; code, arrays, or algorithms usually indicate programming; signatures or encryption usually indicate cryptography.

For abbreviations, give the context-appropriate expansion only when you know it with confidence. Never invent an expansion from the letters. If more than one meaning remains plausible, briefly state the ambiguity and ask one focused clarifying question. For definitions, give the direct meaning and only established, relevant details; do not add standards, algorithm families, or other technical claims unless asked and sure.

Be accurate and honest about uncertainty. Do not fabricate facts, sources, or actions. Unless the user asks for detail, answer a simple "What is X?" question in one to three sentences. For acronym definitions, give only the expansion and a short meaning; avoid tangential details. Keep answers concise but complete, avoid filler or repetition, match the user's language and requested format, and explain technical topics step by step when the user asks for an explanation."""


def resolve_model_path(path: str | os.PathLike[str] | None = None) -> Path:
    configured = path or os.environ.get("LOCALMIND_MODEL_PATH")
    return Path(configured).expanduser().resolve() if configured else DEFAULT_MODEL_PATH


@dataclass
class ConversationHistory:
    """Bounded, per-session source history used to build model context."""

    messages: list[dict[str, str]] = field(default_factory=list)
    summary: list[str] = field(default_factory=list)

    def append(self, message: dict[str, str]) -> None:
        self.messages.append(message)

    def pop(self) -> dict[str, str]:
        return self.messages.pop()

    def clear(self) -> None:
        self.messages.clear()
        self.summary.clear()


class LocalMindEngine:
    """Loads one local model pipeline and reuses it across chat requests."""

    def __init__(
        self,
        model_path: str | os.PathLike[str] | None = None,
        device: str = "CPU",
    ) -> None:
        self.model_path = resolve_model_path(model_path)
        if not (self.model_path / "openvino_model.xml").is_file():
            raise FileNotFoundError(
                f"OpenVINO model XML was not found in {self.model_path}. "
                "Download or configure the model directory first."
            )

        try:
            import openvino_genai as ov_genai
        except ImportError as exc:
            raise RuntimeError(
                "OpenVINO GenAI is not installed in the active Python environment."
            ) from exc

        self._ov_genai = ov_genai
        self.device = device
        self._generation_lock = threading.Lock()
        started = time.perf_counter()
        self.pipeline = ov_genai.LLMPipeline(str(self.model_path), device)
        self._tokenizer = self.pipeline.get_tokenizer()
        self.load_time_ms = (time.perf_counter() - started) * 1000
        self.model_size_bytes = sum(
            item.stat().st_size for item in self.model_path.rglob("*") if item.is_file()
        )

    def new_history(self) -> ConversationHistory:
        return ConversationHistory()

    def respond(
        self,
        history: Any,
        message: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> tuple[str, dict[str, int | float | bool]]:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("Message must contain non-whitespace text.")
        if len(message) > 4000:
            raise ValueError("Message is too long (maximum 4000 characters).")
        if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool):
            raise ValueError("max_new_tokens must be an integer.")
        if not 1 <= max_new_tokens <= MAX_NEW_TOKENS_LIMIT:
            raise ValueError(
                f"max_new_tokens must be between 1 and {MAX_NEW_TOKENS_LIMIT}."
            )

        history.append({"role": "user", "content": message.strip()})
        try:
            config = self._ov_genai.GenerationConfig()
            config.max_new_tokens = max_new_tokens
            config.do_sample = False
            config.ignore_eos = False
            config.set_eos_token_id(self._tokenizer.get_eos_token_id())
            config.repetition_penalty = 1.05
            context_started = time.perf_counter()
            model_history, context_metrics = self._build_model_history(
                history, max_new_tokens
            )
            context_metrics["context_preparation_ms"] = (
                time.perf_counter() - context_started
            ) * 1000
            with self._generation_lock:
                result = self.pipeline.generate(
                    model_history, generation_config=config
                )
            if not result.texts:
                raise RuntimeError("OpenVINO GenAI returned no generated text.")
            response = result.texts[0]
            metrics = self._metrics(result.perf_metrics)
        except Exception:
            history.pop()
            raise

        history.append({"role": "assistant", "content": response})
        metrics.update(context_metrics)
        metrics["output_limit_reached"] = metrics["output_tokens"] >= max_new_tokens
        return response, metrics

    def _build_model_history(
        self,
        history: Any,
        max_new_tokens: int,
    ) -> tuple[Any, dict[str, int | float]]:
        if not isinstance(history, ConversationHistory):
            return history, {
                "context_tokens": 0,
                "recent_turns_in_context": 0,
                "summarized_turns": 0,
            }

        input_budget = min(
            INPUT_CONTEXT_TOKENS,
            MODEL_CONTEXT_TOKENS - max_new_tokens - CONTEXT_SAFETY_TOKENS,
        )
        summarized_turns = 0

        def make_context() -> tuple[Any, int]:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            if history.summary:
                earlier_context = (
                    "Earlier conversation excerpts (background only; use only if "
                    "relevant to the current request):\n"
                    + "\n".join(f"- {item}" for item in history.summary)
                )
                messages.extend(
                    [
                        {"role": "user", "content": earlier_context},
                        {
                            "role": "assistant",
                            "content": "I will use those earlier details only when relevant.",
                        },
                    ]
                )
            messages.extend(history.messages)

            model_history = self._ov_genai.ChatHistory()
            model_history.set_messages(messages)
            prompt = self._tokenizer.apply_chat_template(
                model_history, add_generation_prompt=True
            )
            token_count = int(self._tokenizer.encode(prompt).input_ids.shape[1])
            return model_history, token_count

        model_history, context_tokens = make_context()
        while history.messages and (
            context_tokens > input_budget
            or len(history.messages) > MAX_RECENT_TURNS * 2 + 1
        ):
            if (
                len(history.messages) < 3
                or history.messages[0].get("role") != "user"
                or history.messages[1].get("role") != "assistant"
            ):
                break

            earlier_user_message = history.messages.pop(0).get("content", "")
            earlier_assistant_message = history.messages.pop(0).get("content", "")
            user_excerpt = " ".join(earlier_user_message.split())[:120]
            assistant_excerpt = " ".join(earlier_assistant_message.split())[:140]
            excerpt = f"User: {user_excerpt} | Assistant: {assistant_excerpt}"
            if excerpt:
                history.summary.append(excerpt)
                history.summary = history.summary[-MAX_SUMMARY_ITEMS:]
                summarized_turns += 1

            model_history, context_tokens = make_context()

        return model_history, {
            "context_tokens": context_tokens,
            "recent_turns_in_context": (len(history.messages) + 1) // 2,
            "summarized_turns": summarized_turns,
        }

    def status(self) -> dict[str, str | int | float]:
        return {
            "model_path": str(self.model_path),
            "model_name": self.model_path.name,
            "device": self.device,
            "model_size_bytes": self.model_size_bytes,
            "load_time_ms": round(self.load_time_ms, 3),
        }

    @staticmethod
    def _metrics(perf_metrics: Any) -> dict[str, int | float]:
        def mean(method_name: str) -> float:
            return float(getattr(perf_metrics, method_name)().mean)

        return {
            "ttft_ms": mean("get_ttft"),
            "tpot_ms": mean("get_tpot"),
            "throughput_tokens_per_second": mean("get_throughput"),
            "generation_time_ms": mean("get_generate_duration"),
            "inference_time_ms": mean("get_inference_duration"),
            "input_tokens": int(perf_metrics.get_num_input_tokens()),
            "output_tokens": int(perf_metrics.get_num_generated_tokens()),
        }
