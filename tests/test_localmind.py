import threading
import unittest

from localmind import (
    INPUT_CONTEXT_TOKENS,
    SYSTEM_PROMPT,
    LocalMindEngine,
)


class FakeHistory:
    def __init__(self):
        self.messages = []

    def append(self, message):
        self.messages.append(message)

    def pop(self):
        return self.messages.pop()

    def clear(self):
        self.messages.clear()

    def get_messages(self):
        return list(self.messages)

    def set_messages(self, messages):
        self.messages = list(messages)


class FakeTokenizedInputs:
    def __init__(self, token_count):
        self.input_ids = type("Tokens", (), {"shape": [1, token_count]})()


class FakeTokenizer:
    def apply_chat_template(self, history, add_generation_prompt):
        messages = history.get_messages()
        return "\n".join(
            f"{message['role']}: {message['content']}" for message in messages
        ) + ("\nassistant:" if add_generation_prompt else "")

    def encode(self, prompt):
        return FakeTokenizedInputs(max(1, len(prompt) // 4))

    def get_eos_token_id(self):
        return 151645


class FakeGenerationConfig:
    def __init__(self):
        self.max_new_tokens = None
        self.do_sample = None
        self.ignore_eos = None
        self.repetition_penalty = None
        self.eos_token_id = None

    def set_eos_token_id(self, token_id):
        self.eos_token_id = token_id


class FakeResult:
    texts = ["A local response."]

    class Metrics:
        def get_ttft(self):
            return type("Pair", (), {"mean": 12.5})()

        def get_tpot(self):
            return type("Pair", (), {"mean": 3.0})()

        def get_throughput(self):
            return type("Pair", (), {"mean": 10.0})()

        def get_generate_duration(self):
            return type("Pair", (), {"mean": 24.0})()

        def get_inference_duration(self):
            return type("Pair", (), {"mean": 23.0})()

        def get_num_input_tokens(self):
            return 8

        def get_num_generated_tokens(self):
            return 4

    perf_metrics = Metrics()


class FakePipeline:
    def __init__(self, fail=False):
        self.fail = fail
        self.history_messages = None
        self.generation_config = None

    def generate(self, history, generation_config):
        if self.fail:
            raise RuntimeError("inference failed")
        self.history_messages = history.get_messages()
        self.generation_config = generation_config
        return FakeResult()


class FakeGenAI:
    ChatHistory = FakeHistory
    GenerationConfig = FakeGenerationConfig


class LocalMindTests(unittest.TestCase):
    def setUp(self):
        self.engine = LocalMindEngine.__new__(LocalMindEngine)
        self.engine._ov_genai = FakeGenAI
        self.engine._tokenizer = FakeTokenizer()
        self.engine._generation_lock = threading.Lock()
        self.engine.pipeline = FakePipeline()

    def test_response_uses_system_prompt_and_returns_metrics(self):
        history = self.engine.new_history()
        text, metrics = self.engine.respond(history, "hello", 16)

        self.assertEqual(text, "A local response.")
        self.assertEqual(
            history.messages,
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "A local response."},
            ],
        )
        self.assertEqual(self.engine.pipeline.history_messages[0]["role"], "system")
        self.assertEqual(
            self.engine.pipeline.history_messages[0]["content"], SYSTEM_PROMPT
        )
        self.assertEqual(metrics["input_tokens"], 8)
        self.assertEqual(metrics["output_tokens"], 4)
        self.assertGreaterEqual(metrics["context_preparation_ms"], 0)
        self.assertFalse(metrics["output_limit_reached"])
        config = self.engine.pipeline.generation_config
        self.assertEqual(config.eos_token_id, 151645)
        self.assertFalse(config.ignore_eos)
        self.assertFalse(config.do_sample)
        self.assertEqual(config.repetition_penalty, 1.05)

    def test_reaching_output_limit_is_reported(self):
        _, metrics = self.engine.respond(self.engine.new_history(), "hello", 4)
        self.assertTrue(metrics["output_limit_reached"])

    def test_invalid_message_does_not_change_history(self):
        history = self.engine.new_history()
        with self.assertRaises(ValueError):
            self.engine.respond(history, "  ", 16)
        self.assertEqual(history.messages, [])

    def test_failed_generation_removes_unanswered_user_message(self):
        self.engine.pipeline = FakePipeline(fail=True)
        history = self.engine.new_history()
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            self.engine.respond(history, "hello", 16)
        self.assertEqual(history.messages, [])

    def test_separate_histories_do_not_share_context(self):
        java_history = self.engine.new_history()
        crypto_history = self.engine.new_history()
        java_history.append({"role": "user", "content": "I am learning Java."})
        java_history.append({"role": "assistant", "content": "Understood."})
        crypto_history.append(
            {"role": "user", "content": "I am studying cryptography."}
        )
        crypto_history.append({"role": "assistant", "content": "Understood."})

        self.engine.respond(java_history, "What is DSA?", 16)
        java_context = self.engine.pipeline.history_messages
        self.engine.respond(crypto_history, "What is DSA?", 16)
        crypto_context = self.engine.pipeline.history_messages

        self.assertIn("I am learning Java.", [item["content"] for item in java_context])
        self.assertNotIn(
            "I am studying cryptography.",
            [item["content"] for item in java_context],
        )
        self.assertIn(
            "I am studying cryptography.",
            [item["content"] for item in crypto_context],
        )
        self.assertNotIn("I am learning Java.", [item["content"] for item in crypto_context])

    def test_old_turns_are_compacted_to_fit_context(self):
        history = self.engine.new_history()
        for index in range(10):
            history.append(
                {"role": "user", "content": f"Earlier topic {index}. " + "word " * 500}
            )
            history.append(
                {"role": "assistant", "content": "Background answer. " + "word " * 500}
            )

        _, metrics = self.engine.respond(history, "What should I remember?", 32)

        self.assertLessEqual(metrics["context_tokens"], INPUT_CONTEXT_TOKENS)
        self.assertLessEqual(len(history.messages), 2 * 8 + 2)
        self.assertLessEqual(len(history.summary), 8)
        self.assertTrue(any("Earlier topic 0." in item for item in history.summary))
        self.assertTrue(any("Assistant: Background answer." in item for item in history.summary))
        self.assertEqual(history.messages[-2]["content"], "What should I remember?")

    def test_clear_removes_messages_and_compacted_context(self):
        history = self.engine.new_history()
        history.append({"role": "user", "content": "Earlier topic"})
        history.summary.append("Remembered context")

        history.clear()

        self.assertEqual(history.messages, [])
        self.assertEqual(history.summary, [])


if __name__ == "__main__":
    unittest.main()
