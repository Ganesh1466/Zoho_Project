import json
import threading
import unittest
import uuid
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from localmind import ConversationHistory
import web_chat
from web_chat import make_handler


class FakeEngine:
    def __init__(self):
        self.histories = []

    def new_history(self):
        history = ConversationHistory()
        self.histories.append(history)
        return history

    def respond(self, history, message, _max_new_tokens):
        self.last_max_new_tokens = _max_new_tokens
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "Local response"})
        return "Local response", {"throughput_tokens_per_second": 4.5}

    def status(self):
        return {"model_name": "fake-model", "device": "CPU"}


class WebChatSessionTests(unittest.TestCase):
    def setUp(self):
        self.engine = FakeEngine()
        self.server = ThreadingHTTPServer(
            ("127.0.0.1", 0), make_handler(self.engine)
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self, route, payload):
        request = Request(
            self.base_url + route,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))

    def test_sessions_are_isolated_and_reset_clears_only_selected_session(self):
        first_session = str(uuid.uuid4())
        second_session = str(uuid.uuid4())
        self.request(
            "/api/chat", {"session_id": first_session, "message": "Java topic"}
        )
        self.request(
            "/api/chat", {"session_id": first_session, "message": "Java arrays"}
        )
        self.request(
            "/api/chat", {"session_id": second_session, "message": "Crypto topic"}
        )

        first_history, second_history = self.engine.histories
        self.assertEqual(len(first_history.messages), 4)
        self.assertEqual(len(second_history.messages), 2)
        self.assertIn("Java topic", first_history.messages[0]["content"])
        self.assertNotIn(
            "Crypto topic",
            [message["content"] for message in first_history.messages],
        )

        result = self.request("/api/reset", {"session_id": first_session})
        self.assertTrue(result["cleared"])
        self.assertEqual(first_history.messages, [])
        self.assertEqual(len(second_history.messages), 2)

        self.request(
            "/api/chat", {"session_id": first_session, "message": "Fresh topic"}
        )
        fresh_history = self.engine.histories[-1]
        self.assertIsNot(fresh_history, first_history)
        self.assertEqual(len(fresh_history.messages), 2)
        self.assertEqual(
            [message["content"] for message in fresh_history.messages],
            ["Fresh topic", "Local response"],
        )

    def test_reset_frees_a_slot_without_silently_evicting_another_session(self):
        previous_limit = web_chat.MAX_ACTIVE_SESSIONS
        web_chat.MAX_ACTIVE_SESSIONS = 1
        first_session = str(uuid.uuid4())
        second_session = str(uuid.uuid4())
        try:
            self.request(
                "/api/chat", {"session_id": first_session, "message": "First"}
            )
            with self.assertRaises(HTTPError) as raised:
                self.request(
                    "/api/chat", {"session_id": second_session, "message": "Second"}
                )
            self.assertEqual(raised.exception.code, 503)
            raised.exception.close()

            self.request("/api/reset", {"session_id": first_session})
            self.request(
                "/api/chat", {"session_id": second_session, "message": "Second"}
            )
            self.assertEqual(len(self.engine.histories[-1].messages), 2)
        finally:
            web_chat.MAX_ACTIVE_SESSIONS = previous_limit


if __name__ == "__main__":
    unittest.main()
