"""Contract tests for the fail-closed recognition API."""
import asyncio
import unittest
from unittest.mock import patch

import app as api


class FakeRuntime:
    def __init__(self, status):
        self._status = status
        self.attempts = []

    def status(self, *, attempt_load=False):
        self.attempts.append(attempt_load)
        return dict(self._status)


class FakeWebSocket:
    def __init__(self):
        self.accepted = False
        self.messages = []
        self.close_args = None

    async def accept(self):
        self.accepted = True

    async def send_json(self, payload):
        self.messages.append(payload)

    async def close(self, code, reason):
        self.close_args = (code, reason)


UNAVAILABLE = {
    "service": "signbridge-sign-stream",
    "status": "model_not_configured",
    "model_loaded": False,
    "checkpoint_language": None,
    "expected_language": "ASL",
    "recognition_enabled": False,
    "translation_enabled": False,
    "message": "No usable language-matched recognition checkpoint is loaded.",
}


class RecognitionAPIContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_health_is_separate_from_model_readiness(self):
        response = await api.healthz()
        self.assertEqual(response, {"ok": True, "service": "signbridge-sign-stream"})

    async def test_status_endpoint_stays_unavailable_without_pipeline(self):
        runtime = FakeRuntime(UNAVAILABLE)
        with patch.object(api, "runtime", runtime):
            response = await api.recognition_status()
        self.assertEqual(response.status_code, 503)
        self.assertFalse(response.body is None)
        self.assertIn(b'"recognition_enabled":false', response.body)
        self.assertIn(b'"translation_enabled":false', response.body)
        self.assertEqual(runtime.attempts, [True])

    async def test_websocket_sends_safe_fallback_then_closes(self):
        runtime = FakeRuntime(UNAVAILABLE)
        socket = FakeWebSocket()
        with patch.object(api, "runtime", runtime):
            await api.recognition_session(socket)
        self.assertTrue(socket.accepted)
        self.assertEqual(len(socket.messages), 1)
        message = socket.messages[0]
        self.assertEqual(message["type"], "recognition_unavailable")
        self.assertEqual(message["safe_fallback"], "Use text input or a qualified human interpreter.")
        self.assertFalse(message["recognition_enabled"])
        self.assertFalse(message["translation_enabled"])
        self.assertEqual(socket.close_args[0], 1013)
        self.assertEqual(runtime.attempts, [True])


if __name__ == "__main__":
    unittest.main()
