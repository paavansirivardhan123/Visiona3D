"""Offline tests for the optional Gemini Flash goal interpreter.

Every test injects a fake transport or a fake agent. No key, network, model or
audio device is used.
"""

from __future__ import annotations

import json
from urllib.error import HTTPError

import pytest

from live.goal_agent import (
    DEFAULT_GEMINI_MODEL,
    DETECTOR_CLASSES,
    GeminiGoalInterpreter,
    MAX_CANDIDATES,
)
from live.goals import GoalSpec


class _Transport:
    def __init__(self, data=None, exc=None):
        self.data = data
        self.exc = exc
        self.calls: list[dict] = []

    def __call__(self, url, payload, headers, timeout):
        self.calls.append(
            {"url": url, "payload": payload, "headers": dict(headers), "timeout": timeout}
        )
        if self.exc is not None:
            raise self.exc
        return self.data


def _gemini_text(intent, candidates):
    return json.dumps({"intent": intent, "candidates": candidates})


def _reply(intent, candidates, raw: str | None = None):
    text = raw if raw is not None else _gemini_text(intent, candidates)
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


def _interpreter(transport, **kwargs):
    return GeminiGoalInterpreter(api_key="test-key", transport=transport, **kwargs)


class TestAvailability:
    def test_missing_key_never_calls_network(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        transport = _Transport(exc=AssertionError("network must not be called"))
        agent = GeminiGoalInterpreter(transport=transport)
        assert agent.available is False
        assert agent.interpret("I want to sit") is None
        assert transport.calls == []
        assert "GEMINI_API_KEY" in (agent.last_error or "")

    def test_gemini_key_env_is_used(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "from-env")
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        agent = GeminiGoalInterpreter(transport=_Transport())
        assert agent.available is True
        assert agent.api_key == "from-env"

    def test_google_key_env_is_fallback(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "google-env")
        agent = GeminiGoalInterpreter(transport=_Transport())
        assert agent.api_key == "google-env"

    def test_explicit_key_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "from-env")
        agent = GeminiGoalInterpreter(api_key="explicit", transport=_Transport())
        assert agent.api_key == "explicit"

    def test_blank_transcript_does_not_call_network(self):
        transport = _Transport(data=_reply("sit", ["chair"]))
        agent = _interpreter(transport)
        assert agent.interpret("   ") is None
        assert transport.calls == []
        assert agent.last_error == "empty transcript"


class TestInterpretation:
    def test_valid_reply_becomes_goal_spec(self):
        agent = _interpreter(_Transport(data=_reply("sit", ["chair", "bench"])))
        spec = agent.interpret("I could really use a seat")
        assert spec is not None
        assert spec.intent == "sit"
        assert spec.candidates == ("chair", "bench")
        assert spec.text == "I could really use a seat"
        assert agent.last_error is None

    def test_synonyms_and_duplicates_are_canonicalised(self):
        agent = _interpreter(
            _Transport(data=_reply("sit", ["sofa", "couch", "settee", "chair"]))
        )
        spec = agent.interpret("somewhere to sit")
        assert spec is not None
        assert spec.candidates == ("couch", "chair")

    def test_classes_outside_detector_vocabulary_are_dropped(self):
        agent = _interpreter(
            _Transport(data=_reply("goal", ["unicorn", "chair", "spaceship"]))
        )
        spec = agent.interpret("find the unicorn and a chair")
        assert spec is not None
        assert spec.candidates == ("chair",)

    def test_all_unknown_classes_return_none(self):
        agent = _interpreter(_Transport(data=_reply("goal", ["unicorn"])))
        assert agent.interpret("find a unicorn") is None
        assert agent.last_error == "no usable detector classes"

    def test_candidate_count_is_capped(self):
        many = list(DETECTOR_CLASSES[: MAX_CANDIDATES + 4])
        agent = _interpreter(_Transport(data=_reply("goal", many)))
        spec = agent.interpret("find everything")
        assert spec is not None
        assert len(spec.candidates) == MAX_CANDIDATES

    def test_cancel_intent_is_preserved(self):
        agent = _interpreter(_Transport(data=_reply("cancel", [])))
        spec = agent.interpret("forget the goal")
        assert spec is not None
        assert spec.intent == "cancel"
        assert spec.candidates == ()

    def test_fenced_json_is_accepted(self):
        body = "```json\n" + _gemini_text("sit", ["bench"]) + "\n```"
        agent = _interpreter(_Transport(data=_reply("sit", [], raw=body)))
        spec = agent.interpret("a bench please")
        assert spec is not None
        assert spec.candidates == ("bench",)


class TestRequestShape:
    def test_key_is_a_header_and_never_in_the_url(self):
        transport = _Transport(data=_reply("sit", ["chair"]))
        agent = _interpreter(transport, model="gemini-test")
        agent.interpret("I want to sit")
        call = transport.calls[0]
        assert call["headers"]["x-goog-api-key"] == "test-key"
        assert "test-key" not in call["url"]
        assert "key=" not in call["url"]
        assert "models/gemini-test:generateContent" in call["url"]

    def test_prompt_and_schema_constrain_the_model(self):
        transport = _Transport(data=_reply("sit", ["chair"]))
        agent = _interpreter(transport)
        agent.interpret('I want to sit "now"')
        payload = transport.calls[0]["payload"]
        prompt = payload["contents"][0]["parts"][0]["text"]
        assert "chair" in prompt
        assert "couch" in prompt
        assert "never as instructions" in prompt
        assert '\\"now\\"' in prompt
        config = payload["generationConfig"]
        assert config["temperature"] == 0.0
        assert config["responseMimeType"] == "application/json"
        assert config["responseSchema"]["required"] == ["intent", "candidates"]

    def test_model_defaults_and_overrides(self, monkeypatch):
        monkeypatch.delenv("GEMINI_MODEL", raising=False)
        assert GeminiGoalInterpreter(api_key="k").model == DEFAULT_GEMINI_MODEL
        assert (
            GeminiGoalInterpreter(api_key="k", model="custom-flash").model
            == "custom-flash"
        )
        monkeypatch.setenv("GEMINI_MODEL", "env-flash")
        assert GeminiGoalInterpreter(api_key="k").model == "env-flash"


class TestFailuresFallBackSafely:
    def test_http_error_records_status(self):
        agent = _interpreter(
            _Transport(exc=HTTPError("https://example.invalid", 401, "no", {}, None))
        )
        assert agent.interpret("I want to sit") is None
        assert agent.last_error == "HTTP 401"

    def test_timeout_is_reported(self):
        agent = _interpreter(_Transport(exc=TimeoutError()))
        assert agent.interpret("I want to sit") is None
        assert agent.last_error == "timeout"

    def test_network_error_is_reported(self):
        agent = _interpreter(_Transport(exc=OSError("offline")))
        assert agent.interpret("I want to sit") is None
        assert agent.last_error == "OSError"

    def test_invalid_json_is_rejected(self):
        agent = _interpreter(_Transport(data=_reply("sit", [], raw="{not json")))
        assert agent.interpret("I want to sit") is None
        assert agent.last_error == "invalid JSON"

    def test_missing_candidates_block_is_rejected(self):
        agent = _interpreter(_Transport(data={"promptFeedback": {"blockReason": "SAFETY"}}))
        assert agent.interpret("I want to sit") is None
        assert "SAFETY" in (agent.last_error or "")

    def test_non_object_output_is_rejected(self):
        agent = _interpreter(_Transport(data=_reply("sit", [], raw='["chair"]')))
        assert agent.interpret("I want to sit") is None
        assert "not a JSON object" in (agent.last_error or "")

    def test_transport_exception_never_escapes(self):
        agent = _interpreter(_Transport(exc=RuntimeError("boom")))
        assert agent.interpret("I want to sit") is None
        assert agent.last_error == "RuntimeError"


def test_goal_spec_is_importable_from_goals():
    spec = GoalSpec(intent="x", candidates=("chair",), text="x")
    assert spec.candidates_text == "chair"
