"""Voice commands must degrade gracefully when the LLM is unavailable."""

import agents.orchestrator as orchestrator
from agents.orchestrator import AgentEngine


OFFLINE_REPLY = "I can't reach the assistant right now."


def test_missing_key_is_safe_and_offline_reply_is_spoken(monkeypatch):
    monkeypatch.setattr(orchestrator.Config, "GROQ_API_KEY", "")
    monkeypatch.setattr(orchestrator, "register_search_intent_callback", lambda _cb: None)
    spoken = []
    engine = AgentEngine(tts_callback=spoken.append)

    assert engine.llm is None
    assert engine.extract_memory_label("remember this") is None

    engine.process_voice_command("where is the door", "a door is ahead")
    assert spoken == [OFFLINE_REPLY]


def test_unreachable_llm_speaks_fallback_instead_of_raising(monkeypatch):
    class UnreachableLLM:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            raise ConnectionError("connection refused")

    monkeypatch.setattr(orchestrator.Config, "GROQ_API_KEY", "test-key")
    monkeypatch.setattr(orchestrator, "ChatGroq", lambda **kwargs: UnreachableLLM())
    monkeypatch.setattr(orchestrator, "register_search_intent_callback", lambda _cb: None)
    spoken = []
    engine = AgentEngine(tts_callback=spoken.append)

    assert engine.llm_with_tools is not None
    engine.process_voice_command("where is the door", "a door is ahead")
    assert spoken == [OFFLINE_REPLY]
