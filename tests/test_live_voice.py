"""Offline tests for voice capture helpers and push-to-talk gating.

No microphone, recognizer or network is used here.
"""

from __future__ import annotations

import io
import threading
import time
import wave

from live.voice import (
    MAX_TRANSCRIPT_CHARS,
    PushToTalkVoice,
    capture_seconds,
    frames_to_wav,
    key_char,
    normalize_transcript,
)


class _Key:
    def __init__(self, char: str | None) -> None:
        self.char = char


class _OtherKey:
    pass


class TestVoiceHelpers:
    def test_frames_to_wav_has_expected_format(self):
        payload = frames_to_wav([b"\x01\x00" * 800, b"\x02\x00" * 800])
        with wave.open(io.BytesIO(payload), "rb") as handle:
            assert handle.getnchannels() == 1
            assert handle.getsampwidth() == 2
            assert handle.getframerate() == 16000
            assert handle.getnframes() == 1600

    def test_capture_seconds_uses_samples(self):
        chunks = [b"\x00\x00" * 8000, b"\x00\x00" * 8000]
        assert capture_seconds(chunks) == 1.0

    def test_key_char_normalises_and_ignores_special_keys(self):
        assert key_char(_Key("S")) == "s"
        assert key_char(_Key(None)) is None
        assert key_char(_OtherKey()) is None

    def test_normalize_transcript_trims_and_caps(self):
        assert normalize_transcript("  I   want to SIT \n") == "i want to sit"
        long_text = "a" * (MAX_TRANSCRIPT_CHARS + 50)
        assert len(normalize_transcript(long_text)) == MAX_TRANSCRIPT_CHARS

    def test_construct_without_start_does_not_touch_hardware(self):
        voice = PushToTalkVoice(on_transcript=lambda _text: None, autostart=False)
        assert voice.ready is False
        assert voice.error is None
        voice.stop()
        voice.stop()


class _FakeAudioFile:
    def __init__(self, stream) -> None:
        self.stream = stream
        self.entered = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *_exc):
        self.entered = False
        return False


class _FakeSR:
    class UnknownValueError(Exception):
        pass

    class RequestError(Exception):
        pass

    def __init__(self) -> None:
        self.sources = []

    def AudioFile(self, stream):
        source = _FakeAudioFile(stream)
        self.sources.append(source)
        return source


class _FakeRecognizer:
    def __init__(self, text: str) -> None:
        self.text = text
        self.recorded = []

    def record(self, source):
        assert source.entered, "record() called outside a 'with' block"
        source.stream.seek(0)
        self.recorded.append(source.stream.read(4))
        return object()

    def recognize_google(self, _audio, language=None):
        return self.text


class _RaisingRecognizer:
    def record(self, _source):
        raise RuntimeError("boom")


class TestTranscribe:
    def test_source_is_entered_before_recording_and_transcript_is_clean(self):
        events = []
        voice = PushToTalkVoice(on_transcript=events.append, autostart=False)
        fake_sr = _FakeSR()
        recognizer = _FakeRecognizer("  I want to SIT \n")
        voice._sr = fake_sr
        voice._recognizer = recognizer
        voice._running = True

        voice._transcribe([b"\x00\x00" * 800])

        assert recognizer.recorded == [b"RIFF"]
        assert fake_sr.sources[0].entered is False
        assert events == ["i want to sit"]

    def test_transcribe_failure_is_reported_not_raised(self):
        errors = []
        voice = PushToTalkVoice(
            on_transcript=lambda _text: None,
            on_error=errors.append,
            autostart=False,
        )
        voice._sr = _FakeSR()
        voice._recognizer = _RaisingRecognizer()
        voice._running = True

        voice._transcribe([b"\x00\x00" * 800])

        assert errors == ["Voice recognition failed (RuntimeError: boom)"]
        assert voice._processing is False


class TestPushToTalkGating:
    def _voice(self, events):
        listening = []
        voice = PushToTalkVoice(
            on_transcript=events.append,
            on_listening=listening.append,
            autostart=False,
        )
        voice._running = True
        return voice, listening

    def test_hold_and_release_dispatches_one_transcript(self):
        events = []
        voice, listening = self._voice(events)
        done = threading.Event()

        def fake_transcribe(frames):
            events.append(f"frames={len(frames)}")
            done.set()

        voice._transcribe = fake_transcribe
        voice._on_press(_Key("s"))
        assert voice._active is True
        assert listening == [True]
        with voice._lock:
            voice._frames = [b"\x00\x00" * 1600]
            voice._started_ts -= 0.5
        voice._on_release(_Key("s"))
        assert done.wait(1.0)
        assert listening == [True, False]
        assert events == ["frames=1"]
        voice._running = False

    def test_legacy_v_key_also_works(self):
        events = []
        voice, listening = self._voice(events)
        voice._transcribe = lambda frames: events.append("called")
        voice._on_press(_Key("v"))
        assert listening == [True]
        voice._running = False

    def test_unrelated_key_is_ignored(self):
        events = []
        voice, listening = self._voice(events)
        voice._on_press(_Key("q"))
        assert listening == []
        assert voice._active is False

    def test_release_without_matching_press_is_ignored(self):
        events = []
        voice, listening = self._voice(events)
        voice._transcribe = lambda frames: events.append("called")
        voice._on_release(_Key("s"))
        time.sleep(0.05)
        assert listening == []
        assert events == []

    def test_short_tap_is_ignored(self):
        events = []
        voice, listening = self._voice(events)
        voice._transcribe = lambda frames: events.append("called")
        voice._on_press(_Key("s"))
        with voice._lock:
            voice._frames = [b"\x00\x00" * 10]
            voice._started_ts = time.monotonic()
        voice._on_release(_Key("s"))
        time.sleep(0.05)
        assert listening == [True, False]
        assert events == []

    def test_second_press_while_held_does_not_restart(self):
        events = []
        voice, listening = self._voice(events)
        voice._on_press(_Key("s"))
        with voice._lock:
            voice._frames = [b"a"]
            started = voice._started_ts
        voice._on_press(_Key("s"))
        with voice._lock:
            assert voice._frames == [b"a"]
            assert voice._started_ts == started
        assert listening == [True]

    def test_listening_callback_errors_do_not_break_capture(self):
        voice = PushToTalkVoice(
            on_transcript=lambda _text: None,
            on_listening=lambda _active: (_ for _ in ()).throw(RuntimeError("boom")),
            autostart=False,
        )
        voice._running = True
        voice._on_press(_Key("s"))
        assert voice._active is True
