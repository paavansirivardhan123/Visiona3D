"""Hold-to-talk voice input for live goals.

Hold ``S`` (or the legacy ``V``) anywhere while the live window is open, speak,
and release. The microphone is pre-warmed so recording starts immediately; the
buffer is sent to Google Speech-to-Text on release. Recognition needs an
internet connection. No LLM is used: the transcript is interpreted locally by
:mod:`live.goals`.

The optional runtime dependencies (PyAudio, SpeechRecognition, pynput) are
imported lazily. If they or a microphone are unavailable, the live system keeps
running and reports voice as unavailable instead of failing.

All numeric defaults are TO TUNE ON REAL DATA.
"""

from __future__ import annotations

import io
import threading
import time
import wave
from typing import Callable, Iterable

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_SAMPLE_WIDTH = 2  # paInt16
DEFAULT_CHUNK = 1024
DEFAULT_LANGUAGE = "en-US"
DEFAULT_KEYS = ("s", "v")
MIN_CAPTURE_S = 0.25  # TO TUNE ON REAL DATA; ignores accidental taps
MAX_TRANSCRIPT_CHARS = 200  # TO TUNE ON REAL DATA

OnText = Callable[[str], None]
OnFlag = Callable[[bool], None]
OnError = Callable[[str], None]


def frames_to_wav(
    frames: Iterable[bytes],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    sample_width: int = DEFAULT_SAMPLE_WIDTH,
) -> bytes:
    """Pack raw mono PCM chunks into a WAV byte string."""
    payload = b"".join(bytes(chunk) for chunk in frames)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(int(sample_width))
        handle.setframerate(int(sample_rate))
        handle.writeframes(payload)
    return buffer.getvalue()


def capture_seconds(
    frames: Iterable[bytes],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    sample_width: int = DEFAULT_SAMPLE_WIDTH,
) -> float:
    """Duration of a captured PCM buffer in seconds."""
    total = sum(len(chunk) for chunk in frames)
    bytes_per_second = max(1, int(sample_rate) * int(sample_width))
    return total / float(bytes_per_second)


def key_char(key: object) -> str | None:
    """Return a lowercase character for a pynput key event, if any."""
    value = getattr(key, "char", None)
    if isinstance(value, str) and value:
        return value.lower()
    return None


def normalize_transcript(text: str) -> str:
    """Trim and cap a recognizer transcript before intent parsing."""
    value = " ".join(str(text or "").split()).strip().lower()
    return value[:MAX_TRANSCRIPT_CHARS]


class PushToTalkVoice:
    """Global hold-to-talk listener with a pre-warmed microphone."""

    def __init__(
        self,
        on_transcript: OnText,
        on_listening: OnFlag | None = None,
        on_error: OnError | None = None,
        keys: tuple[str, ...] = DEFAULT_KEYS,
        language: str = DEFAULT_LANGUAGE,
        autostart: bool = False,
    ) -> None:
        self.on_transcript = on_transcript
        self.on_listening = on_listening or (lambda _active: None)
        self.on_error = on_error or (lambda _message: None)
        self.keys = tuple(str(key).lower() for key in keys)
        self.language = str(language)
        self.ready = False
        self.error: str | None = None

        self._lock = threading.Lock()
        self._running = False
        self._active = False
        self._processing = False
        self._frames: list[bytes] = []
        self._started_ts = 0.0
        self._held_key: str | None = None
        self._stream = None
        self._pa = None
        self._listener = None
        self._capture_thread: threading.Thread | None = None
        self._recognizer = None
        self._sr = None
        self._pyaudio = None
        self._keyboard = None
        if autostart:
            self.start()

    def start(self) -> bool:
        """Open the microphone and global key listener.

        Returns True when voice is usable. Never raises for missing hardware,
        missing optional packages, or a busy microphone.
        """
        if self.ready:
            return True
        if self._running:
            return False
        try:
            import pyaudio
            import pynput.keyboard as keyboard
            import speech_recognition as sr
        except ImportError as exc:
            self._fail(
                "Voice needs pyaudio, SpeechRecognition and pynput. "
                "Install them with: uv add pyaudio SpeechRecognition pynput "
                f"({type(exc).__name__})"
            )
            return False

        try:
            audio = pyaudio.PyAudio()
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=DEFAULT_SAMPLE_RATE,
                input=True,
                frames_per_buffer=DEFAULT_CHUNK,
                start=True,
            )
            listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
            )
            listener.start()
        except Exception as exc:  # noqa: BLE001 - hardware boundary
            try:
                if "stream" in locals():
                    stream.stop_stream()
                    stream.close()
                if "audio" in locals():
                    audio.terminate()
            except Exception:  # noqa: BLE001
                pass
            self._fail(
                "Microphone could not be opened. Check that it is connected "
                f"and not in use by another app ({type(exc).__name__}: {exc})"
            )
            return False

        self._pyaudio = pyaudio
        self._keyboard = keyboard
        self._sr = sr
        self._recognizer = sr.Recognizer()
        self._pa = audio
        self._stream = stream
        self._listener = listener
        self._running = True
        self._capture_thread = threading.Thread(
            target=self._capture_loop, name="voice-capture", daemon=True
        )
        self._capture_thread.start()
        self.ready = True
        self.error = None
        print(
            f"  [Voice] Ready. Hold {self.keys[0].upper()} to speak a goal, "
            "release to transcribe."
        )
        return True

    def stop(self) -> None:
        """Release the key listener and microphone. Safe to call twice."""
        self._running = False
        self._active = False
        self._held_key = None
        listener = self._listener
        self._listener = None
        if listener is not None:
            try:
                listener.stop()
            except Exception:  # noqa: BLE001
                pass
        thread = self._capture_thread
        self._capture_thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:  # noqa: BLE001
                pass
        audio = self._pa
        self._pa = None
        if audio is not None:
            try:
                audio.terminate()
            except Exception:  # noqa: BLE001
                pass
        self.ready = False

    def _fail(self, message: str) -> None:
        self.error = message
        print(f"  [Voice] {message}")
        try:
            self.on_error(message)
        except Exception:  # noqa: BLE001 - callbacks must not kill voice
            pass

    def _on_press(self, key: object) -> None:
        char = key_char(key)
        if (
            not self._running
            or char is None
            or char not in self.keys
            or self._active
            or self._processing
        ):
            return
        with self._lock:
            self._active = True
            self._frames = []
            self._started_ts = time.monotonic()
            self._held_key = char
        self._notify_listening(True)
        print("  [Voice] Listening... release to transcribe.")

    def _on_release(self, key: object) -> None:
        char = key_char(key)
        if not self._running or self._held_key is None or char != self._held_key:
            return
        with self._lock:
            self._active = False
            captured = list(self._frames)
            self._frames = []
            self._held_key = None
            duration = time.monotonic() - self._started_ts
        self._notify_listening(False)
        if duration < MIN_CAPTURE_S or not captured:
            print("  [Voice] Ignored: recording was too short.")
            return
        self._processing = True
        threading.Thread(
            target=self._transcribe,
            args=(captured,),
            name="voice-transcribe",
            daemon=True,
        ).start()

    def _capture_loop(self) -> None:
        stream = self._stream
        chunk = DEFAULT_CHUNK
        while self._running and stream is not None:
            try:
                data = stream.read(chunk, exception_on_overflow=False)
            except Exception as exc:  # noqa: BLE001 - device boundary
                if self._running:
                    self._fail(
                        f"Microphone read failed ({type(exc).__name__}: {exc})"
                    )
                break
            with self._lock:
                if self._active:
                    self._frames.append(data)

    def _transcribe(self, frames: list[bytes]) -> None:
        try:
            wav_bytes = frames_to_wav(frames)
            with self._sr.AudioFile(io.BytesIO(wav_bytes)) as source:
                audio = self._recognizer.record(source)
            text = normalize_transcript(
                self._recognizer.recognize_google(audio, language=self.language)
            )
            if not text:
                print("  [Voice] Heard nothing usable.")
                return
            print(f'  [Voice] Heard: "{text}"')
            if self._running:
                self.on_transcript(text)
        except self._sr.UnknownValueError:
            self._fail("Could not understand that. Please say the goal again.")
        except self._sr.RequestError:
            self._fail("Speech recognition needs an internet connection.")
        except Exception as exc:  # noqa: BLE001 - recognition boundary
            self._fail(f"Voice recognition failed ({type(exc).__name__}: {exc})")
        finally:
            self._processing = False

    def _notify_listening(self, active: bool) -> None:
        try:
            self.on_listening(bool(active))
        except Exception:  # noqa: BLE001
            pass


__all__ = [
    "DEFAULT_CHUNK",
    "DEFAULT_KEYS",
    "DEFAULT_LANGUAGE",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_SAMPLE_WIDTH",
    "MAX_TRANSCRIPT_CHARS",
    "MIN_CAPTURE_S",
    "PushToTalkVoice",
    "capture_seconds",
    "frames_to_wav",
    "key_char",
    "normalize_transcript",
]
