"""Visiona3D starter - the one public entry point.

Interactive, no setup knowledge needed - seven short questions:
    python starter.py

Direct passthrough (no questions, every live option forwarded):
    python starter.py --source 0 --hfov 70
    python starter.py --source sample-vid/FRONT/VID2.mp4 --intrinsics camera.json

Every question shows an example and a default; press Enter to accept it.
Without a calibration file the field of view is estimated, distances are
estimates, and the HUD shows a red warning bar.

While the live window is open, hold S to speak a goal and release to set it.
Voice is optional; microphone or speech-recognition problems never stop
navigation, and ``--goal "I want to sit"`` works without a microphone.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_VIDEO = REPO_ROOT / "sample-vid" / "FRONT" / "VID2.mp4"
DEFAULT_MAP = "reports/session_map.png"
DEFAULT_VIDEO_OUT = "reports/live/demo.mp4"
DEFAULT_HFOV = "75"
QUESTIONS = 7

PROFILES = {"1": "fast", "2": "balanced", "3": "quality"}

HEADER = """
============================ Visiona3D ============================
 Seven short questions: input, frames, speed, speech, map image,
 annotated video and calibration. Press Enter to accept [default].
 After it starts: hold S for a voice goal, m mutes, q quits.
===================================================================
"""

CALIBRATION_HELP = """
A calibration file is a one-time measurement of your camera.
It makes distances metres instead of estimates. You need a
printed checkerboard with 9x6 inner corners.

1. Print the board and measure one square in metres.
2. Run these two commands with the camera connected:

   python scripts\\capture_calibration.py --camera 0 --out calib --board 9x6 --views 15
   python scripts\\calibrate_camera.py --images calib --board 9x6 --square 0.018 --out reports\\camera_intrinsics.json

   Replace 0.018 with your measured square size.
3. Enter reports\\camera_intrinsics.json at the calibration question.

Capture at the same resolution you will run at. A file made
for another resolution is rejected instead of being rescaled.
"""


def _ask(text: str) -> str:
    return input(text).strip().strip('"').strip("'")


def _question(
    index: int | None, question: str, example: str, default: str = ""
) -> str:
    if index is None:
        print(f"\n  {question}")
    else:
        print(f"\nQuestion {index} of {QUESTIONS}: {question}")
    print(f"  example: {example}")
    hint = f" [{default}]" if default else ""
    return _ask(f"  Enter yours{hint}: ")


def _ask_choice(
    index: int, question: str, example: str, options: dict[str, str], default: str
) -> str:
    while True:
        raw = _question(index, question, example, default) or default
        if raw in options:
            return raw
        print(f"  Enter {' or '.join(options)}.")


def _ask_yes_no(index: int, question: str, default: bool) -> bool:
    while True:
        raw = _question(index, question, "yes or no", "yes" if default else "no").lower()
        if not raw:
            return default
        if raw in ("y", "yes", "1"):
            return True
        if raw in ("n", "no", "0"):
            return False
        print("  Enter yes or no (1 and 0 work too).")


def _ask_frames(index: int) -> int | None:
    while True:
        raw = _question(
            index,
            "How many frames should it analyse? (fewer = faster)",
            "10, 60, 100   (Enter = the whole video)",
        )
        if not raw:
            return None
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
        print("  Enter a whole number like 10, 60 or 100, or press Enter for all frames.")


def _ask_input_file(
    index: int | None, question: str, example: str, default: Path | None
) -> Path | None:
    label = str(default) if default is not None else ""
    while True:
        raw = _question(index, question, example, label)
        if not raw:
            return default
        path = Path(raw).expanduser()
        if path.exists():
            return path
        print(f"  Not found: {path}. Try again, or press Enter for the default.")


def _ask_output(
    index: int | None, question: str, example: str, default: str, suffix: str
) -> Path:
    raw = _question(index, question, example, default)
    if not raw:
        return Path(default)
    path = Path(raw).expanduser()
    if raw.endswith(("\\", "/")) or path.is_dir():
        path = path / Path(default).name
    if path.suffix == "":
        path = path.with_suffix(suffix)
    return path


def _ask_calibration(index: int) -> str | None:
    while True:
        raw = _question(
            index,
            "Calibration file for accurate metres (Enter = estimate the field of view)",
            "reports\\camera_intrinsics.json   (type ? for calibration steps)",
        )
        if not raw:
            return None
        if raw == "?":
            print(CALIBRATION_HELP)
            continue
        path = Path(raw).expanduser()
        if path.exists():
            return str(path)
        print(f"  Not found: {path}. Press Enter to estimate instead, or ? for help.")


def questions() -> list[str]:
    """Ask the seven questions and return the equivalent CLI arguments."""
    print(HEADER)
    args: list[str] = []

    mode = _ask_choice(
        1,
        "What should it look at?",
        "1 = my webcam, 2 = a video file",
        {"1": "webcam", "2": "video"},
        "1",
    )
    if mode == "1":
        args += ["--source", "0"]
    else:
        source = _ask_input_file(
            None,
            "Which video file?",
            "C:\\videos\\walk.mp4   (drag the file into this window)",
            DEFAULT_VIDEO if DEFAULT_VIDEO.exists() else None,
        )
        if source is None:
            print("  No video entered; starting the webcam instead.")
            args += ["--source", "0"]
        else:
            args += ["--source", str(source)]

    frames = _ask_frames(2)
    if frames is not None:
        args += ["--max-frames", str(frames)]

    profile = _ask_choice(
        3,
        "Speed or quality?",
        "1 = fast, 2 = balanced, 3 = quality",
        PROFILES,
        "1",
    )
    args += ["--profile", PROFILES[profile]]

    if not _ask_yes_no(4, "Do you need spoken guidance?", True):
        args += ["--no-speech", "--no-voice"]

    if _ask_yes_no(5, "Save the session map image?", True):
        image = _ask_output(
            None,
            "Where should the image go? A file or a folder works.",
            "reports\\session_map.png   (or a folder, e.g. reports\\)",
            DEFAULT_MAP,
            ".png",
        )
        args += ["--map-png", str(image)]

    if _ask_yes_no(
        6, "Also save an annotated video of the HUD? (runs without a window)", False
    ):
        video = _ask_output(
            None,
            "Where should the video go?",
            "reports\\live\\demo.mp4",
            DEFAULT_VIDEO_OUT,
            ".mp4",
        )
        args += ["--out-video", str(video), "--headless"]

    intrinsics = _ask_calibration(7)
    if intrinsics:
        args += ["--intrinsics", intrinsics]
    else:
        print(f"  Using a {DEFAULT_HFOV} deg field of view; distances are estimates.")
        args += ["--hfov", DEFAULT_HFOV]

    return args


def _preview(args: list[str]) -> None:
    print("\n-------------------------------------------------------------------")
    print(" Starting with:")
    index = 0
    while index < len(args):
        key = args[index]
        value = ""
        if index + 1 < len(args) and not args[index + 1].startswith("--"):
            value = args[index + 1]
        if value:
            print(f"   {key:<14} {value}")
            index += 2
        else:
            print(f"   {key}")
            index += 1
    print("-------------------------------------------------------------------")


def _run(args: list[str]) -> int:
    from scripts.live_demo import main as live_main

    return live_main(args)


def menu() -> int:
    args = questions()
    _preview(args)
    try:
        code = _run(args)
    except KeyboardInterrupt:
        print("\n  Run interrupted.")
        return 0
    if code:
        print(f"\n  The run ended with code {code}. See the message above.")
        if "--source" in args and args[args.index("--source") + 1] == "0":
            print("  No webcam? Run again and choose 2 for a video file.")
    return code


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args:
        try:
            return _run(args)
        except KeyboardInterrupt:
            print("\nInterrupted.", file=sys.stderr)
            return 130
    try:
        code = menu()
    except (EOFError, KeyboardInterrupt):
        print()
        return 0
    if sys.stdin.isatty():
        try:
            input("\nPress Enter to close...")
        except (EOFError, KeyboardInterrupt):
            pass
    return code


if __name__ == "__main__":
    raise SystemExit(main())
