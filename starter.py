"""Visiona3D starter - the one public entry point.

Interactive, no setup knowledge needed:
    python starter.py

Direct passthrough (every live option is forwarded):
    python starter.py --source 0 --hfov 70
    python starter.py --source sample-vid/FRONT/VID2.mp4 --intrinsics camera.json

The first menu choice starts a webcam with an estimated field of view.
Distances are then estimates; the HUD shows a red warning bar. Accurate metres
require a one-time camera calibration, available under Advanced options.

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

MENU = """
======================= Visiona3D =======================
 1) Start with my webcam             (recommended)
 2) Start with a video file
 3) Advanced options and calibration
 0) Exit
=========================================================

Guidance is spoken automatically. Hold S to say a goal
(for example "I want to sit"), then release. Press m to
mute speech and q to quit. The first start can take up
to a minute while the models load.

Press 1, 2, 3 or 0 and then Enter. Pressing Enter alone
starts the webcam.
"""

ADVANCED_MENU = """
=================== Advanced options ===================
 1) Webcam with a calibration file (accurate metres)
 2) Video with a calibration file
 3) Show every command-line option
 4) How to make a calibration file
 0) Back
=======================================================
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
3. Use Advanced option 1 and enter reports\\camera_intrinsics.json.

Capture at the same resolution you will run at. A file made
for another resolution is rejected instead of being rescaled.
"""


def _ask(text: str) -> str:
    return input(text).strip().strip('"').strip("'")


def _ask_path(text: str, default: Path | None = None) -> Path | None:
    while True:
        raw = _ask(text)
        if not raw:
            return default
        path = Path(raw).expanduser()
        if path.exists():
            return path
        print(f"  Not found: {path}. Try again, or press Enter to go back.")


def _run(args: list[str]) -> int:
    from scripts.live_demo import main as live_main

    return live_main(args)


def _pause() -> bool:
    try:
        input("\nPress Enter to return to the menu...")
        return True
    except EOFError:
        return False


def _launch(args: list[str], estimated: bool) -> bool:
    print()
    if estimated:
        print("  No camera calibration file is used, so distances are estimates.")
        print("  The red bar at the top of the picture means the same thing.")
    try:
        code = _run(args)
    except KeyboardInterrupt:
        print("\n  Run interrupted.")
        code = 0
    if code:
        print(f"\n  The run ended with code {code}. See the message above.")
        if args[:2] == ["--source", "0"]:
            print("  No webcam? Start again and choose option 2 for a video file.")
    return _pause()


def _start_webcam(calibrated: bool) -> bool:
    if calibrated:
        intrinsics = _ask_path("Calibration file path: ")
        if intrinsics is None:
            print("  No file entered; returning to the menu.")
            return True
        return _launch(["--source", "0", "--intrinsics", str(intrinsics)], False)
    return _launch(["--source", "0", "--hfov", "70"], True)


def _start_video(calibrated: bool) -> bool:
    hint = f"Video file path [{DEFAULT_VIDEO}]" if DEFAULT_VIDEO.exists() else "Video file path"
    print("\n  Tip: drag the video file into this window and press Enter.")
    source = _ask_path(f"  {hint}: ", DEFAULT_VIDEO if DEFAULT_VIDEO.exists() else None)
    if source is None:
        print("  No video file entered; returning to the menu.")
        return True
    args = ["--source", str(source)]
    if calibrated:
        intrinsics = _ask_path("Calibration file path: ")
        if intrinsics is None:
            print("  No file entered; returning to the menu.")
            return True
        args += ["--intrinsics", str(intrinsics)]
        return _launch(args, False)
    return _launch(args + ["--hfov", "70"], True)


def advanced_menu() -> None:
    while True:
        print(ADVANCED_MENU)
        choice = _ask("Choice [0]: ") or "0"
        if choice == "0":
            return
        if choice == "1":
            if not _start_webcam(calibrated=True):
                return
        elif choice == "2":
            if not _start_video(calibrated=True):
                return
        elif choice == "3":
            _run(["--help"])
            if not _pause():
                return
        elif choice == "4":
            print(CALIBRATION_HELP)
            if not _pause():
                return
        else:
            print("  Enter 0-4.")


def menu() -> int:
    while True:
        print(MENU)
        choice = _ask("Choice [1]: ") or "1"
        if choice == "0":
            return 0
        if choice == "3":
            advanced_menu()
            continue
        if choice == "1":
            if not _start_webcam(calibrated=False):
                return 0
        elif choice == "2":
            if not _start_video(calibrated=False):
                return 0
        else:
            print("  Enter 0-3.")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args:
        try:
            return _run(args)
        except KeyboardInterrupt:
            print("\nInterrupted.", file=sys.stderr)
            return 130
    try:
        return menu()
    except (EOFError, KeyboardInterrupt):
        print()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
