"""Visiona3D live RGB system: camera/video -> HUD -> map -> DB -> speech.

Usually launched through ``starter.py``; this module holds the full CLI.

Examples:
    python starter.py --source 0 --intrinsics reports/camera_intrinsics.json
    python starter.py --source sample-vid/FRONT/VID2.mp4 --hfov 70 \
        --max-frames 10 --headless --out-video reports/live/demo.mp4 --no-speech
    python starter.py --source 0 --intrinsics camera.json --goal "I want to sit"
    python starter.py --source 0 --intrinsics camera.json --goal-class chair

With a window open, hold S to speak a goal and release to set it. If
``GEMINI_API_KEY`` is set, voice goals are understood by Gemini Flash and the
local deterministic parser is the fallback; ``--no-voice-agent`` forces local.
``--goal`` also works headless, so goals can be tested on a video without a
microphone. The local parser handles "I want to sit" without any API key.

Calibrated intrinsics are required for a real camera. ``--hfov`` is an
explicit, labelled demo fallback and makes every distance an estimate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from live.db import DEFAULT_DB_PATH
from live.runner import LiveConfig, LiveRunner, PROFILES


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visiona3D live RGB HUD, 3D object grid, planning and speech"
    )
    parser.add_argument("--source", default="0", help="camera index or video file")
    parser.add_argument("--intrinsics", default=None, help="calibrated camera JSON")
    parser.add_argument(
        "--hfov",
        type=float,
        default=None,
        help="DEMO ONLY: assume this horizontal FOV in degrees instead of calibrating",
    )
    parser.add_argument("--mount-height", type=float, default=1.4, help="camera height in metres (TO TUNE)")
    parser.add_argument("--pitch", type=float, default=15.0, help="camera downward pitch in degrees (TO TUNE)")
    parser.add_argument("--proc-width", type=int, default=640, help="processing width cap")
    parser.add_argument("--profile", choices=PROFILES, default="fast", help="model input-size profile")
    parser.add_argument("--weights-dir", default="weights", help="local model weights folder")
    parser.add_argument("--depth-variant", choices=("indoor", "outdoor"), default="indoor")
    parser.add_argument("--floor-threshold", type=float, default=0.90, help="floor certification threshold (TO TUNE)")
    parser.add_argument("--fusion-config", default=None, help="optional tuned MobileSAM fusion JSON")
    parser.add_argument("--conformal-depth", default=None, help="optional fitted depth interval JSON")
    goal_group = parser.add_mutually_exclusive_group()
    goal_group.add_argument(
        "--goal",
        default=None,
        help='natural-language goal, e.g. "I want to sit" (expands to chair/bench/couch/bed)',
    )
    goal_group.add_argument(
        "--goal-class",
        default=None,
        help="route to the nearest detected object of this class",
    )
    parser.add_argument(
        "--arrival-distance",
        type=float,
        default=1.5,
        help="distance in metres that counts as reaching the goal (TO TUNE)",
    )
    parser.add_argument(
        "--arrival-frames",
        type=int,
        default=3,
        help="consecutive frames within arrival distance before speaking (TO TUNE)",
    )
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite database path")
    parser.add_argument("--no-db", action="store_true", help="disable persistence")
    parser.add_argument("--no-speech", dest="speech", action="store_false", help="start with speech muted")
    parser.add_argument("--no-speak-objects", dest="speak_objects", action="store_false", help="only speak navigation")
    parser.add_argument("--no-voice", dest="voice", action="store_false", help="do not open the microphone or key listener")
    parser.add_argument(
        "--goal-model",
        default=None,
        help="Gemini Flash model for voice goals (default gemini-2.5-flash; needs GEMINI_API_KEY)",
    )
    parser.add_argument(
        "--no-voice-agent",
        dest="voice_agent",
        action="store_false",
        help="use only the local deterministic goal parser",
    )
    parser.add_argument("--headless", action="store_true", help="no window; use with --out-video")
    parser.add_argument("--out-video", default=None, help="write the annotated HUD to this MP4")
    parser.add_argument(
        "--map-png",
        default=None,
        help="write a top-down session map image with an ESTIMATED travelled trail (display only)",
    )
    parser.add_argument("--video-fps", type=float, default=4.0, help="output video frame rate")
    parser.add_argument("--max-frames", type=int, default=None, help="stop after this many processed frames")
    parser.add_argument("--start-frame", type=int, default=0, help="skip to this video frame")
    parser.add_argument("--loop", action="store_true", help="loop the video file")
    parser.add_argument("--no-realtime", dest="realtime", action="store_false", help="do not pace video files")
    parser.add_argument("--camera-width", type=int, default=None, help="requested camera width")
    parser.add_argument("--camera-height", type=int, default=None, help="requested camera height")
    parser.add_argument("--summary-json", default=None, help="write the run summary JSON here")
    parser.set_defaults(
        speech=True,
        speak_objects=True,
        realtime=True,
        voice=True,
        voice_agent=True,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.intrinsics and args.hfov is None:
        print(
            "ERROR: live mode needs calibrated intrinsics. Pass --intrinsics "
            "<camera.json>. For an uncalibrated demo only, pass --hfov 75.",
            file=sys.stderr,
        )
        return 2
    if args.intrinsics and args.hfov is not None:
        print(
            "ERROR: pass either --intrinsics or --hfov, not both.",
            file=sys.stderr,
        )
        return 2
    if args.out_video and not args.headless:
        print(
            "ERROR: --out-video requires --headless (the HUD writing loop is "
            "the headless loop).",
            file=sys.stderr,
        )
        return 2

    config = LiveConfig(
        source=args.source,
        intrinsics_json=args.intrinsics,
        hfov_deg=args.hfov,
        mount_height_m=args.mount_height,
        pitch_deg=args.pitch,
        proc_width=args.proc_width,
        profile=args.profile,
        weights_dir=args.weights_dir,
        depth_variant=args.depth_variant,
        floor_threshold=args.floor_threshold,
        fusion_config_json=args.fusion_config,
        conformal_depth_json=args.conformal_depth,
        goal_text=args.goal,
        goal_class=args.goal_class,
        goal_model=args.goal_model,
        voice_agent=args.voice_agent,
        arrival_distance_m=args.arrival_distance,
        arrival_stable_frames=args.arrival_frames,
        db_path=args.db,
        enable_db=not args.no_db,
        speech=args.speech,
        speak_objects=args.speak_objects,
        voice=args.voice,
        headless=args.headless,
        out_video=args.out_video,
        map_png=args.map_png,
        video_fps=args.video_fps,
        max_frames=args.max_frames,
        start_frame=args.start_frame,
        loop=args.loop,
        realtime=args.realtime,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
    )

    try:
        runner = LiveRunner(config)
    except (ValueError, RuntimeError, FileNotFoundError, OSError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        summary = runner.run()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"ERROR: live run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("\nLive run summary")
    print(json.dumps(summary, indent=2))
    if args.summary_json:
        path = Path(args.summary_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved summary to {path}")

    if summary.get("session_id"):
        print(
            "Query the stored 3D object grid with:\n"
            "  sqlite3 "
            f"{summary['database']} \"SELECT track_id, class_name, distance_m, "
            "anchor_x, anchor_y, anchor_z FROM object_registry "
            f"WHERE session_id='{summary['session_id']}' ORDER BY distance_m;\""
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
