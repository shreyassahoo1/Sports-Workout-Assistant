"""
FORMScope — AI-Based Workout Form Correction System
RV College of Engineering | ACY 2025-26

Entry point. Run with:
    python main.py                  # webcam mode (default)
    python main.py --source video.mp4   # video file mode
    python main.py --exercise pushup    # start on a specific exercise
    python main.py --no-tts             # disable voice feedback
"""

import argparse
import sys
from core.session import WorkoutSession


def parse_args():
    parser = argparse.ArgumentParser(
        description="FORMScope — AI Workout Form Correction System"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: '0' for webcam, or path to a video file",
    )
    parser.add_argument(
        "--exercise",
        type=str,
        default="squat",
        choices=["squat", "pushup", "plank", "lunge"],
        help="Starting exercise (default: squat)",
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Disable text-to-speech audio feedback",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        default=True,
        help="Mirror webcam feed horizontally (default: True)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        metavar="OUTPUT.mp4",
        help="Save the annotated output to a video file",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve source: int for webcam index, str for file path
    source = args.source
    if source.isdigit():
        source = int(source)

    print("\n╔══════════════════════════════════════════╗")
    print("║       FORMScope — AI Workout Coach       ║")
    print("╠══════════════════════════════════════════╣")
    print(f"║  Source   : {'Webcam' if isinstance(source, int) else source:<29}║")
    print(f"║  Exercise : {args.exercise:<29}║")
    print(f"║  TTS      : {'Enabled' if not args.no_tts else 'Disabled':<29}║")
    print(f"║  Mirror   : {str(args.mirror):<29}║")
    print(f"║  Save     : {str(args.save or 'No'):<29}║")
    print("╚══════════════════════════════════════════╝")
    print("\nControls:")
    print("  [SPACE]  Pause / Resume")
    print("  [1]      Squat      [2] Push-up")
    print("  [3]      Plank      [4] Lunge")
    print("  [R]      Reset reps / sets")
    print("  [S]      Save screenshot")
    print("  [Q/ESC]  Quit\n")

    session = WorkoutSession(
        source=source,
        exercise=args.exercise,
        tts_enabled=not args.no_tts,
        mirror=args.mirror,
        save_path=args.save,
    )
    session.run()


if __name__ == "__main__":
    main()
