"""
FORMScope — main.py (web dashboard edition)

Run:
    python main.py                        # webcam, opens browser automatically
    python main.py --source video.mp4     # video file
    python main.py --exercise pushup      # start on push-ups
    python main.py --no-tts               # disable voice
    python main.py --debug                # also show local cv2 window
    python main.py --port 5001            # use a different port
"""

import argparse
import os
import sys
import time
import webbrowser
import threading

def parse_args():
    p = argparse.ArgumentParser(description="FORMScope — AI Workout Form Correction")
    p.add_argument("--source",   default="0",
                   help="Video source: '0' for webcam, or path to video file")
    p.add_argument("--exercise", default="squat",
                   choices=["squat","pushup","plank","lunge"])
    p.add_argument("--no-tts",   action="store_true", help="Disable TTS audio")
    p.add_argument("--mirror",   action="store_true", default=True)
    p.add_argument("--save",     default=None, metavar="OUT.mp4")
    p.add_argument("--port",     type=int, default=5000)
    p.add_argument("--host",     default="127.0.0.1")
    p.add_argument("--debug",    action="store_true",
                   help="Also show a local OpenCV window")
    return p.parse_args()


def main():
    args   = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    # Locate the dashboard HTML (sits next to main.py)
    static_dir = os.path.dirname(os.path.abspath(__file__))

    from server.app import StateStore, WebServer
    from core.session import WorkoutSession

    store  = StateStore()
    server = WebServer(store, static_folder=static_dir,
                       host=args.host, port=args.port)
    server.start()

    # Open browser after a short delay
    url = f"http://{args.host}:{args.port}"
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    print(f"\n  Press  Ctrl+C  to stop.\n")

    session = WorkoutSession(
        source      = source,
        exercise    = args.exercise,
        tts_enabled = not args.no_tts,
        mirror      = args.mirror,
        save_path   = args.save,
        store       = store,
        show_window = args.debug,
    )

    try:
        session.run()
    except KeyboardInterrupt:
        print("\n  Stopping…")


if __name__ == "__main__":
    main()
