"""
server/app.py — Local web server for FORMScope dashboard.

Two endpoints:
  GET /              → serves dashboard.html
  GET /video         → MJPEG stream of the annotated webcam feed
  WS  /ws            → WebSocket: pushes JSON state every frame

Architecture:
  WorkoutSession runs on the main thread (OpenCV + MediaPipe).
  Flask runs on a background thread.
  A shared StateStore (thread-safe) is the bridge.

  WorkoutSession  →  StateStore  →  Flask/WS  →  Browser
       ↓                                ↑
  MJPEG queue  ──────────────────────────
"""

import json
import queue
import threading
import time
import logging
import os
from typing import Optional

from flask import Flask, Response, send_from_directory
from flask_sock import Sock

logger = logging.getLogger(__name__)


# ── Shared state between workout loop and web server ──────────────────────────

class StateStore:
    """Thread-safe store for the latest workout frame + state JSON."""

    def __init__(self):
        self._lock       = threading.Lock()
        self._state: dict = {}
        # Bounded queue for MJPEG frames; drop old frames if consumer is slow
        self._frame_q: queue.Queue = queue.Queue(maxsize=4)
        self._clients: list = []      # connected WebSocket clients
        self._clients_lock = threading.Lock()

    # ── Called by WorkoutSession (producer) ──────────────────────────────────

    def push_frame(self, jpeg_bytes: bytes):
        """Put an encoded JPEG frame into the MJPEG queue (non-blocking)."""
        try:
            self._frame_q.put_nowait(jpeg_bytes)
        except queue.Full:
            try:
                self._frame_q.get_nowait()   # drop oldest
                self._frame_q.put_nowait(jpeg_bytes)
            except queue.Empty:
                pass

    def push_state(self, state: dict):
        """Broadcast latest state JSON to all connected WebSocket clients."""
        with self._lock:
            self._state = state
        payload = json.dumps(state)
        dead = []
        with self._clients_lock:
            for ws in self._clients:
                try:
                    ws.send(payload)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self._clients.remove(ws)

    # ── Called by Flask (consumer) ────────────────────────────────────────────

    def get_frame(self, timeout: float = 0.1) -> Optional[bytes]:
        try:
            return self._frame_q.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_state(self) -> dict:
        with self._lock:
            return dict(self._state)

    def register_client(self, ws):
        with self._clients_lock:
            self._clients.append(ws)

    def unregister_client(self, ws):
        with self._clients_lock:
            if ws in self._clients:
                self._clients.remove(ws)


# ── Flask app factory ─────────────────────────────────────────────────────────

def create_app(store: StateStore, static_folder: str) -> Flask:
    app = Flask(__name__, static_folder=static_folder)
    sock = Sock(app)

    # Silence Flask request logs (they'd spam the terminal)
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)

    @app.route("/")
    def index():
        return send_from_directory(static_folder, "dashboard.html")

    @app.route("/video")
    def video_feed():
        """MJPEG stream endpoint."""
        def generate():
            while True:
                frame = store.get_frame(timeout=0.5)
                if frame is None:
                    # Send a keepalive boundary so the browser doesn't time out
                    continue
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    frame +
                    b"\r\n"
                )
        return Response(
            generate(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @sock.route("/ws")
    def websocket(ws):
        """WebSocket: push state JSON each frame, handle exercise switch commands."""
        store.register_client(ws)
        try:
            while True:
                # Block waiting for a message from the browser
                # (exercise switch command: {"cmd":"switch","exercise":"pushup"})
                try:
                    msg = ws.receive(timeout=30)
                    if msg:
                        data = json.loads(msg)
                        if data.get("cmd") == "switch":
                            store.push_state({
                                **store.get_state(),
                                "_cmd": f"switch:{data['exercise']}",
                            })
                except Exception:
                    break
        finally:
            store.unregister_client(ws)

    return app


class WebServer:
    """Starts Flask in a daemon thread so the main loop isn't blocked."""

    def __init__(self, store: StateStore, static_folder: str,
                 host: str = "127.0.0.1", port: int = 5000):
        self.store  = store
        self.host   = host
        self.port   = port
        self._app   = create_app(store, static_folder)
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._thread = threading.Thread(
            target=self._app.run,
            kwargs=dict(host=self.host, port=self.port,
                        debug=False, use_reloader=False,
                        threaded=True),
            daemon=True,
            name="Flask-Server",
        )
        self._thread.start()
        time.sleep(0.8)   # let Flask bind before we print the URL
        print(f"\n  🌐  Dashboard → http://{self.host}:{self.port}")
        print(f"  📹  Video feed → http://{self.host}:{self.port}/video\n")
