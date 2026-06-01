"""
server/app.py — Local web server for FORMScope dashboard.

Endpoints:
  GET /       → dashboard.html
  GET /video  → MJPEG annotated webcam stream
  WS  /ws     → JSON state pushed every frame; commands received from browser

Command flow (browser → Python):
  Browser sends: {"cmd": "pause"} / {"cmd": "reset"} / {"cmd": "end"} / {"cmd": "switch", "exercise": "pushup"}
  app.py puts command string onto a thread-safe queue (cmd_queue)
  session.py drains that queue every frame — no race condition with state pushes
"""

import json
import queue
import threading
import time
import logging
from typing import Optional

from flask import Flask, Response, send_from_directory
from flask_sock import Sock

logger = logging.getLogger(__name__)


class StateStore:
    """Thread-safe store for latest workout state + MJPEG frames + command queue."""

    def __init__(self):
        self._lock         = threading.Lock()
        self._state: dict  = {}
        self._frame_q      = queue.Queue(maxsize=4)
        self._clients      = []
        self._clients_lock = threading.Lock()
        # Commands from browser land here; session.py drains this every frame
        self.cmd_queue     = queue.Queue()

    # ── Producer: WorkoutSession ──────────────────────────────────────────────

    def push_frame(self, jpeg_bytes: bytes):
        try:
            self._frame_q.put_nowait(jpeg_bytes)
        except queue.Full:
            try:
                self._frame_q.get_nowait()
                self._frame_q.put_nowait(jpeg_bytes)
            except queue.Empty:
                pass

    def push_state(self, state: dict):
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

    # ── Consumer: Flask ───────────────────────────────────────────────────────

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


def create_app(store: StateStore, static_folder: str) -> Flask:
    app  = Flask(__name__, static_folder=static_folder)
    sock = Sock(app)

    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    @app.route("/")
    def index():
        return send_from_directory(static_folder, "dashboard.html")

    @app.route("/video")
    def video_feed():
        def generate():
            while True:
                frame = store.get_frame(timeout=0.5)
                if frame is None:
                    continue
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    frame + b"\r\n"
                )
        return Response(generate(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    @sock.route("/ws")
    def websocket(ws):
        store.register_client(ws)
        try:
            while True:
                try:
                    msg = ws.receive(timeout=30)
                    if not msg:
                        continue
                    data = json.loads(msg)
                    cmd  = data.get("cmd", "")
                    # Put command onto the queue — session.py reads it next frame
                    if cmd == "switch":
                        store.cmd_queue.put(f"switch:{data.get('exercise','squat')}")
                    elif cmd in ("pause", "reset", "end"):
                        store.cmd_queue.put(cmd)
                except Exception:
                    break
        finally:
            store.unregister_client(ws)

    return app


class WebServer:
    def __init__(self, store: StateStore, static_folder: str,
                 host: str = "127.0.0.1", port: int = 5000):
        self.store = store
        self.host  = host
        self.port  = port
        self._app  = create_app(store, static_folder)
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._thread = threading.Thread(
            target=self._app.run,
            kwargs=dict(host=self.host, port=self.port,
                        debug=False, use_reloader=False, threaded=True),
            daemon=True,
            name="Flask-Server",
        )
        self._thread.start()
        time.sleep(0.8)
        print(f"\n  🌐  Dashboard → http://{self.host}:{self.port}")
        print(f"  📹  Video feed → http://{self.host}:{self.port}/video\n")
