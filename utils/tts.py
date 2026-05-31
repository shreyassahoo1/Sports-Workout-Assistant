"""
utils/tts.py — Non-blocking text-to-speech via a dedicated daemon thread.

Uses pyttsx3 so it works offline on Linux, macOS, and Windows.
The audio engine runs on its own thread; the video loop enqueues messages
and immediately returns — no frame drops from TTS latency.

Cooldown logic prevents the same phrase from repeating within COOLDOWN_SEC.
Priority queue ensures critical warnings ("Stop! Injury risk") jump the queue.
"""

import queue
import threading
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

COOLDOWN_SEC = 4.0          # minimum seconds between identical messages
MAX_QUEUE    = 8            # drop oldest if queue overflows (keep it snappy)


class TTSManager:
    """
    Thread-safe text-to-speech manager.

    Usage
    -----
        tts = TTSManager(enabled=True, rate=160, volume=0.9)
        tts.start()
        tts.say("Keep your back straight.", priority=False)
        tts.say("Stop! Possible injury.", priority=True)   # jumps the queue
        tts.stop()
    """

    def __init__(
        self,
        enabled: bool = True,
        rate: int = 155,
        volume: float = 0.92,
    ):
        self.enabled   = enabled
        self.rate      = rate
        self.volume    = volume

        self._queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=MAX_QUEUE)
        self._thread: Optional[threading.Thread] = None
        self._running  = False
        self._last_said: dict[str, float] = {}   # message → timestamp
        self._engine   = None
        self._seq      = 0   # tie-breaker for equal-priority items

    # ── public API ────────────────────────────────────────────────────────────

    def start(self):
        if not self.enabled:
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._worker, daemon=True, name="TTS-Worker"
        )
        self._thread.start()
        logger.debug("TTS worker thread started.")

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            # Unblock the worker if it's waiting on an empty queue
            try:
                self._queue.put_nowait((0, 0, "__STOP__"))
            except queue.Full:
                pass
            self._thread.join(timeout=3)
        logger.debug("TTS worker thread stopped.")

    def say(self, message: str, priority: bool = False):
        """
        Enqueue a message.

        Parameters
        ----------
        message  : str   — text to speak
        priority : bool  — if True, jump ahead of normal messages
        """
        if not self.enabled or not self._running:
            return

        # Cooldown check
        now = time.time()
        last = self._last_said.get(message, 0)
        if now - last < COOLDOWN_SEC:
            return

        prio_val = 0 if priority else 1
        self._seq += 1
        item = (prio_val, self._seq, message)

        try:
            if self._queue.full():
                # Drop the lowest-priority tail to make room
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
            self._queue.put_nowait(item)
        except queue.Full:
            pass   # silently drop if still full

    def flush(self):
        """Clear all pending messages (e.g. when switching exercises)."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    # ── internal worker ───────────────────────────────────────────────────────

    def _worker(self):
        """Runs on the daemon thread. Initialises its own pyttsx3 engine."""
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate",   self.rate)
            self._engine.setProperty("volume", self.volume)

            # Prefer a female voice when available (friendlier coaching tone)
            voices = self._engine.getProperty("voices")
            female = next(
                (v for v in voices if "female" in v.name.lower() or
                 "zira" in v.id.lower() or "victoria" in v.id.lower()),
                None,
            )
            if female:
                self._engine.setProperty("voice", female.id)

        except Exception as e:
            logger.warning(f"pyttsx3 initialisation failed: {e}. TTS disabled.")
            self._running = False
            return

        while self._running:
            try:
                priority, seq, message = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if message == "__STOP__":
                break

            try:
                self._engine.say(message)
                self._engine.runAndWait()
                self._last_said[message] = time.time()
            except Exception as e:
                logger.debug(f"TTS speak error: {e}")

        # Cleanup
        try:
            self._engine.stop()
        except Exception:
            pass
