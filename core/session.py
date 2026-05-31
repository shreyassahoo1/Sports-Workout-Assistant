"""
core/session.py — WorkoutSession: the main run loop.

Responsibilities:
  - Open the video source (webcam or file)
  - Run MediaPipe Pose on every frame
  - Delegate to the active Exercise for angle computation + form checking
  - Drive the HUD renderer
  - Handle keyboard input (pause, switch exercise, screenshot, quit)
  - Handle optional video save output
  - Drive the TTS manager
  - Plank second-counting (time-based rather than rep-based)
"""

import cv2
import time
import os
import logging
from datetime import datetime
from typing import Union

import mediapipe as mp

from exercises.exercises import EXERCISE_REGISTRY, EXERCISE_KEYS
from utils.tts import TTSManager
from utils.hud import (
    draw_header, draw_feedback_banner, draw_joint_angles, draw_rep_counter,
    draw_form_score, draw_score_breakdown, draw_exercise_selector,
    draw_controls_hint, draw_landmark_connections, draw_landmark_points,
    draw_angle_annotation, draw_no_pose, JointAngleDisplay,
    C_GREEN, C_RED, C_ACC, C_BLUE,
)

logger = logging.getLogger(__name__)

EXERCISES_ORDER = ["squat", "pushup", "plank", "lunge"]
EXERCISE_DISPLAY = ["Squat", "Push-up", "Plank", "Lunge"]

# ── MediaPipe colour palette ───────────────────────────────────────────────────
MP_GOOD_COLOR  = (122, 214,  34)   # BGR green
MP_WARN_COLOR  = ( 60,  64, 240)   # BGR red
MP_JOINT_COLOR = ( 64, 192, 240)   # BGR gold


class WorkoutSession:
    def __init__(
        self,
        source: Union[int, str] = 0,
        exercise: str = "squat",
        tts_enabled: bool = True,
        mirror: bool = True,
        save_path: str = None,
    ):
        self.source     = source
        self.tts_enabled = tts_enabled
        self.mirror     = mirror
        self.save_path  = save_path

        # TTS
        self.tts = TTSManager(enabled=tts_enabled)

        # Exercise
        self._exercise_key = exercise
        self._exercise = EXERCISE_REGISTRY[exercise](tts=self.tts)

        # Session state
        self._paused      = False
        self._running     = True
        self._session_start: float = None
        self._elapsed_str = "00:00"

        # Plank timer
        self._plank_last_tick: float = 0.0

        # Video writer
        self._writer = None

        # MediaPipe
        self._mp_pose  = mp.solutions.pose
        self._mp_draw  = mp.solutions.drawing_utils
        self._mp_style = mp.solutions.drawing_styles
        self._pose_config = dict(
            static_image_mode=False,
            model_complexity=1,            # 0=lite 1=full 2=heavy
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.50,
        )

    # ── Entry point ────────────────────────────────────────────────────────────

    def run(self):
        self.tts.start()

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source!r}")

        fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.save_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(self.save_path, fourcc, fps, (W, H))

        self._session_start = time.time()

        with self._mp_pose.Pose(**self._pose_config) as pose:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    # Video file ended → loop or quit
                    if isinstance(self.source, str):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break

                if self.mirror and isinstance(self.source, int):
                    frame = cv2.flip(frame, 1)

                key = cv2.waitKey(1) & 0xFF
                self._handle_key(key, frame)

                if not self._paused:
                    annotated = self._process_frame(frame, pose)
                else:
                    annotated = self._render_paused(frame)

                cv2.imshow("FORMScope — AI Workout Coach", annotated)

                if self._writer:
                    self._writer.write(annotated)

        cap.release()
        if self._writer:
            self._writer.release()
        cv2.destroyAllWindows()
        self.tts.stop()
        self._print_summary()

    # ── Per-frame processing ───────────────────────────────────────────────────

    def _process_frame(self, frame, pose) -> "np.ndarray":
        import numpy as np

        # Update elapsed time
        elapsed_s = int(time.time() - self._session_start)
        self._elapsed_str = f"{elapsed_s//60:02d}:{elapsed_s%60:02d}"

        # MediaPipe inference (RGB)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = pose.process(rgb)
        rgb.flags.writeable = True

        out = frame.copy()

        if not results.pose_landmarks:
            draw_no_pose(out)
            self._draw_hud(out, None)
            return out

        lm = results.pose_landmarks.landmark

        # ── Exercise logic ──────────────────────────────────────────────────
        ex = self._exercise

        # Plank: tick seconds when a plank second passes
        if ex.IS_TIMED:
            now = time.time()
            if now - self._plank_last_tick >= 1.0:
                ex.state.reps += 1
                ex.state.total_reps += 1
                self._plank_last_tick = now
                if ex.state.reps >= ex.GOAL_REPS:
                    ex.state.sets += 1
                    ex.state.reps  = 0
                    if self.tts:
                        self.tts.say(
                            f"Set {ex.state.sets} complete! Rest for 60 seconds.",
                            priority=True,
                        )

        state = ex.process_frame(lm)

        # ── Draw skeleton ───────────────────────────────────────────────────
        skel_color = MP_GOOD_COLOR if state.is_good_form else MP_WARN_COLOR
        draw_landmark_connections(out, lm, ex.PRIMARY_CONNECTIONS, skel_color, 3)
        draw_landmark_points(out, lm, ex.PRIMARY_JOINTS, MP_JOINT_COLOR, 5)

        # Annotate key joint angles on the skeleton
        self._annotate_angles_on_pose(out, lm, state.angles)

        self._draw_hud(out, state)
        return out

    def _annotate_angles_on_pose(self, frame, lm, angles):
        """Render angle values floating near the key joints."""
        ex  = self._exercise
        ann = []

        # Map exercise → annotated joint index + angle index
        if self._exercise_key == "squat":
            ann = [
                (mp.solutions.pose.PoseLandmark.LEFT_KNEE.value,  0),
                (mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value, 1),
                (mp.solutions.pose.PoseLandmark.LEFT_HIP.value,   2),
            ]
        elif self._exercise_key == "pushup":
            ann = [
                (mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value,  0),
                (mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value, 1),
            ]
        elif self._exercise_key == "plank":
            ann = [
                (mp.solutions.pose.PoseLandmark.LEFT_HIP.value, 1),
            ]
        elif self._exercise_key == "lunge":
            ann = [
                (mp.solutions.pose.PoseLandmark.LEFT_KNEE.value,  0),
                (mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value, 1),
            ]

        for joint_idx, angle_idx in ann:
            if angle_idx < len(angles):
                val = angles[angle_idx]
                lo, hi = ex.ANGLE_RANGES[angle_idx]
                draw_angle_annotation(frame, lm, joint_idx, val, lo <= val <= hi)

    def _draw_hud(self, frame, state):
        ex = self._exercise

        if state is None:
            # Minimal HUD when no pose
            draw_header(frame, ex.NAME, 0, 0, 0.0, self._elapsed_str)
            draw_exercise_selector(frame, EXERCISE_DISPLAY, self._exercise_key)
            draw_controls_hint(frame)
            return

        # Header
        draw_header(
            frame,
            ex.NAME,
            state.total_reps,
            state.sets,
            state.form_score,
            self._elapsed_str,
        )

        # Feedback banner
        draw_feedback_banner(
            frame,
            state.feedback_msg,
            state.is_good_form,
        )

        # Joint angle gauges (left panel)
        angle_displays = [
            JointAngleDisplay(
                name=ex.JOINTS[i],
                value=state.angles[i] if i < len(state.angles) else 0,
                min_val=ex.ANGLE_RANGES[i][0],
                max_val=ex.ANGLE_RANGES[i][1],
            )
            for i in range(len(ex.JOINTS))
        ]
        draw_joint_angles(frame, angle_displays)

        # Exercise selector (left panel, above angles)
        draw_exercise_selector(frame, EXERCISE_DISPLAY, self._exercise_key)

        # Rep ring (bottom-right)
        draw_rep_counter(frame, state.reps, ex.GOAL_REPS)

        # Form score + breakdown (top-right)
        draw_form_score(frame, state.form_score)
        draw_score_breakdown(frame, state.scores)

        # Hint bar
        draw_controls_hint(frame)

    # ── Key input ──────────────────────────────────────────────────────────────

    def _handle_key(self, key, frame):
        if key == ord("q") or key == 27:   # Q or ESC
            self._running = False

        elif key == ord(" "):              # SPACE → pause/resume
            self._paused = not self._paused
            if self._paused:
                self.tts.say("Session paused.")
            else:
                self.tts.say("Resuming.")

        elif key == ord("r"):              # R → reset reps
            self._exercise.reset_reps()
            if self._exercise.IS_TIMED:
                self._plank_last_tick = time.time()
            self.tts.say("Reps reset.")

        elif key == ord("s"):              # S → screenshot
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"formscope_{ts}.jpg"
            cv2.imwrite(fname, frame)
            print(f"  Screenshot saved: {fname}")

        elif key in [ord(str(d)) for d in range(1, 5)]:
            digit = chr(key)
            new_ex = EXERCISE_KEYS.get(digit)
            if new_ex and new_ex != self._exercise_key:
                self._switch_exercise(new_ex)

    def _switch_exercise(self, new_key: str):
        prev_state = self._exercise.state
        self._exercise_key = new_key
        self._exercise = EXERCISE_REGISTRY[new_key](tts=self.tts)
        # Carry sets forward so the HUD doesn't reset mid-session
        self.tts.flush()
        self.tts.say(f"Switching to {EXERCISE_REGISTRY[new_key].NAME}.", priority=True)
        if new_key == "plank":
            self._plank_last_tick = time.time()
        print(f"  ► Switched to: {EXERCISE_REGISTRY[new_key].NAME}")

    # ── Paused frame ───────────────────────────────────────────────────────────

    def _render_paused(self, frame):
        import numpy as np
        out = frame.copy()
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (out.shape[1], out.shape[0]), (18, 22, 28), -1)
        cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

        H, W = out.shape[:2]
        msg = "PAUSED"
        (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 2.0, 3)
        cv2.putText(out, msg, ((W-tw)//2, H//2), cv2.FONT_HERSHEY_DUPLEX,
                    2.0, (64, 192, 240), 3, cv2.LINE_AA)
        sub = "Press SPACE to resume"
        (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
        cv2.putText(out, sub, ((W-sw)//2, H//2 + 50), cv2.FONT_HERSHEY_DUPLEX,
                    0.6, (120, 130, 140), 1, cv2.LINE_AA)
        return out

    # ── Session summary ────────────────────────────────────────────────────────

    def _print_summary(self):
        st = self._exercise.state
        elapsed = int(time.time() - self._session_start) if self._session_start else 0
        print("\n╔══════════════════════════════════════╗")
        print("║         SESSION SUMMARY              ║")
        print("╠══════════════════════════════════════╣")
        print(f"║  Exercise    : {self._exercise.NAME:<20}║")
        print(f"║  Total Reps  : {st.total_reps:<20}║")
        print(f"║  Sets Done   : {st.sets:<20}║")
        print(f"║  Avg Score   : {st.form_score:<19.1f}%║")
        print(f"║  Duration    : {elapsed//60}m {elapsed%60}s{'':<16}║")
        print("╚══════════════════════════════════════╝\n")
