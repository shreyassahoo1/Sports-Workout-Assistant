"""
core/session.py — WorkoutSession (web-server edition).

Changes from the standalone version:
  - No cv2.imshow() — frames go to StateStore as JPEG bytes (MJPEG)
  - State dict is pushed to StateStore every frame -> WebSocket -> browser
  - Exercise-switch commands can arrive from the browser via WS
  - Optional local cv2 debug window (--debug flag)
"""

import cv2
import time
import logging
from datetime import datetime
from typing import Union, Optional

import mediapipe as mp

from exercises.exercises import EXERCISE_REGISTRY, EXERCISE_KEYS, EXERCISE_DISPLAY
from utils.tts import TTSManager
from utils.hud import (
    draw_header, draw_feedback_banner, draw_joint_angles,
    draw_rep_counter, draw_form_score, draw_score_breakdown,
    draw_exercise_selector, draw_controls_hint,
    draw_landmark_connections, draw_landmark_points,
    draw_angle_annotation, draw_no_pose, JointAngleDisplay,
)

logger = logging.getLogger(__name__)


class WorkoutSession:
    def __init__(
        self,
        source: Union[int, str] = 0,
        exercise: str = "squat",
        tts_enabled: bool = True,
        mirror: bool = True,
        save_path: str = None,
        store=None,
        show_window: bool = False,
    ):
        self.source      = source
        self.tts_enabled = tts_enabled
        self.mirror      = mirror
        self.save_path   = save_path
        self.store       = store
        self.show_window = show_window

        self.tts = TTSManager(enabled=tts_enabled)

        self._exercise_key = exercise
        self._exercise     = EXERCISE_REGISTRY[exercise](tts=self.tts)

        self._paused        = False
        self._running       = True
        self._session_start: float = None
        self._elapsed_str   = "00:00"
        self._plank_last_tick: float = 0.0
        self._writer        = None
        self._jpeg_quality  = 80

        self._mp_pose  = mp.solutions.pose
        self._pose_config = dict(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.50,
        )

    def run(self):
        self.tts.start()

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source!r}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.save_path:
            fourcc       = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(self.save_path, fourcc, fps, (W, H))

        self._session_start   = time.time()
        self._plank_last_tick = time.time()

        if self.show_window:
            cv2.namedWindow("FORMScope — debug", cv2.WINDOW_NORMAL)

        with self._mp_pose.Pose(**self._pose_config) as pose:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    if isinstance(self.source, str):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break

                if self.mirror and isinstance(self.source, int):
                    frame = cv2.flip(frame, 1)

                # Poll for browser WS commands
                if self.store:
                    st  = self.store.get_state()
                    cmd = st.get("_cmd", "")
                    if cmd.startswith("switch:"):
                        new_ex = cmd.split(":")[1]
                        if new_ex in EXERCISE_REGISTRY and new_ex != self._exercise_key:
                            self._switch_exercise(new_ex)
                        self.store.push_state({**st, "_cmd": ""})
                    if st.get("_pause"):
                        self._paused = not self._paused
                        self.store.push_state({**st, "_pause": False})
                    if st.get("_reset"):
                        self._exercise.reset_reps()
                        if self._exercise.IS_TIMED:
                            self._plank_last_tick = time.time()
                        self.store.push_state({**st, "_reset": False})

                if self.show_window:
                    key = cv2.waitKey(1) & 0xFF
                    self._handle_key(key, frame)
                    if not self._running:
                        break
                else:
                    cv2.waitKey(1)

                annotated = (
                    self._process_frame(frame, pose)
                    if not self._paused
                    else self._render_paused(frame)
                )

                if self.store:
                    ok, buf = cv2.imencode(
                        ".jpg", annotated,
                        [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality],
                    )
                    if ok:
                        self.store.push_frame(buf.tobytes())

                if self.show_window:
                    cv2.imshow("FORMScope — debug", annotated)

                if self._writer:
                    self._writer.write(annotated)

        cap.release()
        if self._writer:
            self._writer.release()
        if self.show_window:
            cv2.destroyAllWindows()
        self.tts.stop()
        self._print_summary()

    # ── Per-frame ─────────────────────────────────────────────────────────────

    def _process_frame(self, frame, pose):
        elapsed_s         = int(time.time() - self._session_start)
        self._elapsed_str = f"{elapsed_s//60:02d}:{elapsed_s%60:02d}"

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = pose.process(rgb)
        rgb.flags.writeable = True

        out = frame.copy()

        if not results.pose_landmarks:
            draw_no_pose(out)
            self._draw_hud(out, None)
            self._push_state(None)
            return out

        lm = results.pose_landmarks.landmark
        ex = self._exercise

        if ex.IS_TIMED:
            now = time.time()
            if now - self._plank_last_tick >= 1.0:
                ex.state.reps       += 1
                ex.state.total_reps += 1
                self._plank_last_tick = now
                if ex.state.reps >= ex.GOAL_REPS:
                    ex.state.sets += 1
                    ex.state.reps  = 0
                    self.tts.say(
                        f"Set {ex.state.sets} complete! Rest 60 seconds.",
                        priority=True,
                    )

        state = ex.process_frame(lm)

        skel_color = (122, 214, 34) if state.is_good_form else (60, 64, 240)
        draw_landmark_connections(out, lm, ex.PRIMARY_CONNECTIONS, skel_color, 3)
        draw_landmark_points(out, lm, ex.PRIMARY_JOINTS, (64, 192, 240), 5)
        self._annotate_angles_on_pose(out, lm, state.angles)
        self._draw_hud(out, state)
        self._push_state(state)
        return out

    def _push_state(self, state):
        if not self.store:
            return
        ex = self._exercise
        base = {
            "exercise":     ex.NAME,
            "exercise_key": self._exercise_key,
            "elapsed":      self._elapsed_str,
            "paused":       self._paused,
            "goal_reps":    ex.GOAL_REPS,
            "is_timed":     ex.IS_TIMED,
            "joints":       ex.JOINTS,
            "angle_ranges": ex.ANGLE_RANGES,
        }
        if state is None:
            self.store.push_state({
                **base,
                "reps": 0, "sets": 0, "total_reps": 0,
                "form_score": 0, "scores": {}, "angles": [],
                "feedback": "No pose detected — step into frame",
                "is_good": False, "phase": "up",
            })
        else:
            self.store.push_state({
                **base,
                "reps":       state.reps,
                "sets":       state.sets,
                "total_reps": state.total_reps,
                "form_score": round(state.form_score, 1),
                "scores":     {k: round(v, 1) for k, v in state.scores.items()},
                "angles":     [round(a, 1) for a in state.angles],
                "feedback":   state.feedback_msg,
                "is_good":    state.is_good_form,
                "phase":      state.phase,
            })

    def _annotate_angles_on_pose(self, frame, lm, angles):
        MP_LM = mp.solutions.pose.PoseLandmark
        ann = {
            "squat":  [(MP_LM.LEFT_KNEE.value, 0), (MP_LM.RIGHT_KNEE.value, 1), (MP_LM.LEFT_HIP.value, 2)],
            "pushup": [(MP_LM.LEFT_ELBOW.value, 0), (MP_LM.RIGHT_ELBOW.value, 1)],
            "plank":  [(MP_LM.LEFT_HIP.value, 1)],
            "lunge":  [(MP_LM.LEFT_KNEE.value, 0), (MP_LM.RIGHT_KNEE.value, 1)],
        }.get(self._exercise_key, [])

        for joint_idx, angle_idx in ann:
            if angle_idx < len(angles):
                val = angles[angle_idx]
                lo, hi = self._exercise.ANGLE_RANGES[angle_idx]
                draw_angle_annotation(frame, lm, joint_idx, val, lo <= val <= hi)

    def _draw_hud(self, frame, state):
        ex = self._exercise
        if state is None:
            draw_header(frame, ex.NAME, 0, 0, 0.0, self._elapsed_str)
            draw_exercise_selector(frame, EXERCISE_DISPLAY, self._exercise_key)
            draw_controls_hint(frame)
            return
        draw_header(frame, ex.NAME, state.total_reps, state.sets,
                    state.form_score, self._elapsed_str)
        draw_feedback_banner(frame, state.feedback_msg, state.is_good_form)
        draw_joint_angles(frame, [
            JointAngleDisplay(
                name=ex.JOINTS[i],
                value=state.angles[i] if i < len(state.angles) else 0,
                min_val=ex.ANGLE_RANGES[i][0],
                max_val=ex.ANGLE_RANGES[i][1],
            )
            for i in range(len(ex.JOINTS))
        ])
        draw_exercise_selector(frame, EXERCISE_DISPLAY, self._exercise_key)
        draw_rep_counter(frame, state.reps, ex.GOAL_REPS)
        draw_form_score(frame, state.form_score)
        draw_score_breakdown(frame, state.scores)
        draw_controls_hint(frame)

    def _switch_exercise(self, new_key):
        self._exercise_key = new_key
        self._exercise     = EXERCISE_REGISTRY[new_key](tts=self.tts)
        self.tts.flush()
        self.tts.say(f"Switching to {EXERCISE_REGISTRY[new_key].NAME}.", priority=True)
        if new_key == "plank":
            self._plank_last_tick = time.time()
        print(f"  Switched to: {EXERCISE_REGISTRY[new_key].NAME}")

    def _handle_key(self, key, frame):
        if key in (ord("q"), 27):
            self._running = False
        elif key == ord(" "):
            self._paused = not self._paused
            self.tts.say("Paused." if self._paused else "Resuming.")
        elif key == ord("r"):
            self._exercise.reset_reps()
        elif key == ord("s"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"formscope_{ts}.jpg", frame)
        elif key in [ord(str(d)) for d in range(1, 5)]:
            new_ex = EXERCISE_KEYS.get(chr(key))
            if new_ex and new_ex != self._exercise_key:
                self._switch_exercise(new_ex)

    def _render_paused(self, frame):
        out = frame.copy()
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (out.shape[1], out.shape[0]), (18, 22, 28), -1)
        cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)
        H, W = out.shape[:2]
        msg = "PAUSED"
        (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 2.0, 3)
        cv2.putText(out, msg, ((W-tw)//2, H//2),
                    cv2.FONT_HERSHEY_DUPLEX, 2.0, (64, 192, 240), 3, cv2.LINE_AA)
        sub = "Press SPACE or click Pause in the dashboard"
        (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
        cv2.putText(out, sub, ((W-sw)//2, H//2+50),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (120, 130, 140), 1, cv2.LINE_AA)
        return out

    def _print_summary(self):
        st      = self._exercise.state
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
