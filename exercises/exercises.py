"""
exercises/exercises.py — Concrete exercise implementations.

Each class defines:
  - Joint angle thresholds (from biomechanics literature)
  - Form checks with scored sub-metrics
  - Rep detection state machine
  - TTS feedback messages per fault
"""

from typing import List, Tuple, Dict
from exercises.base import BaseExercise
from utils.angles import (
    knee_angle, hip_angle, elbow_angle, shoulder_angle,
    ankle_angle, spine_angle, body_alignment_angle,
    visibility_ok, MP,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  SQUAT
# ═══════════════════════════════════════════════════════════════════════════════
class Squat(BaseExercise):
    NAME       = "Squat"
    GOAL_REPS  = 12
    TARGET_SETS = 3

    JOINTS = ["L. Knee", "R. Knee", "Hip Flex", "Spine"]
    ANGLE_RANGES = [
        (60, 175),    # knee (descent to near-parallel)
        (60, 175),    # knee right
        (65, 175),    # hip flexion
        (0,  15),     # spine lean from vertical (degrees)
    ]

    PRIMARY_CONNECTIONS = [
        (MP.LEFT_SHOULDER,  MP.RIGHT_SHOULDER),
        (MP.LEFT_SHOULDER,  MP.LEFT_HIP),
        (MP.RIGHT_SHOULDER, MP.RIGHT_HIP),
        (MP.LEFT_HIP,       MP.RIGHT_HIP),
        (MP.LEFT_HIP,       MP.LEFT_KNEE),
        (MP.RIGHT_HIP,      MP.RIGHT_KNEE),
        (MP.LEFT_KNEE,      MP.LEFT_ANKLE),
        (MP.RIGHT_KNEE,     MP.RIGHT_ANKLE),
    ]
    PRIMARY_JOINTS = [
        MP.LEFT_SHOULDER, MP.RIGHT_SHOULDER,
        MP.LEFT_HIP, MP.RIGHT_HIP,
        MP.LEFT_KNEE, MP.RIGHT_KNEE,
        MP.LEFT_ANKLE, MP.RIGHT_ANKLE,
    ]

    def __init__(self, tts=None):
        super().__init__(tts)
        self._phase = "up"   # "up" | "down"

    def _compute_angles(self, lm) -> List[float]:
        lk = knee_angle(lm, "left")
        rk = knee_angle(lm, "right")
        lh = hip_angle(lm, "left")
        sp = spine_angle(lm)
        return [lk, rk, lh, sp]

    def _check_form(self, lm, angles) -> Tuple[Dict, List[str], bool]:
        lk, rk, lh, sp = angles
        msgs = []
        is_good = True

        # ── Spine ─────────────────────────────────────────
        if sp > 20:
            msgs.append("Chest up — keep your spine neutral.")
            is_good = False
        elif sp > 12:
            msgs.append("Slight forward lean — brace your core.")

        # ── Knees ─────────────────────────────────────────
        if abs(lk - rk) > 18:
            msgs.append("Uneven knees — balance your weight.")
            is_good = False

        # ── Depth ─────────────────────────────────────────
        in_bottom = lk < 130 or rk < 130
        if in_bottom:
            if lk < 60 or rk < 60:
                msgs.append("Too deep — risk of knee strain.")
                is_good = False
            else:
                if not msgs:
                    msgs.append("Good depth! Drive through your heels.")

        # ── Hip ───────────────────────────────────────────
        if lh < 80 and in_bottom:
            msgs.append("Hip flexion too deep — check mobility.")

        if not msgs:
            if lk > 160:
                msgs.append("Lower into the squat — reach parallel.")
            else:
                msgs.append("Great squat form!")

        scores = {
            "Spine":     self.score_from_angle(sp, 0, 15, 0, 8),
            "Depth":     self.score_from_angle(min(lk, rk), 60, 115, 75, 100),
            "Knee Sym":  max(0.0, 100.0 - abs(lk - rk) * 3),
            "Hip Flex":  self.score_from_angle(lh, 65, 120, 75, 100),
        }
        return scores, msgs, is_good

    def _check_rep(self, angles) -> bool:
        lk = angles[0]
        if self._phase == "up" and lk < 110:
            self._phase = "down"
        elif self._phase == "down" and lk > 160:
            self._phase = "up"
            self.state.phase = "up"
            return True
        self.state.phase = self._phase
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#  PUSH-UP
# ═══════════════════════════════════════════════════════════════════════════════
class PushUp(BaseExercise):
    NAME       = "Push-up"
    GOAL_REPS  = 10
    TARGET_SETS = 3

    JOINTS = ["L. Elbow", "R. Elbow", "Shoulder", "Body Line"]
    ANGLE_RANGES = [
        (25, 165),    # elbow flexion (full ROM)
        (25, 165),
        (45, 100),    # shoulder at bottom of push-up
        (165, 185),   # body alignment (straight plank-like position)
    ]

    PRIMARY_CONNECTIONS = [
        (MP.LEFT_SHOULDER,  MP.RIGHT_SHOULDER),
        (MP.LEFT_SHOULDER,  MP.LEFT_ELBOW),
        (MP.RIGHT_SHOULDER, MP.RIGHT_ELBOW),
        (MP.LEFT_ELBOW,     MP.LEFT_WRIST),
        (MP.RIGHT_ELBOW,    MP.RIGHT_WRIST),
        (MP.LEFT_SHOULDER,  MP.LEFT_HIP),
        (MP.RIGHT_SHOULDER, MP.RIGHT_HIP),
        (MP.LEFT_HIP,       MP.LEFT_KNEE),
        (MP.RIGHT_HIP,      MP.RIGHT_KNEE),
        (MP.LEFT_KNEE,      MP.LEFT_ANKLE),
        (MP.RIGHT_KNEE,     MP.RIGHT_ANKLE),
    ]
    PRIMARY_JOINTS = [
        MP.LEFT_SHOULDER, MP.RIGHT_SHOULDER,
        MP.LEFT_ELBOW, MP.RIGHT_ELBOW,
        MP.LEFT_WRIST, MP.RIGHT_WRIST,
        MP.LEFT_HIP, MP.RIGHT_HIP,
    ]

    def __init__(self, tts=None):
        super().__init__(tts)
        self._phase = "up"

    def _compute_angles(self, lm) -> List[float]:
        le = elbow_angle(lm, "left")
        re = elbow_angle(lm, "right")
        ls = shoulder_angle(lm, "left")
        ba = body_alignment_angle(lm)
        return [le, re, ls, ba]

    def _check_form(self, lm, angles) -> Tuple[Dict, List[str], bool]:
        le, re, ls, ba = angles
        msgs = []
        is_good = True

        # ── Body line ─────────────────────────────────────
        if ba < 160:
            msgs.append("Hips sagging — squeeze your core and glutes.")
            is_good = False
        elif ba > 190:
            msgs.append("Hips too high — lower into a flat body line.")
            is_good = False

        # ── Elbow flare ───────────────────────────────────
        if abs(le - re) > 20:
            msgs.append("Uneven arms — keep both elbows symmetrical.")
            is_good = False

        # ── Elbow angle at bottom ─────────────────────────
        in_bottom = le < 90 or re < 90
        if in_bottom:
            if le < 30 or re < 30:
                msgs.append("Wrists past elbows — risk of strain.")
                is_good = False
            elif not msgs:
                msgs.append("Great range! Full chest to floor.")

        # ── Lockout at top ────────────────────────────────
        if le > 155 and re > 155 and not in_bottom:
            if not msgs:
                msgs.append("Good lockout — controlled descent next.")

        if not msgs:
            msgs.append("Solid push-up form!")

        scores = {
            "Body Line":   self.score_from_angle(ba, 165, 185, 170, 180),
            "Elbow Flex":  self.score_from_angle(min(le, re), 25, 90, 40, 80),
            "Elbow Sym":   max(0.0, 100.0 - abs(le - re) * 3),
            "Shoulder":    self.score_from_angle(ls, 45, 100, 55, 90),
        }
        return scores, msgs, is_good

    def _check_rep(self, angles) -> bool:
        le = angles[0]
        if self._phase == "up" and le < 85:
            self._phase = "down"
        elif self._phase == "down" and le > 155:
            self._phase = "up"
            self.state.phase = "up"
            return True
        self.state.phase = self._phase
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#  PLANK  (time-based, not rep-based)
# ═══════════════════════════════════════════════════════════════════════════════
class Plank(BaseExercise):
    NAME       = "Plank"
    GOAL_REPS  = 60    # seconds
    TARGET_SETS = 3
    IS_TIMED   = True

    JOINTS = ["Shoulder", "Hip Align", "Neck", "Core"]
    ANGLE_RANGES = [
        (82, 98),     # shoulder stack (near 90°)
        (170, 190),   # hip alignment (flat)
        (165, 185),   # neck (neutral)
        (170, 190),   # core line
    ]

    PRIMARY_CONNECTIONS = [
        (MP.LEFT_SHOULDER,  MP.RIGHT_SHOULDER),
        (MP.LEFT_SHOULDER,  MP.LEFT_ELBOW),
        (MP.RIGHT_SHOULDER, MP.RIGHT_ELBOW),
        (MP.LEFT_SHOULDER,  MP.LEFT_HIP),
        (MP.RIGHT_SHOULDER, MP.RIGHT_HIP),
        (MP.LEFT_HIP,       MP.LEFT_KNEE),
        (MP.RIGHT_HIP,      MP.RIGHT_KNEE),
        (MP.LEFT_KNEE,      MP.LEFT_ANKLE),
        (MP.RIGHT_KNEE,     MP.RIGHT_ANKLE),
    ]
    PRIMARY_JOINTS = [
        MP.LEFT_SHOULDER, MP.RIGHT_SHOULDER,
        MP.LEFT_HIP, MP.RIGHT_HIP,
        MP.LEFT_ANKLE, MP.RIGHT_ANKLE,
        MP.LEFT_ELBOW, MP.RIGHT_ELBOW,
    ]

    def __init__(self, tts=None):
        super().__init__(tts)
        self._frame_count = 0

    def _compute_angles(self, lm) -> List[float]:
        ls  = shoulder_angle(lm, "left")
        ba  = body_alignment_angle(lm)
        sp  = spine_angle(lm)
        core = body_alignment_angle(lm)   # re-used as core proxy
        return [ls, ba, sp, core]

    def _check_form(self, lm, angles) -> Tuple[Dict, List[str], bool]:
        ls, ba, sp, core = angles
        msgs = []
        is_good = True

        if ba < 165:
            msgs.append("Hips dropping — push them back up.")
            is_good = False
        elif ba > 192:
            msgs.append("Hips too high — lower to a straight line.")
            is_good = False

        if sp > 18:
            msgs.append("Head dropping — keep neck neutral, eyes down.")
            is_good = False

        if not msgs:
            msgs.append("Solid plank! Breathe steadily.")

        # Timed encouragement
        self._frame_count += 1
        if self._frame_count % 150 == 0 and is_good:  # ~5 s at 30 fps
            msgs = ["Keep going — you're doing great!"]
            if self.tts:
                self.tts.say("Keep it up! Stay strong.", priority=False)

        scores = {
            "Hip Align":  self.score_from_angle(ba,  165, 190, 172, 185),
            "Shoulder":   self.score_from_angle(ls,   82,  98,  86,  94),
            "Neck":       self.score_from_angle(sp,    0,  15,   0,   8),
            "Core":       self.score_from_angle(core, 170, 190, 175, 185),
        }
        return scores, msgs, is_good

    def _check_rep(self, angles) -> bool:
        """Plank 'reps' are ticked externally by WorkoutSession as seconds."""
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#  LUNGE
# ═══════════════════════════════════════════════════════════════════════════════
class Lunge(BaseExercise):
    NAME       = "Lunge"
    GOAL_REPS  = 10
    TARGET_SETS = 3

    JOINTS = ["Front Knee", "Back Knee", "Hip", "Torso"]
    ANGLE_RANGES = [
        (70, 175),    # front knee (90° at bottom)
        (80, 175),    # back knee
        (80, 175),    # hip angle
        (0,  18),     # torso lean from vertical
    ]

    PRIMARY_CONNECTIONS = [
        (MP.LEFT_SHOULDER,  MP.RIGHT_SHOULDER),
        (MP.LEFT_SHOULDER,  MP.LEFT_HIP),
        (MP.RIGHT_SHOULDER, MP.RIGHT_HIP),
        (MP.LEFT_HIP,       MP.RIGHT_HIP),
        (MP.LEFT_HIP,       MP.LEFT_KNEE),
        (MP.RIGHT_HIP,      MP.RIGHT_KNEE),
        (MP.LEFT_KNEE,      MP.LEFT_ANKLE),
        (MP.RIGHT_KNEE,     MP.RIGHT_ANKLE),
    ]
    PRIMARY_JOINTS = [
        MP.LEFT_SHOULDER, MP.RIGHT_SHOULDER,
        MP.LEFT_HIP, MP.RIGHT_HIP,
        MP.LEFT_KNEE, MP.RIGHT_KNEE,
        MP.LEFT_ANKLE, MP.RIGHT_ANKLE,
    ]

    def __init__(self, tts=None):
        super().__init__(tts)
        self._phase = "up"

    def _compute_angles(self, lm) -> List[float]:
        lk = knee_angle(lm, "left")
        rk = knee_angle(lm, "right")
        lh = hip_angle(lm, "left")
        sp = spine_angle(lm)
        return [lk, rk, lh, sp]

    def _check_form(self, lm, angles) -> Tuple[Dict, List[str], bool]:
        front_k, back_k, hip_a, sp = angles
        msgs = []
        is_good = True

        # Torso upright
        if sp > 22:
            msgs.append("Lean back — torso should be upright.")
            is_good = False

        # Front knee over toe check (proxy via knee angle)
        in_bottom = front_k < 115
        if in_bottom:
            if front_k < 65:
                msgs.append("Front knee too far forward — step wider.")
                is_good = False
            elif not msgs:
                msgs.append("Good lunge depth!")

        # Step length proxy
        if abs(front_k - back_k) < 20 and front_k > 140:
            msgs.append("Lunge deeper — lower back knee toward floor.")

        if not msgs:
            msgs.append("Drive up through your front heel!")

        scores = {
            "Front Knee":  self.score_from_angle(front_k, 70, 100, 80, 95),
            "Torso":       self.score_from_angle(sp,       0,  18,  0,  10),
            "Hip Flex":    self.score_from_angle(hip_a,   80, 120, 85, 110),
            "Symmetry":    self.score_from_angle(back_k,  80, 120, 85, 110),
        }
        return scores, msgs, is_good

    def _check_rep(self, angles) -> bool:
        front_k = angles[0]
        if self._phase == "up" and front_k < 110:
            self._phase = "down"
        elif self._phase == "down" and front_k > 160:
            self._phase = "up"
            self.state.phase = "up"
            return True
        self.state.phase = self._phase
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#  BICEP CURL  — KNN model (lean detection) + angle rules
# ═══════════════════════════════════════════════════════════════════════════════
class BicepCurl(BaseExercise):
    NAME        = "Bicep Curl"
    GOAL_REPS   = 10
    TARGET_SETS = 3

    JOINTS = ["L. Elbow", "R. Elbow", "Upper Arm", "Torso"]
    ANGLE_RANGES = [
        (20,  160),   # elbow flexion full ROM
        (20,  160),   # elbow right
        (0,   40),    # upper-arm drift from vertical (loose arm check)
        (0,   15),    # torso lean from vertical
    ]

    PRIMARY_CONNECTIONS = [
        (MP.LEFT_SHOULDER,  MP.RIGHT_SHOULDER),
        (MP.LEFT_SHOULDER,  MP.LEFT_ELBOW),
        (MP.RIGHT_SHOULDER, MP.RIGHT_ELBOW),
        (MP.LEFT_ELBOW,     MP.LEFT_WRIST),
        (MP.RIGHT_ELBOW,    MP.RIGHT_WRIST),
        (MP.LEFT_SHOULDER,  MP.LEFT_HIP),
        (MP.RIGHT_SHOULDER, MP.RIGHT_HIP),
    ]
    PRIMARY_JOINTS = [
        MP.LEFT_SHOULDER, MP.RIGHT_SHOULDER,
        MP.LEFT_ELBOW,    MP.RIGHT_ELBOW,
        MP.LEFT_WRIST,    MP.RIGHT_WRIST,
        MP.LEFT_HIP,      MP.RIGHT_HIP,
    ]

    _knn          = None
    _scaler       = None
    _model_loaded = False

    @classmethod
    def _load_model(cls):
        if cls._model_loaded:
            return
        import os, pickle, warnings
        model_dir   = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "model")
        )
        knn_path    = os.path.join(model_dir, "KNN_model.pkl")
        scaler_path = os.path.join(model_dir, "input_scaler.pkl")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with open(knn_path, "rb") as f:
                    cls._knn = pickle.load(f)
                with open(scaler_path, "rb") as f:
                    cls._scaler = pickle.load(f)
            print("  ✓ BicepCurl KNN model loaded.")
        except Exception as e:
            print(f"  ⚠ BicepCurl model not found ({e}). Using angle-only mode.")
        cls._model_loaded = True

    def __init__(self, tts=None):
        super().__init__(tts)
        self.__class__._load_model()
        self._phase = "down"

    def _compute_angles(self, lm) -> List[float]:
        import math
        le    = elbow_angle(lm, "left")
        re    = elbow_angle(lm, "right")
        ls_pt = (lm[MP.LEFT_SHOULDER].x, lm[MP.LEFT_SHOULDER].y)
        le_pt = (lm[MP.LEFT_ELBOW].x,    lm[MP.LEFT_ELBOW].y)
        dx    = le_pt[0] - ls_pt[0]
        dy    = le_pt[1] - ls_pt[1]
        upper_arm_drift = math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))
        sp    = spine_angle(lm)
        return [le, re, upper_arm_drift, sp]

    def _landmarks_to_row(self, lm) -> list:
        USED = [
            MP.NOSE,
            MP.LEFT_SHOULDER,  MP.RIGHT_SHOULDER,
            MP.RIGHT_ELBOW,    MP.LEFT_ELBOW,
            MP.RIGHT_WRIST,    MP.LEFT_WRIST,
            MP.LEFT_HIP,       MP.RIGHT_HIP,
        ]
        row = []
        for idx in USED:
            p = lm[idx]
            row.extend([p.x, p.y, p.z, p.visibility])
        return row  # 9 × 4 = 36 features

    def _check_form(self, lm, angles) -> Tuple[Dict, List[str], bool]:
        le, re, upper_arm_drift, sp = angles
        msgs    = []
        is_good = True

        # 1. Lean detection via KNN model
        lean_error = False
        if self._knn is not None and self._scaler is not None:
            try:
                import numpy as np, warnings
                row    = self._landmarks_to_row(lm)
                scaled = self._scaler.transform([row])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pred = self._knn.predict(scaled)[0]
                if pred == "L":
                    msgs.append("Don't lean back — keep torso upright.")
                    lean_error = True
                    is_good    = False
            except Exception:
                pass

        # 2. Loose upper arm
        if upper_arm_drift > 40:
            msgs.append("Upper arm moving — pin elbows to your sides.")
            is_good = False

        # 3. Weak peak contraction
        in_top = le < 70 or re < 70
        if in_top:
            if le > 60 and re > 60:
                msgs.append("Curl higher — squeeze at the top!")
                is_good = False
            elif not msgs:
                msgs.append("Good contraction at the top!")

        # 4. Spine lean fallback (no model)
        if not lean_error and sp > 18:
            msgs.append("Torso leaning back — stay upright.")
            is_good = False

        if not msgs:
            msgs.append("Great curl form!")

        scores = {
            "Elbow ROM":  self.score_from_angle(min(le, re), 20, 60, 25, 50),
            "Upper Arm":  self.score_from_angle(upper_arm_drift, 0, 40, 0, 20),
            "Torso":      self.score_from_angle(sp, 0, 15, 0, 8),
            "Symmetry":   max(0.0, 100.0 - abs(le - re) * 2),
        }
        return scores, msgs, is_good

    def _check_rep(self, angles) -> bool:
        le = angles[0]
        if self._phase == "down" and le < 70:
            self._phase = "up"
        elif self._phase == "up" and le > 140:
            self._phase = "down"
            self.state.phase = "up"
            return True
        self.state.phase = self._phase
        return False


# ── Registry ──────────────────────────────────────────────────────────────────
EXERCISE_REGISTRY = {
    "squat":  Squat,
    "pushup": PushUp,
    "plank":  Plank,
    "lunge":  Lunge,
    "bicep":  BicepCurl,
}

EXERCISE_KEYS = {
    "1": "squat",
    "2": "pushup",
    "3": "plank",
    "4": "lunge",
    "5": "bicep",
}

EXERCISE_DISPLAY = ["Squat", "Push-up", "Plank", "Lunge", "Bicep Curl"]
