"""
exercises/base.py — Abstract base class for all exercises.

Each exercise subclass implements:
    - JOINTS           : list of joint names shown in the HUD
    - ANGLE_RANGES     : list of (min, max) tuples per joint
    - _compute_angles  : extract angles from MediaPipe landmarks
    - _check_form      : return (score_dict, feedback_msgs, is_good)
    - _check_rep       : return True when a full rep is completed

The base class owns:
    - Rep counting state machine
    - Smoothed angle history
    - Score aggregation
    - TTS cooldown delegation to TTSManager
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from utils.angles import smooth_angle


@dataclass
class ExerciseState:
    reps:          int   = 0
    sets:          int   = 0
    total_reps:    int   = 0
    form_score:    float = 0.0
    scores:        Dict[str, float] = field(default_factory=dict)
    angles:        List[float]      = field(default_factory=list)
    feedback_msg:  str  = "Get into position"
    is_good_form:  bool = True
    phase:         str  = "up"      # generic up/down phase for rep counting
    errors_this_set: int = 0


class BaseExercise(ABC):
    # ── Subclasses must define these ──────────────────────────────────────────
    NAME:         str
    GOAL_REPS:    int           # target reps per set
    TARGET_SETS:  int = 3
    IS_TIMED:     bool = False  # True for Plank (counts seconds, not reps)
    JOINTS:       List[str]     = []
    ANGLE_RANGES: List[Tuple[float, float]] = []

    # Skeleton connections to highlight (pairs of MediaPipe indices)
    PRIMARY_CONNECTIONS: List[Tuple[int, int]] = []
    # Landmark dots to draw
    PRIMARY_JOINTS: List[int] = []

    def __init__(self, tts=None):
        self.tts   = tts
        self.state = ExerciseState()
        self._angle_history: List[List[float]] = [[] for _ in self.JOINTS]

    # ── Public API (called by WorkoutSession each frame) ──────────────────────

    def process_frame(self, landmarks) -> ExerciseState:
        """
        Full per-frame pipeline:
          1. Compute raw angles
          2. Smooth angles
          3. Check form → scores + feedback
          4. Detect rep completion
          5. Return updated state
        """
        raw_angles = self._compute_angles(landmarks)
        smoothed   = [
            smooth_angle(self._angle_history[i], raw_angles[i])
            for i in range(len(raw_angles))
        ]
        self.state.angles = smoothed

        scores, msgs, is_good = self._check_form(landmarks, smoothed)
        self.state.scores     = scores
        self.state.is_good_form = is_good
        self.state.form_score = (
            sum(scores.values()) / len(scores) if scores else 0.0
        )

        if msgs:
            primary_msg = msgs[0]
            self.state.feedback_msg = primary_msg
            if self.tts:
                bad = not is_good
                self.tts.say(primary_msg, priority=bad)

        rep_done = self._check_rep(smoothed)
        if rep_done:
            self._on_rep_complete()

        return self.state

    def reset_reps(self):
        """Reset reps within the current set (not sets)."""
        self.state.reps   = 0
        self.state.phase  = "up"
        self.state.errors_this_set = 0
        for h in self._angle_history:
            h.clear()

    def reset_all(self):
        self.state = ExerciseState()
        for h in self._angle_history:
            h.clear()

    # ── Subclass interface ────────────────────────────────────────────────────

    @abstractmethod
    def _compute_angles(self, landmarks) -> List[float]:
        """Return a list of raw angles, one per entry in JOINTS."""

    @abstractmethod
    def _check_form(
        self,
        landmarks,
        angles: List[float],
    ) -> Tuple[Dict[str, float], List[str], bool]:
        """
        Returns
        -------
        scores   : dict {metric_name: 0–100}
        messages : list of feedback strings (first is shown on HUD)
        is_good  : True if form is acceptable overall
        """

    @abstractmethod
    def _check_rep(self, angles: List[float]) -> bool:
        """Return True exactly once when a full rep cycle completes."""

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _on_rep_complete(self):
        self.state.reps       += 1
        self.state.total_reps += 1

        if self.state.reps >= self.GOAL_REPS:
            self.state.sets += 1
            self.state.reps  = 0
            self.state.phase = "up"
            self.state.errors_this_set = 0
            if self.tts:
                self.tts.say(
                    f"Set {self.state.sets} complete. Rest for 60 seconds.",
                    priority=True,
                )

    def angle_in_range(self, angle: float, idx: int) -> bool:
        lo, hi = self.ANGLE_RANGES[idx]
        return lo <= angle <= hi

    def score_from_angle(
        self,
        angle: float,
        lo: float,
        hi: float,
        perfect_lo: float = None,
        perfect_hi: float = None,
    ) -> float:
        """
        Returns 0–100 based on how well `angle` sits within [lo, hi].
        If perfect_lo/hi supplied, 100 within that inner band, tapering to
        50 at the outer edge, 0 outside.
        """
        if lo <= angle <= hi:
            if perfect_lo is not None and perfect_hi is not None:
                if perfect_lo <= angle <= perfect_hi:
                    return 100.0
                # taper toward outer edge
                if angle < perfect_lo:
                    return 50.0 + 50.0 * (angle - lo) / max(perfect_lo - lo, 1)
                return 50.0 + 50.0 * (hi - angle) / max(hi - perfect_hi, 1)
            return 100.0

        # Outside range — score drops quickly
        margin = (hi - lo) * 0.3
        if angle < lo:
            dist = lo - angle
        else:
            dist = angle - hi
        return max(0.0, 50.0 - 50.0 * (dist / max(margin, 1)))
