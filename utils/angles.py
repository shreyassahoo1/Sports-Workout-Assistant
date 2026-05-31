"""
utils/angles.py — Joint angle computation from MediaPipe landmarks.

All functions operate on mediapipe.framework.formats.landmark_pb2.NormalizedLandmark
objects (accessible via pose_results.pose_landmarks.landmark[idx]).

Coordinate system:
    x  — normalised [0,1], left → right
    y  — normalised [0,1], top  → bottom
    z  — depth (negative = closer to camera), used sparingly
    visibility — confidence [0,1]
"""

import math
import numpy as np
from typing import Tuple, Optional


# ─── MediaPipe landmark indices ────────────────────────────────────────────────
class MP:
    NOSE            = 0
    LEFT_EYE        = 2
    RIGHT_EYE       = 5
    LEFT_SHOULDER   = 11
    RIGHT_SHOULDER  = 12
    LEFT_ELBOW      = 13
    RIGHT_ELBOW     = 14
    LEFT_WRIST      = 15
    RIGHT_WRIST     = 16
    LEFT_HIP        = 23
    RIGHT_HIP       = 24
    LEFT_KNEE       = 25
    RIGHT_KNEE      = 26
    LEFT_ANKLE      = 27
    RIGHT_ANKLE     = 28
    LEFT_HEEL       = 29
    RIGHT_HEEL      = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX= 32


def lm_to_point(landmark) -> Tuple[float, float]:
    """Return (x, y) from a normalised landmark."""
    return landmark.x, landmark.y


def angle_between(a, b, c) -> float:
    """
    Compute the angle at point B formed by the vectors BA and BC.

    Parameters
    ----------
    a, b, c : tuple (x, y)   — all normalised [0,1]

    Returns
    -------
    angle : float in degrees [0, 180]
    """
    ax, ay = a
    bx, by = b
    cx, cy = c

    ba = (ax - bx, ay - by)
    bc = (cx - bx, cy - by)

    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.hypot(*ba)
    mag_bc = math.hypot(*bc)

    if mag_ba < 1e-6 or mag_bc < 1e-6:
        return 0.0

    cos_angle = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_angle))


def spine_angle(landmarks) -> float:
    """
    Angle of the spine relative to vertical (0° = perfectly upright).
    Uses mid-shoulder → mid-hip vector.
    """
    ls = lm_to_point(landmarks[MP.LEFT_SHOULDER])
    rs = lm_to_point(landmarks[MP.RIGHT_SHOULDER])
    lh = lm_to_point(landmarks[MP.LEFT_HIP])
    rh = lm_to_point(landmarks[MP.RIGHT_HIP])

    mid_shoulder = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
    mid_hip      = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)

    dx = mid_hip[0] - mid_shoulder[0]
    dy = mid_hip[1] - mid_shoulder[1]          # positive = downward

    angle_from_vertical = math.degrees(math.atan2(abs(dx), abs(dy)))
    return round(angle_from_vertical, 1)


def knee_angle(landmarks, side: str = "left") -> float:
    """Flexion angle at the knee (180° = straight leg)."""
    if side == "left":
        hip, knee, ankle = MP.LEFT_HIP, MP.LEFT_KNEE, MP.LEFT_ANKLE
    else:
        hip, knee, ankle = MP.RIGHT_HIP, MP.RIGHT_KNEE, MP.RIGHT_ANKLE

    return round(angle_between(
        lm_to_point(landmarks[hip]),
        lm_to_point(landmarks[knee]),
        lm_to_point(landmarks[ankle]),
    ), 1)


def hip_angle(landmarks, side: str = "left") -> float:
    """Flexion angle at the hip (180° = straight body)."""
    if side == "left":
        shoulder, hip, knee = MP.LEFT_SHOULDER, MP.LEFT_HIP, MP.LEFT_KNEE
    else:
        shoulder, hip, knee = MP.RIGHT_SHOULDER, MP.RIGHT_HIP, MP.RIGHT_KNEE

    return round(angle_between(
        lm_to_point(landmarks[shoulder]),
        lm_to_point(landmarks[hip]),
        lm_to_point(landmarks[knee]),
    ), 1)


def elbow_angle(landmarks, side: str = "left") -> float:
    """Flexion angle at the elbow (180° = arm fully extended)."""
    if side == "left":
        shoulder, elbow, wrist = MP.LEFT_SHOULDER, MP.LEFT_ELBOW, MP.LEFT_WRIST
    else:
        shoulder, elbow, wrist = MP.RIGHT_SHOULDER, MP.RIGHT_ELBOW, MP.RIGHT_WRIST

    return round(angle_between(
        lm_to_point(landmarks[shoulder]),
        lm_to_point(landmarks[elbow]),
        lm_to_point(landmarks[wrist]),
    ), 1)


def shoulder_angle(landmarks, side: str = "left") -> float:
    """Angle at the shoulder (between hip → shoulder → elbow vectors)."""
    if side == "left":
        hip, shoulder, elbow = MP.LEFT_HIP, MP.LEFT_SHOULDER, MP.LEFT_ELBOW
    else:
        hip, shoulder, elbow = MP.RIGHT_HIP, MP.RIGHT_SHOULDER, MP.RIGHT_ELBOW

    return round(angle_between(
        lm_to_point(landmarks[hip]),
        lm_to_point(landmarks[shoulder]),
        lm_to_point(landmarks[elbow]),
    ), 1)


def ankle_angle(landmarks, side: str = "left") -> float:
    """Dorsiflexion angle at the ankle."""
    if side == "left":
        knee, ankle, foot = MP.LEFT_KNEE, MP.LEFT_ANKLE, MP.LEFT_FOOT_INDEX
    else:
        knee, ankle, foot = MP.RIGHT_KNEE, MP.RIGHT_ANKLE, MP.RIGHT_FOOT_INDEX

    return round(angle_between(
        lm_to_point(landmarks[knee]),
        lm_to_point(landmarks[ankle]),
        lm_to_point(landmarks[foot]),
    ), 1)


def body_alignment_angle(landmarks) -> float:
    """
    Full-body alignment: angle between shoulder-midpoint and ankle-midpoint.
    For planks: should be close to 180° (flat body).
    """
    ls = lm_to_point(landmarks[MP.LEFT_SHOULDER])
    rs = lm_to_point(landmarks[MP.RIGHT_SHOULDER])
    la = lm_to_point(landmarks[MP.LEFT_ANKLE])
    ra = lm_to_point(landmarks[MP.RIGHT_ANKLE])
    lh = lm_to_point(landmarks[MP.LEFT_HIP])
    rh = lm_to_point(landmarks[MP.RIGHT_HIP])

    mid_s = ((ls[0]+rs[0])/2, (ls[1]+rs[1])/2)
    mid_h = ((lh[0]+rh[0])/2, (lh[1]+rh[1])/2)
    mid_a = ((la[0]+ra[0])/2, (la[1]+ra[1])/2)

    return round(angle_between(mid_s, mid_h, mid_a), 1)


def visibility_ok(landmarks, indices: list, threshold: float = 0.5) -> bool:
    """Return True if all listed landmarks are visible above threshold."""
    return all(landmarks[i].visibility >= threshold for i in indices)


def smooth_angle(history: list, new_val: float, window: int = 5) -> float:
    """
    Append new_val to history and return an exponential moving average.
    Keeps history bounded to `window` entries.
    """
    history.append(new_val)
    if len(history) > window:
        history.pop(0)
    weights = [2 ** i for i in range(len(history))]
    return round(
        sum(v * w for v, w in zip(history, weights)) / sum(weights), 1
    )
