"""
utils/hud.py — OpenCV HUD rendering helpers.

All drawing functions take a numpy frame (BGR) and modify it in-place.
Uses semi-transparent overlay blending (frame.copy() + cv2.addWeighted)
so backgrounds don't fully obscure the pose underneath.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


# ── Colour palette (BGR) ──────────────────────────────────────────────────────
C_ACC    = (64,  192, 240)   # gold-ish accent
C_GREEN  = (122, 214,  34)   # good form
C_RED    = ( 60,  64, 240)   # bad form / warning
C_BLUE   = (240, 144,  64)   # informational
C_ORANGE = ( 32,  90, 224)   # medium concern
C_WHITE  = (240, 240, 240)
C_MUTED  = (120, 130, 140)
C_DARK   = ( 18,  22,  28)
C_SURF   = ( 28,  38,  48)

FONT       = cv2.FONT_HERSHEY_DUPLEX
FONT_MONO  = cv2.FONT_HERSHEY_PLAIN


@dataclass
class JointAngleDisplay:
    name:     str
    value:    float
    min_val:  float
    max_val:  float


def alpha_rect(
    frame: np.ndarray,
    x: int, y: int, w: int, h: int,
    color: Tuple[int, int, int],
    alpha: float = 0.55,
    radius: int = 8,
) -> None:
    """Draw a rounded semi-transparent rectangle."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    # Border
    cv2.rectangle(frame, (x, y), (x + w, y + h), C_ACC, 1, cv2.LINE_AA)


def text(
    frame: np.ndarray,
    msg: str,
    x: int, y: int,
    color: Tuple[int, int, int] = C_WHITE,
    scale: float = 0.55,
    thickness: int = 1,
    font=FONT,
) -> int:
    """Draw text and return the baseline y for chaining."""
    cv2.putText(frame, msg, (x, y), font, scale, C_DARK, thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, msg, (x, y), font, scale, color,  thickness,     cv2.LINE_AA)
    return y


def big_text(
    frame: np.ndarray,
    msg: str,
    x: int, y: int,
    color: Tuple[int, int, int] = C_WHITE,
    scale: float = 1.2,
    thickness: int = 2,
) -> None:
    """Large bold text with drop-shadow."""
    cv2.putText(frame, msg, (x + 2, y + 2), FONT, scale, C_DARK, thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, msg, (x, y),          FONT, scale, color,  thickness,     cv2.LINE_AA)


def draw_header(
    frame: np.ndarray,
    exercise_name: str,
    reps: int,
    sets: int,
    accuracy: float,
    elapsed: str,
) -> None:
    """Top bar: branding, exercise name, session stats."""
    H, W = frame.shape[:2]
    alpha_rect(frame, 0, 0, W, 52, C_DARK, alpha=0.80)
    # Accent line under header
    cv2.line(frame, (0, 52), (W, 52), C_ACC, 2)

    # Brand
    big_text(frame, "FORMSCOPE", 12, 36, C_ACC, scale=0.9, thickness=2)

    # Exercise badge
    ex_x = 200
    alpha_rect(frame, ex_x, 8, 150, 36, C_SURF, alpha=0.9)
    text(frame, exercise_name.upper(), ex_x + 10, 32, C_ACC, scale=0.65, thickness=1)

    # Stats row (right side)
    stats = [
        ("REPS", str(reps)),
        ("SETS", str(sets)),
        ("ACC",  f"{accuracy:.0f}%"),
        ("TIME", elapsed),
    ]
    sx = W - 280
    for label, val in stats:
        text(frame, label, sx, 22, C_MUTED, scale=0.38)
        text(frame, val,   sx, 42, C_WHITE, scale=0.65, thickness=1)
        sx += 68


def draw_feedback_banner(
    frame: np.ndarray,
    message: str,
    good: bool,
    sub_message: str = "",
) -> None:
    """Large centred feedback banner below the header."""
    H, W = frame.shape[:2]
    color = C_GREEN if good else C_RED
    banner_y = 62

    alpha_rect(frame, 0, banner_y, W, 46, C_DARK, alpha=0.75)
    cv2.line(frame, (0, banner_y), (W, banner_y), color, 2)
    cv2.line(frame, (0, banner_y + 46), (W, banner_y + 46), color, 1)

    # Pulsing dot
    dot_x = 18
    cv2.circle(frame, (dot_x, banner_y + 23), 6, color, -1, cv2.LINE_AA)

    # Main message
    (tw, _), _ = cv2.getTextSize(message, FONT, 0.70, 2)
    mx = (W - tw) // 2
    big_text(frame, message, mx, banner_y + 30, color, scale=0.70, thickness=2)

    # Sub-message
    if sub_message:
        (sw, _), _ = cv2.getTextSize(sub_message, FONT, 0.40, 1)
        sx = (W - sw) // 2
        text(frame, sub_message, sx, banner_y + 44, C_MUTED, scale=0.40)


def draw_joint_angles(
    frame: np.ndarray,
    angles: List[JointAngleDisplay],
    x_start: int = 12,
    y_start: int = None,
) -> None:
    """Left panel: joint angle gauges as arc bars."""
    H, W = frame.shape[:2]
    if y_start is None:
        y_start = H - (len(angles) * 58) - 10

    panel_h = len(angles) * 58 + 12
    alpha_rect(frame, x_start - 4, y_start - 4, 180, panel_h, C_DARK, alpha=0.75)

    for i, a in enumerate(angles):
        y = y_start + i * 58
        in_range = a.min_val <= a.value <= a.max_val
        bar_color = C_GREEN if in_range else C_RED
        neutral   = C_SURF

        # Label + value
        text(frame, a.name, x_start + 2, y + 14, C_MUTED, scale=0.38)
        val_col = C_GREEN if in_range else C_RED
        text(frame, f"{a.value:.0f}°", x_start + 2, y + 30, val_col, scale=0.62, thickness=1)

        # Range indicator bar
        bar_x, bar_y = x_start + 2, y + 36
        bar_w = 160
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 7), neutral, -1)

        # Clip value to range for display
        span = max(a.max_val - a.min_val, 1)
        total_span = 180.0   # full possible range
        start_frac = (a.min_val) / total_span
        end_frac   = (a.max_val) / total_span
        val_frac   = max(0.0, min(1.0, (a.value) / total_span))

        # Good zone
        gz_x  = bar_x + int(start_frac * bar_w)
        gz_w  = int((end_frac - start_frac) * bar_w)
        cv2.rectangle(frame, (gz_x, bar_y), (gz_x + gz_w, bar_y + 7),
                      (40, 80, 40), -1)

        # Current value tick
        tick_x = bar_x + int(val_frac * bar_w)
        cv2.rectangle(frame, (tick_x - 2, bar_y - 2), (tick_x + 2, bar_y + 9),
                      bar_color, -1, cv2.LINE_AA)


def draw_rep_counter(
    frame: np.ndarray,
    reps: int,
    goal: int,
    x: int = None,
    y: int = None,
) -> None:
    """Bottom-right large rep ring."""
    H, W = frame.shape[:2]
    cx = x or W - 70
    cy = y or H - 70
    r  = 50

    # Background circle
    cv2.circle(frame, (cx, cy), r, C_SURF, -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), r, C_ACC,   2, cv2.LINE_AA)

    # Progress arc
    pct   = min(reps / max(goal, 1), 1.0)
    start = -90     # top
    sweep = int(360 * pct)
    if sweep > 0:
        cv2.ellipse(frame, (cx, cy), (r, r), 0, start, start + sweep,
                    C_GREEN, 4, cv2.LINE_AA)

    # Rep number
    rep_str = str(reps)
    (tw, th), _ = cv2.getTextSize(rep_str, FONT, 1.0, 2)
    cv2.putText(frame, rep_str, (cx - tw//2, cy + th//2),
                FONT, 1.0, C_WHITE, 2, cv2.LINE_AA)

    # Goal label
    goal_str = f"/ {goal}"
    text(frame, goal_str, cx - 16, cy + r + 16, C_MUTED, scale=0.40)


def draw_form_score(
    frame: np.ndarray,
    score: float,
    x: int = None,
    y: int = 115,
) -> None:
    """Top-right form score percentage with colour bar."""
    H, W = frame.shape[:2]
    px = x or W - 190
    panel_w = 178

    alpha_rect(frame, px, y, panel_w, 64, C_DARK, alpha=0.78)
    text(frame, "FORM SCORE", px + 8, y + 16, C_MUTED, scale=0.38)

    color = C_GREEN if score >= 70 else (C_ORANGE if score >= 50 else C_RED)
    big_text(frame, f"{score:.0f}%", px + 8, y + 44, color, scale=1.0, thickness=2)

    # Score bar
    bx, by = px + 8, y + 52
    bw = panel_w - 16
    cv2.rectangle(frame, (bx, by), (bx + bw, by + 7), C_SURF, -1)
    fill_w = int(bw * min(score / 100.0, 1.0))
    if fill_w > 0:
        cv2.rectangle(frame, (bx, by), (bx + fill_w, by + 7), color, -1)


def draw_score_breakdown(
    frame: np.ndarray,
    scores: dict,
    x: int = None,
    y: int = 185,
) -> None:
    """Right panel: per-metric scores."""
    H, W = frame.shape[:2]
    px = x or W - 190
    panel_w = 178
    panel_h = len(scores) * 22 + 24

    alpha_rect(frame, px, y, panel_w, panel_h, C_DARK, alpha=0.78)
    text(frame, "BREAKDOWN", px + 8, y + 16, C_MUTED, scale=0.38)

    for i, (name, val) in enumerate(scores.items()):
        ry = y + 30 + i * 22
        color = C_GREEN if val >= 70 else (C_ORANGE if val >= 50 else C_RED)
        text(frame, name[:14], px + 8, ry, C_MUTED, scale=0.36)
        text(frame, f"{val:.0f}%", px + panel_w - 44, ry, color, scale=0.40)
        bx, by = px + 8, ry + 3
        bw = panel_w - 52 - 8
        cv2.rectangle(frame, (bx, by), (bx + bw, by + 5), C_SURF, -1)
        fw = int(bw * min(val / 100.0, 1.0))
        if fw > 0:
            cv2.rectangle(frame, (bx, by), (bx + fw, by + 5), color, -1)


def draw_exercise_selector(
    frame: np.ndarray,
    exercises: List[str],
    current: str,
    x: int = None,
    y: int = 115,
) -> None:
    """Left panel: exercise selection pills (display-only, keys 1-4 switch)."""
    H, W = frame.shape[:2]
    px = x or 12
    keys = ["1", "2", "3", "4"]

    for i, (key, ex) in enumerate(zip(keys, exercises)):
        ey  = y + i * 34
        active = ex.lower() == current.lower()
        bg  = C_ACC if active else C_SURF
        fg  = C_DARK if active else C_MUTED
        alpha_rect(frame, px, ey, 170, 28, bg, alpha=0.88 if active else 0.65)
        text(frame, f"[{key}] {ex.upper()}", px + 8, ey + 19,
             fg, scale=0.46 if active else 0.42,
             thickness=2 if active else 1)


def draw_controls_hint(frame: np.ndarray) -> None:
    """Bottom-centre subtle keyboard hints."""
    H, W = frame.shape[:2]
    hints = "SPACE: pause  |  R: reset  |  S: screenshot  |  Q: quit"
    (tw, _), _ = cv2.getTextSize(hints, FONT, 0.35, 1)
    px = (W - tw) // 2
    text(frame, hints, px, H - 8, C_MUTED, scale=0.35)


def draw_landmark_connections(
    frame: np.ndarray,
    landmarks,
    connections: list,
    color: Tuple[int, int, int] = C_GREEN,
    thickness: int = 2,
) -> None:
    """Draw MediaPipe skeleton connections manually (for custom styling)."""
    H, W = frame.shape[:2]
    for start_idx, end_idx in connections:
        s = landmarks[start_idx]
        e = landmarks[end_idx]
        if s.visibility < 0.4 or e.visibility < 0.4:
            continue
        sx, sy = int(s.x * W), int(s.y * H)
        ex, ey = int(e.x * W), int(e.y * H)
        cv2.line(frame, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)


def draw_landmark_points(
    frame: np.ndarray,
    landmarks,
    indices: list,
    color: Tuple[int, int, int] = C_ACC,
    radius: int = 5,
) -> None:
    """Draw specific landmark joints as filled circles."""
    H, W = frame.shape[:2]
    for idx in indices:
        lm = landmarks[idx]
        if lm.visibility < 0.4:
            continue
        px, py = int(lm.x * W), int(lm.y * H)
        cv2.circle(frame, (px, py), radius + 2, C_DARK, -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), radius, color, -1, cv2.LINE_AA)


def draw_angle_annotation(
    frame: np.ndarray,
    landmarks,
    joint_idx: int,
    angle: float,
    good: bool,
) -> None:
    """Draw the angle value floating next to a joint landmark."""
    H, W = frame.shape[:2]
    lm = landmarks[joint_idx]
    if lm.visibility < 0.4:
        return
    px = int(lm.x * W) + 10
    py = int(lm.y * H) - 10
    color = C_GREEN if good else C_RED
    text(frame, f"{angle:.0f}°", px, py, color, scale=0.45, thickness=1)


def draw_no_pose(frame: np.ndarray) -> None:
    """Overlay shown when no pose is detected."""
    H, W = frame.shape[:2]
    alpha_rect(frame, W//2 - 180, H//2 - 28, 360, 56, C_DARK, alpha=0.85)
    msg = "No pose detected — step into frame"
    (tw, _), _ = cv2.getTextSize(msg, FONT, 0.55, 1)
    text(frame, msg, (W - tw)//2, H//2 + 8, C_MUTED, scale=0.55)
