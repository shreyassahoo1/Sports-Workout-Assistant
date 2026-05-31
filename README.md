# FORMScope — AI-Based Workout Form Correction System

**RV College of Engineering | Cluster: CS | ACY 2025-26**  
**Team:** Sanath Reddy · Shreyas Sahoo · Samarjeet Sujeet Bhonsle (ISE)

---

## What it does

FORMScope uses your webcam (or a pre-recorded video) and **MediaPipe Pose** to detect 33 body keypoints in real time. It calculates joint angles every frame, compares them against biomechanically correct ranges, and gives you:

- **Instant visual feedback** on-screen (green = good, red = fix it)
- **Audio cues via TTS** ("Hips sagging — engage your core.")
- **Rep counting** with a progress ring
- **Form score** broken down by Spine, Depth, Knee Tracking, Hip Position
- **Session summary** printed at the end

Supported exercises: **Squat · Push-up · Plank · Lunge**

---

## Project Structure

```
formscope/
├── main.py                  ← Entry point (run this)
├── requirements.txt
├── README.md
│
├── core/
│   └── session.py           ← Main loop: video capture, MediaPipe, HUD, keys
│
├── exercises/
│   ├── base.py              ← Abstract BaseExercise + ExerciseState
│   └── exercises.py         ← Squat, PushUp, Plank, Lunge + EXERCISE_REGISTRY
│
└── utils/
    ├── angles.py            ← Joint angle math (angle_between, spine_angle, etc.)
    ├── hud.py               ← All OpenCV drawing helpers (HUD, banners, gauges)
    └── tts.py               ← Non-blocking TTS via daemon thread + priority queue
```

---

## Setup

### 1. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Linux only:** if pyttsx3 fails, install `espeak`:
> ```bash
> sudo apt-get install espeak
> ```

> **macOS only:** pyttsx3 uses the built-in `NSSpeechSynthesizer` — no extras needed.

---

## Usage

### Webcam (default)

```bash
python main.py
```

### Webcam with a specific exercise

```bash
python main.py --exercise pushup
python main.py --exercise plank
python main.py --exercise lunge
```

### Video file input

```bash
python main.py --source path/to/video.mp4
```

### Disable TTS (text only)

```bash
python main.py --no-tts
```

### Save annotated output

```bash
python main.py --save output.mp4
```

### All options combined

```bash
python main.py --source workout.mp4 --exercise squat --save result.mp4
```

---

## Keyboard Controls

| Key | Action |
|-----|--------|
| `SPACE` | Pause / Resume |
| `1` | Switch to Squat |
| `2` | Switch to Push-up |
| `3` | Switch to Plank |
| `4` | Switch to Lunge |
| `R` | Reset current rep count |
| `S` | Save screenshot (PNG) |
| `Q` / `ESC` | Quit |

---

## How It Works

### Pose Estimation — MediaPipe
MediaPipe Pose returns 33 normalised 3D landmarks at ~30 fps. FORMScope uses 16 of these (shoulders, elbows, wrists, hips, knees, ankles, heels).

### Angle Computation (`utils/angles.py`)
Every landmark is a point `(x, y)` in [0,1] space. The angle at joint **B** between vectors **BA** and **BC** is:

```
θ = arccos( (BA · BC) / (|BA| × |BC|) )
```

A 5-frame **exponential moving average** smooths noisy measurements before they reach the form-checker.

### Form Checking (`exercises/exercises.py`)
Each exercise defines acceptable angle ranges per joint, e.g. for Squat:

| Joint | Good Range |
|-------|-----------|
| Knee angle | 60° – 175° |
| Hip angle | 65° – 175° |
| Spine lean | 0° – 15° |

The form-checker scores each metric 0–100 and returns a weighted average as the **Form Score**.

### Rep Counting
A simple 2-state machine:
- **`up`** → when the primary angle crosses below the threshold → `down`
- **`down`** → when the angle recovers above return threshold → `up` → **rep counted**

Plank uses a time-based tick (1 rep = 1 second held with acceptable form).

### TTS (`utils/tts.py`)
pyttsx3 runs on a **daemon thread** with a priority queue. The video loop enqueues messages non-blockingly — TTS never stalls frame rendering. A 4-second cooldown prevents the same phrase from repeating.

---

## Adding a New Exercise

1. Subclass `BaseExercise` in `exercises/exercises.py`
2. Define `NAME`, `GOAL_REPS`, `JOINTS`, `ANGLE_RANGES`, `PRIMARY_CONNECTIONS`, `PRIMARY_JOINTS`
3. Implement `_compute_angles`, `_check_form`, `_check_rep`
4. Add it to `EXERCISE_REGISTRY` and `EXERCISE_KEYS`

That's it — the session, HUD, and TTS will all pick it up automatically.

---

## Troubleshooting

**`ModuleNotFoundError: mediapipe`**  
Run `pip install mediapipe` or reinstall via `pip install -r requirements.txt`

**Camera not opening**  
Try `--source 1` (some machines index the webcam as 1, not 0).

**pyttsx3 silent on Linux**  
Install espeak: `sudo apt-get install espeak libespeak-dev`

**Slow performance**  
Set `model_complexity=0` in `core/session.py` → `self._pose_config` for the lite model.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| mediapipe | ≥ 0.10.9 | Pose estimation |
| opencv-python | ≥ 4.9.0 | Video capture + rendering |
| pyttsx3 | ≥ 2.90 | Offline TTS |
| numpy | ≥ 1.24.0 | Array math |
