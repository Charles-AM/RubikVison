# RubikVision

RubikVision is a real-time computer-vision project for tracking Rubik's Cube solves. The current prototype accepts a webcam or video file, displays frames in an OpenCV window, overlays the current frame rate, and locates a square-like visible cube face.

## Current features

- Webcam capture (camera `0` by default)
- Video-file capture
- Live FPS counter
- Color-aware OpenCV face detection using edges and quadrilateral contours
- Live detection outline and status
- Perspective-corrected square face preview
- Nine ordered sticker regions with border-safe center sampling
- Baseline six-color HSV classification with per-sticker confidence
- Warm-color separation using a camera-stable green/red channel ratio
- Rolling five-frame color vote with automatic face-change reset
- Persistent camera/lighting calibration from six center stickers
- Standard nine-character face state such as `RRWBGGYRB`
- Visible-face consistency measured against the center sticker
- Manual solve timer with ready, running, and stopped states
- Graceful shutdown with `Q` or `Esc`
- Unit tests that do not require a physical camera

Solved-state confirmation is intentionally deferred to the next checkpoint.

## Setup

Python 3.11 is the tested runtime. Python 3.14 is currently avoided because the
macOS NumPy/OpenCV combination can stall during import.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Run

Open the default webcam:

```bash
python app/app.py
```

Open another camera:

```bash
python app/app.py --source 1
```

Play a video file:

```bash
python app/app.py --source path/to/video.mp4
```

Show live HSV values instead of confidence for color calibration:

```bash
python app/app.py --debug-colors
```

During a normal run, type `S` in Terminal and press Enter to start or stop the
solve timer. Type `X` and press Enter to reset it.

Calibrate once for the current camera and lighting:

```bash
python app/app.py --calibrate-colors
```

Show one face at a time, then type its key in Terminal and press Enter: `W` for
white, `Y` for yellow, `R` for red, `O` for orange, `B` for blue, and `G` for
green. The center sticker is saved as that color's reference. Camera-window key
shortcuts also remain available. After all six are captured, later runs load
`config/color_calibration.json` automatically. Repeat the command and enter a
color key again to replace that color's reference when room lighting changes
substantially.

Wait for the `Captured <color> (N/6).` confirmation before moving to the next
face. If detection drops at the moment a command is entered, the app keeps that
command pending and captures it automatically when the face is detected again.

Press `Q` or `Esc` while the video window is focused to quit. `Ctrl+C` in the
terminal also stops the app cleanly. The app exits automatically when a video
reaches its final frame. On macOS, check behind the Terminal window if the
OpenCV window does not immediately appear in front.

## Verify Phase 1

Run the automated tests:

```bash
python -m pytest
```

Then run the app. A window titled **RubikVision** should show the selected video source with an `FPS` value in the top-left corner. When a sufficiently large cube face is visible, a green quadrilateral and `Cube face: detected` status should appear. A second window titled **RubikVision - Normalized Face** displays the perspective-corrected square face with nine numbered sticker regions. The smaller green boxes indicate the sampled pixels, and each cell displays its predicted color notation and confidence. Confirm that `Q` closes the windows and releases the source.

If macOS requests camera access, allow it for the terminal or application running Python. If the camera cannot be opened, check that another app is not using it or try `--source 1`.

### macOS `Library not loaded` error

Some macOS `Documents` folders are managed by a file-sync provider. Native
OpenCV libraries can fail to load from a virtual environment stored there. If
the error mentions `cv2.abi3.so`, `libavfilter`, or a failed `mmap`, keep the
environment outside `Documents` and symlink it into the project:

```bash
mkdir -p ~/.virtualenvs
python3.11 -m venv ~/.virtualenvs/rubikvision
~/.virtualenvs/rubikvision/bin/python -m pip install -r requirements.txt
mv .venv .venv-backup
ln -s ~/.virtualenvs/rubikvision .venv
source .venv/bin/activate
```

## Project layout

```text
RubikVison/
├── app/
│   └── app.py          # CLI and OpenCV display loop
├── src/
│   ├── camera.py       # Capture and FPS components
│   ├── color_classifier.py # Baseline HSV sticker classifier
│   ├── cube_state.py   # Standard face-state representation
│   ├── face_detector.py # Edge/contour cube-face detector
│   ├── perspective.py  # Square perspective transform
│   ├── progress.py     # Visible-face consistency metrics
│   ├── sticker_detector.py # 3x3 sticker extraction
│   ├── timer.py        # Manual solve timer
│   └── tracker.py      # Temporal color smoothing
├── config/
│   └── color_calibration.json # Local, generated calibration
├── tests/
│   ├── test_camera.py  # Hardware-independent unit tests
│   ├── test_color_classifier.py
│   ├── test_cube_state.py
│   ├── test_face_detector.py
│   ├── test_perspective.py
│   ├── test_progress.py
│   ├── test_sticker_detector.py
│   ├── test_timer.py
│   └── test_tracker.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Roadmap

The next milestone is stable solved-state detection.
