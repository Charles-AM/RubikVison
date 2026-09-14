# RubikVision

RubikVision is a real-time computer-vision project for tracking Rubik's Cube solves. The current prototype accepts a webcam or video file, displays frames in an OpenCV window, overlays the current frame rate, and locates a square-like visible cube face.

## Current features

- Webcam capture (camera `0` by default)
- Video-file capture
- Live FPS counter
- Traditional OpenCV face detection using edges and quadrilateral contours
- Live detection outline and status
- Graceful shutdown with `Q` or `Esc`
- Unit tests that do not require a physical camera

Perspective correction and sticker color classification are intentionally deferred to later checkpoints.

## Setup

Python 3.11 or newer is recommended.

```bash
python3 -m venv .venv
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

Press `Q` or `Esc` while the video window is focused to quit. The app also exits cleanly when a video reaches its final frame.

## Verify Phase 1

Run the automated tests:

```bash
python -m pytest
```

Then run the app. A window titled **RubikVision** should show the selected video source with an `FPS` value in the top-left corner. When a sufficiently large cube face is visible, a green quadrilateral and `Cube face: detected` status should appear. Confirm that `Q` closes the window and releases the source.

If macOS requests camera access, allow it for the terminal or application running Python. If the camera cannot be opened, check that another app is not using it or try `--source 1`.

## Project layout

```text
RubikVison/
├── app/
│   └── app.py          # CLI and OpenCV display loop
├── src/
│   ├── camera.py       # Capture and FPS components
│   └── face_detector.py # Edge/contour cube-face detector
├── tests/
│   ├── test_camera.py  # Hardware-independent unit tests
│   └── test_face_detector.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Roadmap

The next milestone is perspective correction, followed by 3×3 sticker-region extraction.
