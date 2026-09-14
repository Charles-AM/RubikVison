# RubikVision

RubikVision is a real-time computer-vision project for tracking Rubik's Cube solves. The current Phase 1 foundation accepts a webcam or video file, displays frames in an OpenCV window, and overlays the current frame rate.

## Phase 1 features

- Webcam capture (camera `0` by default)
- Video-file capture
- Live FPS counter
- Graceful shutdown with `Q` or `Esc`
- Unit tests that do not require a physical camera

Cube-face detection is intentionally not part of this phase.

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

Then run the app. A window titled **RubikVision** should show the selected video source with an `FPS` value in the top-left corner. Confirm that `Q` closes the window and releases the source.

If macOS requests camera access, allow it for the terminal or application running Python. If the camera cannot be opened, check that another app is not using it or try `--source 1`.

## Project layout

```text
RubikVison/
├── app/
│   └── app.py          # CLI and OpenCV display loop
├── src/
│   └── camera.py       # Capture and FPS components
├── tests/
│   └── test_camera.py  # Hardware-independent unit tests
├── .gitignore
├── README.md
└── requirements.txt
```

## Roadmap

The next milestone is traditional OpenCV cube-face detection and perspective correction, after Phase 1 has been verified with a real webcam or sample video.

