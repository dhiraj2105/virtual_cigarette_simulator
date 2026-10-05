# Virtual Cigarette Simulator

A lightweight real-time webcam AR demo using the current MediaPipe Tasks API.

## Technology

- Python
- OpenCV
- MediaPipe 1.0.1
- MediaPipe Hand Landmarker
- MediaPipe Face Landmarker
- CPU inference by default

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

On Windows:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

The first launch downloads the official MediaPipe `.task` model files into `models/`. Later launches use the local copies.

## Controls

- Q / ESC: quit
- R: reset smoke
- F: toggle FPS

## Gesture

1. Show your hand to the camera.
2. Extend your index and middle fingers.
3. Bring their tips close together.
4. The virtual cigarette appears between them.
5. Move the cigarette toward your mouth.
6. The mouth proximity activates the stronger smoke effect.
