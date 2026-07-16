
#created by Kairo-VS



# Real-Time Object Tracking
A real-time multi-object tracking server. [YOLO](https://github.com/ultralytics/ultralytics)
(via `ultralytics`) detects objects in a video stream, a lightweight centroid tracker
assigns stable IDs and movement trails, and the results are streamed to a web dashboard
over WebSockets (Flask-SocketIO).

## What it does

- Runs object detection (person, car, bicycle, ...) on each video frame with YOLO.
- Tracks each object across frames using centroid matching, giving every object a
  stable ID and a trail of recent positions.
- Emits live tracking data to a browser dashboard that draws the objects and their
  trails in real time.

## Project structure

```
realtime_tracking/
├── backend/
│   ├── app.py            # Flask + SocketIO server (detection + tracking)
│   └── requirements.txt  # Python dependencies
├── frontend/
│   └── index.html        # Dashboard (connects to the server via WebSocket)
├── .gitignore
└── README.md
```

## Requirements

**Hardware**
- A machine with Python 3.9+ (tested on Python 3.14).
- A webcam (optional) OR a video file to analyze.
- For good performance on 4K video, a CUDA-capable GPU is recommended but not
  required (CPU works, just slower — use `DETECT_WIDTH` to speed it up).

**Software**
- Python 3.9 or newer.
- The packages listed in `backend/requirements.txt`:
  - `Flask`, `Flask-SocketIO` — web server + real-time messaging
  - `opencv-python` — video capture / frame handling
  - `numpy` — numerical operations
  - `ultralytics` + `torch` / `torchvision` — YOLO detection

> The YOLO weights (`yolov8n.pt`) are downloaded automatically the first time you
> run the server, so you do **not** need to download them manually.

## Setup

```bash
# 1. (Recommended) create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 2. Install dependencies
pip install -r backend/requirements.txt
```

## Running with your own video

Set the `VIDEO_SOURCE` environment variable to the path of any video file
(`.mp4`, `.avi`, `.mov`, etc.). The video will loop automatically when it ends.

**Windows (PowerShell):**
```powershell
$env:VIDEO_SOURCE="C:\path\to\your\video.mp4"
$env:DETECT_WIDTH="1280"      # optional: downscale large frames for speed
python backend/app.py
```

**macOS / Linux (bash):**
```bash
export VIDEO_SOURCE="/path/to/your/video.mp4"
export DETECT_WIDTH="1280"
python backend/app.py
```

Then open `frontend/index.html` in your browser to watch the live tracking.

## Running with a webcam

Set `VIDEO_SOURCE` to your camera's index (usually `0` for the default webcam):

```powershell
$env:VIDEO_SOURCE="0"
python backend/app.py
```

## Viewing the dashboard

The server listens on port `5000`. After starting it, open:

```
frontend/index.html
```

The page connects to `http://localhost:5000` and shows:
- A canvas with tracked objects (colored dots) and their movement trails.
- A side panel listing each object's ID, label, confidence, and position.

You can also confirm the server is up by visiting `http://localhost:5000/` in a
browser — it returns a small JSON status message.

## Configuration (environment variables)

All configuration is done through environment variables, so you never have to edit
the code to use your own inputs.

| Variable         | Default        | Description                                                                 |
|------------------|----------------|-----------------------------------------------------------------------------|
| `VIDEO_SOURCE`   | `0` (webcam)   | Camera index (e.g. `0`) or a path to a video file.                          |
| `YOLO_MODEL`     | `yolov8n.pt`   | Ultrytics model name (e.g. `yolov8s.pt`, `yolov8m.pt`) or a path to local weights. Larger models are more accurate but slower. |
| `YOLO_CONF`      | `0.5`          | Detection confidence threshold (0–1). Raise to reduce false positives.     |
| `DETECT_WIDTH`   | _(unset)_      | If set, frames are downscaled to this pixel width before inference; bounding boxes are scaled back to full resolution. Great for 4K video. |

### Example: higher accuracy on a GPU
```powershell
$env:VIDEO_SOURCE="C:\videos\street.mp4"
$env:YOLO_MODEL="yolov8m.pt"
$env:YOLO_CONF="0.6"
python backend/app.py
```

## How it works (brief)

1. `capture_loop` reads frames from the video source (or webcam) in a background thread.
2. `detect_objects` runs YOLO on each frame and returns bounding boxes + labels.
3. `update_tracker` matches new detections to existing tracked objects by nearest
   centroid, creates new tracks for unmatched detections, and drops tracks that
   haven't been seen for 5 seconds.
4. The resulting snapshot is emitted over the WebSocket as a `tracking_update` event,
   which the frontend renders.

## Notes & caveats

- The server uses Flask's built-in development server on `0.0.0.0:5000`. Do **not**
  expose it publicly without a proper WSGI server and security hardening.
- `SECRET_KEY` in `backend/app.py` is a development placeholder — set a real secret
  before any public deployment.
- Large files (model weights `*.pt`, videos, logs) are excluded via `.gitignore`
  and are not committed to the repository.