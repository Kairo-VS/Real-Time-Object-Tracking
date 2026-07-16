import cv2
import os
import numpy as np
import json
import base64
import threading
import time
from flask import Flask, Response, request, jsonify
from flask_socketio import SocketIO, emit
from collections import defaultdict, deque

app = Flask(__name__)
app.config['SECRET_KEY'] = 'tracking-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# In-memory store of tracked objects
# object_id -> {label, positions: deque, last_seen, color}
tracked_objects = {}
object_counter = 0
object_lock = threading.Lock()

# Color palette for objects
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0)
]

# Simple tracker using centroid matching
def update_tracker(detections, frame_shape):
    """Match detections to existing objects via centroid distance."""
    global object_counter
    h, w = frame_shape[:2]
    centers = []
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        centers.append((cx, cy, det['label'], det['confidence']))

    with object_lock:
        # Match existing objects to new centers (greedy nearest)
        used = set()
        for oid, obj in tracked_objects.items():
            if not obj['positions']:
                continue
            last = obj['positions'][-1]
            best_idx, best_dist = None, float('inf')
            for i, (cx, cy, label, conf) in enumerate(centers):
                if i in used:
                    continue
                dist = ((cx - last[0]) ** 2 + (cy - last[1]) ** 2) ** 0.5
                if dist < 80 and dist < best_dist:
                    best_idx, best_dist = i, dist
            if best_idx is not None:
                cx, cy, label, conf = centers[best_idx]
                obj['positions'].append((cx, cy))
                obj['last_seen'] = time.time()
                obj['label'] = label
                obj['confidence'] = conf
                used.add(best_idx)

        # Create new objects for unmatched centers
        for i, (cx, cy, label, conf) in enumerate(centers):
            if i in used:
                continue
            object_counter += 1
            oid = f"obj_{object_counter}"
            tracked_objects[oid] = {
                'label': label,
                'confidence': conf,
                'positions': deque([(cx, cy)], maxlen=50),
                'last_seen': time.time(),
                'color': COLORS[object_counter % len(COLORS)]
            }

        # Remove stale objects
        now = time.time()
        stale = [oid for oid, obj in tracked_objects.items() if now - obj['last_seen'] > 5]
        for oid in stale:
            del tracked_objects[oid]

        # Build snapshot
        snapshot = []
        for oid, obj in tracked_objects.items():
            pos = obj['positions'][-1]
            snapshot.append({
                'id': oid,
                'label': obj['label'],
                'confidence': round(obj['confidence'], 2),
                'x': round(pos[0] / w, 4),
                'y': round(pos[1] / h, 4),
                'trail': [{'x': p[0] / w, 'y': p[1] / h} for p in obj['positions']],
                'color': obj['color']
            })
    return snapshot


# YOLO model (loaded lazily on first use so import is fast)
_yolo_model = None
# Prefer a local weights file next to this script so we never hit the network
# to download. Falls back to the env var YOLO_MODEL, then 'yolov8n.pt'.
_LOCAL_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov8n.pt')
if os.environ.get('YOLO_MODEL'):
    YOLO_MODEL_NAME = os.environ['YOLO_MODEL']
elif os.path.exists(_LOCAL_WEIGHTS):
    YOLO_MODEL_NAME = _LOCAL_WEIGHTS
else:
    YOLO_MODEL_NAME = 'yolov8n.pt'
YOLO_CONF_THRESHOLD = float(os.environ.get('YOLO_CONF', '0.5'))
# Optional inference resize for performance on large frames (e.g. 4K video).
# Set to a width in pixels; the frame is downscaled before inference and
# bounding boxes are scaled back to the original resolution.
DETECT_WIDTH = int(os.environ.get('DETECT_WIDTH', '0')) or None


def get_yolo_model():
    """Lazily load the YOLO model (downloads weights on first call)."""
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        print(f"[INFO] Loading YOLO model: {YOLO_MODEL_NAME}")
        _yolo_model = YOLO(YOLO_MODEL_NAME)
    return _yolo_model


def detect_objects(frame):
    """Run YOLO object detection on a frame.

    Returns list of dicts: {bbox: [x1,y1,x2,y2], label, confidence}
    Bounding boxes are in pixel coordinates relative to the original frame.
    """
    model = get_yolo_model()

    # Optionally downscale large frames for faster inference
    scale = 1.0
    if DETECT_WIDTH and frame.shape[1] > DETECT_WIDTH:
        scale = DETECT_WIDTH / frame.shape[1]
        h = int(frame.shape[0] * scale)
        infer_frame = cv2.resize(frame, (DETECT_WIDTH, h))
    else:
        infer_frame = frame

    # Ultralytics accepts BGR numpy arrays directly
    results = model(infer_frame, conf=YOLO_CONF_THRESHOLD, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            detections.append({
                'bbox': [int(x1 / scale), int(y1 / scale),
                         int(x2 / scale), int(y2 / scale)],
                'label': label,
                'confidence': round(conf, 3)
            })
    return detections


def capture_loop(video_source=0):
    """Capture from a video source and emit tracking data via WebSocket.

    video_source can be:
      * int  -> camera index (e.g. 0 for default webcam)
      * str  -> path to a video file (e.g. "C:/videos/feed.mp4")
    If the source cannot be opened, falls back to synthetic frames.
    """
    # Resolve source: env var VIDEO_SOURCE overrides default
    env_src = os.environ.get('VIDEO_SOURCE')
    if env_src is not None:
        video_source = int(env_src) if env_src.isdigit() else env_src

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[WARN] Could not open video source '{video_source}'. Using synthetic frames.")
        cap = None
    print(f"[INFO] Tracking video source: {video_source}")
    while True:
        if cap is not None:
            ret, frame = cap.read()
            if not ret:
                # End of video file or camera disconnected -> restart/loop
                if isinstance(video_source, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video file
                    continue
                else:
                    time.sleep(0.1)
                    continue
        else:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detect_objects(frame)
        snapshot = update_tracker(detections, frame.shape)
        socketio.emit('tracking_update', {'objects': snapshot, 'timestamp': time.time()})
        time.sleep(0.1)  # ~10 fps updates


@app.route('/')
def index():
    return jsonify({'status': 'realtime tracking server running'})


@app.route('/objects')
def get_objects():
    with object_lock:
        return jsonify({'count': len(tracked_objects), 'objects': list(tracked_objects.keys())})


@socketio.on('connect')
def on_connect():
    print('[INFO] Client connected')
    emit('status', {'msg': 'connected'})


@socketio.on('disconnect')
def on_disconnect():
    print('[INFO] Client disconnected')


if __name__ == '__main__':
    # Set VIDEO_SOURCE env var to a camera index (0) or video file path before running.
    # Example (PowerShell):  $env:VIDEO_SOURCE="C:/videos/feed.mp4"; python backend/app.py
    threading.Thread(target=capture_loop, daemon=True).start()
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
