import asyncio
import random
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path

import cv2
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from hatchet_client import hatchet
from ultralytics import YOLO

frame_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
broadcast_queue: asyncio.Queue = asyncio.Queue(maxsize=100)  # For multiple clients

# TODO: clean this mess, make config files
SNAPSHOTS_DIR = Path("snapshots")
COOLDOWN_PERIOD = timedelta(seconds=60)
model_dir = "model/yolo11n_openvino_model/"
model = YOLO(model_dir, task="detect")
DETECTION_INTERVAL = 3  # Process every 3rd frame (10 FPS)


# Cooldown state
alert_active = False
last_alert_time = None
last_detection_time = None


# TODO: Separate the streaming simulation, add more videos, remove global variable
URLS = [
    "https://outbound-production.explore.org/stream-production-174/.m3u8",
    # "https://outbound-production.explore.org/stream-production-171/.m3u8",
]


@asynccontextmanager
async def camera(url: str):
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    try:
        yield cap
    finally:
        cap.release()


async def _put_frame_nonblocking(queue: asyncio.Queue, frame):
    """Put frame in queue, dropping oldest frame if full."""
    if queue.full():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
    await queue.put(frame.copy())


async def _distribute_frame(frame):
    resized = cv2.resize(frame, (640, 480))

    await _put_frame_nonblocking(frame_queue, resized)
    await _put_frame_nonblocking(broadcast_queue, resized)


async def _read_camera_stream(cap):
    """Read frames from an open camera capture."""
    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        await _distribute_frame(frame)
        await asyncio.sleep(0)


async def read_camera(urls: list[str]):
    """Continuously read frames from camera sources and distribute to consumers."""
    while True:
        url = random.choice(urls)

        async with camera(url) as cap:
            await _read_camera_stream(cap)


def _save_snapshot(frame):
    """Helper function to save frame as snapshot"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = SNAPSHOTS_DIR / f"bear_{timestamp}.jpg"
    cv2.imwrite(str(filename), frame)
    print(f"🐻 BEAR DETECTED! Saved to {filename}")
    return filename


# Refactored helper functions
def _detect_bears(frame) -> bool:
    """Run YOLO detection and return True if bear found."""
    results = model(frame, stream=True, conf=0.25)

    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            if class_name.lower() == "bear":
                return True

    return False


def _should_trigger_alert() -> bool:
    """Check if cooldown allows new alert."""
    return not alert_active


def _trigger_bear_alert(frame):
    """Save snapshot and push Hatchet event."""
    global alert_active, last_alert_time, last_detection_time
    snapshot_path = _save_snapshot(frame)

    hatchet.event.push(
        "bear:detected",
        {
            "snapshot_path": str(snapshot_path),
            "timestamp": datetime.now().isoformat(),
        },
    )

    alert_active = True
    last_alert_time = datetime.now()
    last_detection_time = datetime.now()


def _reset_cooldown_if_needed(bear_found: bool):
    """Reset alert cooldown when appropriate."""
    global alert_active

    if bear_found:
        return

    if not last_alert_time:
        alert_active = False
        return

    time_since_alert = datetime.now() - last_alert_time
    if time_since_alert > COOLDOWN_PERIOD:
        alert_active = False


async def detect_bear():
    """Run detection loop with frame skipping optimization."""
    frame_counter = 0

    while True:
        frame = await frame_queue.get()
        frame_counter += 1

        # Only detect every 3rd frame (10 FPS optimization)
        if frame_counter % DETECTION_INTERVAL != 0:
            await asyncio.sleep(0)
            continue

        bear_found = _detect_bears(frame)

        if bear_found and _should_trigger_alert():
            _trigger_bear_alert(frame)

        _reset_cooldown_if_needed(bear_found)

        await asyncio.sleep(0)


async def frame_generator():
    """Generate MJPEG frames at reduced FPS for streaming."""
    frame_counter = 0
    STREAM_INTERVAL = 2  # Send every 2nd frame (15 FPS)

    while True:
        frame = await broadcast_queue.get()
        frame_counter += 1

        # Only send every 2nd frame to reduce bandwidth
        if frame_counter % STREAM_INTERVAL != 0:
            continue

        # Encode with lower quality
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])

        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )

        await asyncio.sleep(0.01)


@asynccontextmanager
async def lifespan(app: FastAPI):
    camera_task = asyncio.create_task(read_camera(URLS))
    detector_task = asyncio.create_task(detect_bear())
    yield
    camera_task.cancel()
    detector_task.cancel()


app = FastAPI(lifespan=lifespan)


@app.get("/video_feed")
async def video_feed():
    """Serve MJPEG video stream."""
    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/status")
async def get_status():
    """Return current system status."""
    if last_detection_time:
        seconds_ago = int((datetime.now() - last_detection_time).total_seconds())
        if seconds_ago < 60:
            time_str = f"{seconds_ago} seconds ago"
        else:
            minutes_ago = seconds_ago // 60
            time_str = f"{minutes_ago} minute{'s' if minutes_ago != 1 else ''} ago"

        return {"last_detection": time_str, "has_detections": True}

    return {"last_detection": "No detections yet", "has_detections": False}


# Add landing page endpoint
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve minimal landing page."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Wildlife Detection System</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body class="bg-light">
        <div class="container py-4">
            <div class="text-center mb-4">
                <h1 class="display-5">Wildlife Detection System</h1>
                <p class="lead text-muted">Real-time bear monitoring with Hatchet workflows</p>
            </div>

            <div class="card mb-3">
                <div class="card-body">
                    <h5 class="card-title">Detection Status</h5>
                    <p class="card-text" id="status">Loading...</p>
                </div>
            </div>

            <div class="card">
                <div class="card-body p-0">
                    <img src="/video_feed" class="w-100" style="max-height: 600px; object-fit: contain;" alt="Live camera feed">
                </div>
            </div>
        </div>

        <script>
            async function updateStatus() {
                const response = await fetch('/api/status');
                const data = await response.json();
                document.getElementById('status').textContent = `Last bear detected: ${data.last_detection}`;
            }
            updateStatus();
            setInterval(updateStatus, 5000); // Update every 5 seconds
        </script>
    </body>
    </html>
    """
