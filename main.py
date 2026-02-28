import asyncio
import random
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import cv2
from config import load_config
from events import push_bear_detected
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from ultralytics import YOLO

# Load configuration
config = load_config()

# Initialize queues with config
frame_queue = asyncio.Queue(maxsize=config.frame_queue_size)
broadcast_queue = asyncio.Queue(maxsize=config.broadcast_queue_size)

# Load YOLO model
model = YOLO(config.model_dir, task="detect")

# Runtime state (these stay as globals)
alert_active = False
last_alert_time = None
last_detection_time = None


@asynccontextmanager
async def camera(url: str):
    """Context manager for video capture with cleanup."""
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
    """Send frame to all consumers (detector and streaming clients)."""
    # Resize frame to reduce memory and bandwidth
    resized = cv2.resize(frame, config.stream_resolution)

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


async def read_camera():
    """Continuously read frames from camera sources and distribute to consumers."""
    while True:
        url = random.choice(config.stream_urls)

        async with camera(url) as cap:
            await _read_camera_stream(cap)


def _save_snapshot(frame) -> Path:
    """Save detection frame as JPEG with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = config.snapshots_dir / f"bear_{timestamp}.jpg"
    cv2.imwrite(str(filename), frame)
    print(f"BEAR DETECTED! Saved to {filename}")
    return filename


def _detect_bears(frame) -> bool:
    """Run YOLO detection and return True if bear found."""
    results = model(frame, stream=True, conf=config.detection_confidence)

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

    push_bear_detected(snapshot_path)

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
    if time_since_alert > config.cooldown_period:
        alert_active = False


async def detect_bear():
    """Run detection loop with frame skipping optimization."""
    frame_counter = 0

    while True:
        frame = await frame_queue.get()
        frame_counter += 1

        # Only detect every Nth frame (configured interval)
        if frame_counter % config.detection_interval != 0:
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

    while True:
        frame = await broadcast_queue.get()
        frame_counter += 1

        # Only send every Nth frame to reduce bandwidth
        if frame_counter % config.stream_fps_interval != 0:
            continue

        # Encode with configured quality
        _, buffer = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, config.stream_jpeg_quality]
        )

        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )

        await asyncio.sleep(0.01)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage background tasks during application lifecycle."""
    camera_task = asyncio.create_task(read_camera())
    detector_task = asyncio.create_task(detect_bear())
    yield
    camera_task.cancel()
    detector_task.cancel()


app = FastAPI(lifespan=lifespan)


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


@app.get("/video_feed")
async def video_feed():
    """Serve MJPEG video stream."""
    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/", response_class=FileResponse)
async def dashboard():
    """Serve landing page."""
    return FileResponse("templates/index.html")
