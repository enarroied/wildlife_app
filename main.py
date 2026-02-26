import asyncio
import random
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path

import cv2
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
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
    """Send frame to all consumers (detector and streaming clients)."""
    await _put_frame_nonblocking(frame_queue, frame)
    await _put_frame_nonblocking(broadcast_queue, frame)


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
    global alert_active, last_alert_time

    snapshot_path = _save_snapshot(frame)

    hatchet.event.push(  # TODO: Add metadata, maybe other animals
        "bear:detected",
        {
            "snapshot_path": str(snapshot_path),
            "timestamp": datetime.now().isoformat(),
        },
    )

    alert_active = True
    last_alert_time = datetime.now()


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
    """Generates JPEG frames for ONE client from broadcast queue"""
    while True:
        frame = await broadcast_queue.get()

        _, buffer = cv2.imencode(".jpg", frame)
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
    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
