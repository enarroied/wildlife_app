"""Wildlife detection app"""

import asyncio
import random
from contextlib import asynccontextmanager
from contextlib import asynccontextmanager as _acm
from datetime import datetime
from pathlib import Path

import cv2
import state
from config import load_config
from events import push_bear_detected
from fastapi import FastAPI
from routes import create_router
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

config = load_config()

frame_queue: asyncio.Queue = asyncio.Queue(maxsize=config.frame_queue_size)
broadcast_queue: asyncio.Queue = asyncio.Queue(maxsize=config.broadcast_queue_size)

model = YOLO(config.model_dir, task="detect")

# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------


@_acm
async def camera(url: str):
    """Context manager for video capture with cleanup."""
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    try:
        yield cap
    finally:
        cap.release()


async def _put_frame_nonblocking(queue: asyncio.Queue, frame) -> None:
    """Put frame in queue, dropping oldest frame if full."""
    if queue.full():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
    await queue.put(frame.copy())


async def _distribute_frame(frame) -> None:
    """Send frame to all consumers (detector and streaming clients)."""
    resized = cv2.resize(frame, config.stream_resolution)
    await _put_frame_nonblocking(frame_queue, resized)
    await _put_frame_nonblocking(broadcast_queue, resized)


async def _read_camera_stream(cap) -> None:
    """Read frames from an open camera capture."""
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        await _distribute_frame(frame)
        await asyncio.sleep(0)


async def read_camera() -> None:
    """Continuously read frames from camera sources and distribute to consumers."""
    while True:
        url = random.choice(config.stream_urls)
        async with camera(url) as cap:
            await _read_camera_stream(cap)


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def _save_snapshot(frame) -> Path:
    """Save detection frame as JPEG with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = config.snapshots_dir / f"bear_{timestamp}.jpg"
    cv2.imwrite(str(filename), frame)
    print(f"BEAR DETECTED! Saved to {filename}")
    return filename


def _detect_bears(frame) -> bool:
    """Run YOLO detection and return True if a bear is found."""
    results = model(frame, stream=True, conf=config.detection_confidence)
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            if model.names[class_id].lower() == "bear":
                return True
    return False


def _trigger_bear_alert(frame) -> None:
    """Save snapshot and publish detection event."""
    snapshot_path = _save_snapshot(frame)
    push_bear_detected(snapshot_path)

    state.alert_active = True
    state.last_alert_time = datetime.now()
    state.last_detection_time = datetime.now()


def _reset_cooldown_if_needed(bear_found: bool) -> None:
    """Reset alert cooldown when no bear is present and cooldown has elapsed."""
    if bear_found:
        return

    if not state.last_alert_time:
        state.alert_active = False
        return

    if datetime.now() - state.last_alert_time > config.cooldown_period:
        state.alert_active = False


async def detect_bear() -> None:
    """Run detection loop with frame-skipping optimisation."""
    frame_counter = 0

    while True:
        frame = await frame_queue.get()
        frame_counter += 1

        if frame_counter % config.detection_interval != 0:
            await asyncio.sleep(0)
            continue

        bear_found = _detect_bears(frame)

        if bear_found and not state.alert_active:
            _trigger_bear_alert(frame)

        _reset_cooldown_if_needed(bear_found)
        await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage background tasks during application lifecycle."""
    camera_task = asyncio.create_task(read_camera())
    detector_task = asyncio.create_task(detect_bear())
    yield
    camera_task.cancel()
    detector_task.cancel()


app = FastAPI(lifespan=lifespan)
app.include_router(
    create_router(
        broadcast_queue, config.stream_fps_interval, config.stream_jpeg_quality
    )
)
