import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import cv2
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from ultralytics import YOLO

frame_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
broadcast_queue: asyncio.Queue = asyncio.Queue(maxsize=100)  # For multiple clients

SNAPSHOTS_DIR = Path("snapshots")


@asynccontextmanager
async def camera(url: str):
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    try:
        yield cap
    finally:
        cap.release()


model_dir = "model/yolo11n_openvino_model/"
model = YOLO(model_dir, task="detect")


async def camera_reader(urls: list[str]):
    """Reads frames from cameras and broadcasts to all consumers"""
    import random

    while True:
        url = random.choice(urls)

        async with camera(url) as cap:
            while cap.isOpened():
                success, frame = cap.read()

                if not success:
                    break

                # Put in frame_queue for detector
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                await frame_queue.put(frame.copy())

                # Broadcast to all streaming clients
                if broadcast_queue.full():
                    try:
                        broadcast_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                await broadcast_queue.put(frame.copy())

                await asyncio.sleep(0)


def _save_snapshot(frame):
    """Helper function to save frame as snapshot"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = SNAPSHOTS_DIR / f"bear_{timestamp}.jpg"
    cv2.imwrite(str(filename), frame)
    print(f"🐻 BEAR DETECTED! Saved to {filename}")
    return filename


async def bear_detector():
    """Runs detection in background, saves snapshots when bears detected"""
    while True:
        frame = await frame_queue.get()

        results = model(frame, stream=True, conf=0.25)
        for r in results:
            boxes = r.boxes

            # Check if any detected object is a bear (class 21 in COCO)
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                if class_name.lower() == "bear":
                    _save_snapshot(frame)
                    break  # Only save once per frame even if multiple bears

        await asyncio.sleep(0)  # Yield control to other async tasks


async def frame_generator():
    """Generates JPEG frames for ONE client from broadcast queue"""
    while True:
        frame = await broadcast_queue.get()

        _, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )

        await asyncio.sleep(0.01)


URLS = [
    "https://outbound-production.explore.org/stream-production-174/.m3u8",
    # "https://outbound-production.explore.org/stream-production-171/.m3u8",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    camera_task = asyncio.create_task(camera_reader(URLS))
    detector_task = asyncio.create_task(bear_detector())
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
