"""FastAPI route definitions.

Registered on the app in main.py via app.include_router().
Routes read shared runtime state from the state module.
"""

from datetime import datetime
from typing import Optional

import cv2
import state
from events import push_manual_trigger
from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

router = APIRouter()


async def frame_generator(broadcast_queue, fps_interval: int, jpeg_quality: int):
    """Generate MJPEG frames at reduced FPS for streaming."""
    frame_counter = 0

    while True:
        frame = await broadcast_queue.get()
        frame_counter += 1

        # Only send every Nth frame to reduce bandwidth
        if frame_counter % fps_interval != 0:
            continue

        _, buffer = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        )

        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )

        import asyncio

        await asyncio.sleep(0.01)


def create_router(broadcast_queue, fps_interval: int, jpeg_quality: int) -> APIRouter:
    """Build the router with queue/config dependencies injected.

    Call this once in main.py after config and queues are initialised.
    """

    @router.get("/api/status")
    async def get_status():
        """Return current system status."""
        if state.last_detection_time:
            seconds_ago = int(
                (datetime.now() - state.last_detection_time).total_seconds()
            )
            if seconds_ago < 60:
                time_str = f"{seconds_ago} seconds ago"
            else:
                minutes_ago = seconds_ago // 60
                time_str = f"{minutes_ago} minute{'s' if minutes_ago != 1 else ''} ago"

            return {"last_detection": time_str, "has_detections": True}

        return {"last_detection": "No detections yet", "has_detections": False}

    @router.get("/video_feed")
    async def video_feed():
        """Serve MJPEG video stream."""
        return StreamingResponse(
            frame_generator(broadcast_queue, fps_interval, jpeg_quality),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    class ManualAlertRequest(BaseModel):
        location: Optional[str] = None

    @router.post("/api/manual_alert")
    async def manual_alert(body: ManualAlertRequest = ManualAlertRequest()):
        """Trigger a manual bear sighting event from the UI."""
        push_manual_trigger(location=body.location)
        return JSONResponse({"ok": True, "message": "Manual alert triggered"})

    @router.get("/", response_class=FileResponse)
    async def dashboard():
        """Serve landing page."""
        return FileResponse("templates/index.html")

    return router
