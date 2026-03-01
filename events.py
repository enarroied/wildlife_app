from datetime import datetime
from pathlib import Path

from hatchet_client import hatchet


def push_bear_detected(snapshot_path: Path) -> None:
    """Push a bear:detected event to Hatchet."""
    hatchet.event.push(
        "bear:detected",
        {
            "snapshot_path": str(snapshot_path),
            "timestamp": datetime.now().isoformat(),
        },
    )


def push_manual_trigger(location: str | None = None) -> None:
    """Push a manual bear sighting event (e.g. from a field device button).

    Args:
        location: Optional human-readable location string from the device.
    """
    hatchet.event.push(
        "bear:manual",
        {
            "timestamp": datetime.now().isoformat(),
            "location": location or "unknown",
        },
    )
