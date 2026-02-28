from datetime import datetime
from pathlib import Path

from hatchet_client import hatchet


def push_bear_detected(snapshot_path: Path):
    hatchet.event.push(
        "bear:detected",
        {
            "snapshot_path": str(snapshot_path),
            "timestamp": datetime.now().isoformat(),
        },
    )
