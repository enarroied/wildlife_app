"""Hatchet worker — listens for bear events and runs notification workflows."""

import chime
from hatchet_client import hatchet
from hatchet_sdk import Context
from pydantic import BaseModel

chime.theme("material")

DETECTED_EVENT = "bear:detected"
MANUAL_EVENT = "bear:manual"


# Input models


class BearDetectedInput(BaseModel):
    snapshot_path: str
    timestamp: str


class BearManualInput(BaseModel):
    timestamp: str
    location: str = "unknown"


# Workflows

detection_workflow = hatchet.workflow(
    name="BearDetectedWorkflow",
    on_events=[DETECTED_EVENT],
    input_validator=BearDetectedInput,
)

manual_workflow = hatchet.workflow(
    name="BearManualWorkflow",
    on_events=[MANUAL_EVENT],
    input_validator=BearManualInput,
)


@detection_workflow.task()
def notify_detection(input: BearDetectedInput, ctx: Context):
    """Notify when the camera detects a bear automatically."""
    print("=" * 50)
    print("🐻 BEAR DETECTED (camera)!")
    print(f"Time:     {input.timestamp}")
    print(f"Snapshot: {input.snapshot_path}")
    print("=" * 50)
    chime.success()
    return {"notified": True}


@manual_workflow.task()
def notify_manual(input: BearManualInput, ctx: Context):
    """Notify when a field worker manually reports a bear sighting."""
    print("=" * 50)
    print("🐻 BEAR REPORTED (manual)!")
    print(f"Time:     {input.timestamp}")
    print(f"Location: {input.location}")
    print("=" * 50)
    chime.success()
    return {"notified": True}


# Entry point


def main():
    worker = hatchet.worker(
        name="bear-worker",
        workflows=[detection_workflow, manual_workflow],
    )
    worker.start()


if __name__ == "__main__":
    main()
