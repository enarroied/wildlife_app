import chime
from hatchet_client import hatchet
from hatchet_sdk import Context
from pydantic import BaseModel

chime.theme("material")

EVENT_KEY = "bear:detected"


class BearEventInput(BaseModel):  # TODO: add more items
    snapshot_path: str
    timestamp: str


event_workflow = hatchet.workflow(
    name="BearEventWorkflow",
    on_events=[EVENT_KEY],  # TODO: add an extra event that triggers bear notification??
    # Maybe like a manual trigger.
    # Eg use case: a camping worker sees a bear and pushes on a device button
    input_validator=BearEventInput,  # Add this!
)


@event_workflow.task()
def notify(input: BearEventInput, ctx: Context):
    """Notifies when a bear is detected"""

    print("=" * 50)
    print("🐻 BEAR DETECTED!")
    print(f"Time: {input.timestamp}")
    print(f"Snapshot: {input.snapshot_path}")
    print("=" * 50)

    chime.success()

    return {"notified": True}


def main():
    worker = hatchet.worker(name="bear-worker", workflows=[event_workflow])
    worker.start()


if __name__ == "__main__":
    main()
