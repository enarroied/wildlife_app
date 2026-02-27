# config.py
"""Configuration management for wildlife detection system."""

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import yaml


@dataclass
class Config:
    """Application configuration loaded from config.yaml"""

    snapshots_dir: Path
    model_dir: str
    detection_confidence: float
    detection_interval: int
    cooldown_period: timedelta
    stream_urls: list[str]
    stream_fps_interval: int
    stream_jpeg_quality: int
    stream_resolution: tuple[int, int]
    frame_queue_size: int
    broadcast_queue_size: int


def load_config() -> Config:
    """Load configuration from config.yaml"""
    with open("config.yaml") as f:
        data = yaml.safe_load(f)

    return Config(
        snapshots_dir=Path(data["snapshots"]["directory"]),
        model_dir=data["detection"]["model_dir"],
        detection_confidence=data["detection"]["confidence"],
        detection_interval=data["detection"]["interval"],
        cooldown_period=timedelta(seconds=data["detection"]["cooldown_seconds"]),
        stream_urls=data["streaming"]["urls"],
        stream_fps_interval=data["streaming"]["fps_interval"],
        stream_jpeg_quality=data["streaming"]["jpeg_quality"],
        stream_resolution=(
            data["streaming"]["resolution"]["width"],
            data["streaming"]["resolution"]["height"],
        ),
        frame_queue_size=data["queues"]["frame_queue_size"],
        broadcast_queue_size=data["queues"]["broadcast_queue_size"],
    )
