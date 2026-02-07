"""Video data loading and frame extraction for training datasets.

Provides frame extraction from video files with configurable frame counts,
format support, and temporal sampling strategies matching OneTrainer's
LoadVideo pipeline module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# Supported video container formats
SUPPORTED_VIDEO_EXTENSIONS = frozenset({
    ".mp4", ".avi", ".mov", ".webm", ".mkv", ".flv", ".wmv", ".mpeg", ".m4v",
})


@dataclass
class VideoFrames:
    """Container for extracted video frames and metadata."""

    frames: torch.Tensor  # [T, C, H, W] float in [0, 1]
    fps: float
    total_source_frames: int
    width: int
    height: int


def supported_video_extensions() -> frozenset[str]:
    """Return the set of supported video file extensions."""
    return SUPPORTED_VIDEO_EXTENSIONS


def is_video_file(path: Path) -> bool:
    """Check if a path points to a supported video file."""
    return Path(path).suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS


def _load_with_torchvision(
    path: Path,
    target_frame_count: int,
    target_fps: float | None = None,
) -> VideoFrames:
    """Load video frames using torchvision.io."""
    try:
        from torchvision.io import read_video
    except ImportError as exc:
        raise ImportError(
            "torchvision is required for video loading. "
            "Install with: pip install torchvision"
        ) from exc

    # read_video returns (video, audio, info)
    # video is [T, H, W, C] uint8
    video, _audio, info = read_video(str(path), pts_unit="sec")
    source_fps = info.get("video_fps", 24.0)
    total_source_frames = video.shape[0]

    if total_source_frames == 0:
        raise ValueError(f"No frames found in video: {path}")

    # Sample frames to match target count
    frames = _sample_frames(video, target_frame_count, source_fps, target_fps)

    # Convert from [T, H, W, C] uint8 to [T, C, H, W] float [0, 1]
    frames = frames.permute(0, 3, 1, 2).float() / 255.0

    return VideoFrames(
        frames=frames,
        fps=target_fps or source_fps,
        total_source_frames=total_source_frames,
        width=frames.shape[3],
        height=frames.shape[2],
    )


def _load_with_cv2(
    path: Path,
    target_frame_count: int,
    target_fps: float | None = None,
) -> VideoFrames:
    """Load video frames using OpenCV (cv2)."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "opencv-python is required for video loading when torchvision "
            "is unavailable. Install with: pip install opencv-python"
        ) from exc

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {path}")

    try:
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        total_source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Read all frames
        all_frames: list[torch.Tensor] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # cv2 reads BGR, convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            all_frames.append(torch.from_numpy(frame_rgb))
    finally:
        cap.release()

    if not all_frames:
        raise ValueError(f"No frames found in video: {path}")

    # Stack into [T, H, W, C]
    video = torch.stack(all_frames)
    frames = _sample_frames(video, target_frame_count, source_fps, target_fps)

    # Convert from [T, H, W, C] uint8 to [T, C, H, W] float [0, 1]
    frames = frames.permute(0, 3, 1, 2).float() / 255.0

    return VideoFrames(
        frames=frames,
        fps=target_fps or source_fps,
        total_source_frames=total_source_frames,
        width=width,
        height=height,
    )


def _sample_frames(
    video: torch.Tensor,
    target_frame_count: int,
    source_fps: float,
    target_fps: float | None,
) -> torch.Tensor:
    """Sample frames from video tensor [T, H, W, C].

    If target_fps is set and differs from source, temporal resampling is
    applied first.  Then frames are uniformly sampled to reach
    target_frame_count.
    """
    total = video.shape[0]

    # Temporal resampling if target fps differs
    if target_fps is not None and target_fps > 0 and source_fps > 0:
        ratio = target_fps / source_fps
        if ratio < 1.0:
            # Downsample: pick every Nth frame
            step = max(1, round(1.0 / ratio))
            indices = list(range(0, total, step))
            video = video[indices]
            total = video.shape[0]

    if target_frame_count <= 0 or target_frame_count >= total:
        return video

    # Uniform sampling across the video duration
    indices = torch.linspace(0, total - 1, target_frame_count).long()
    return video[indices]


def load_video(
    path: str | Path,
    target_frame_count: int = 16,
    target_fps: float | None = None,
    backend: str = "auto",
) -> VideoFrames:
    """Load and extract frames from a video file.

    Args:
        path: Path to the video file.
        target_frame_count: Number of frames to extract. 0 means all frames.
        target_fps: Target frame rate for temporal resampling. None keeps
            the original frame rate.
        backend: Video loading backend: "torchvision", "cv2", or "auto"
            (tries torchvision first, falls back to cv2).

    Returns:
        VideoFrames with extracted frames as [T, C, H, W] float tensor.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    if path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        raise ValueError(
            f"Unsupported video format: {path.suffix}. "
            f"Supported: {sorted(SUPPORTED_VIDEO_EXTENSIONS)}"
        )

    if backend == "torchvision":
        return _load_with_torchvision(path, target_frame_count, target_fps)
    elif backend == "cv2":
        return _load_with_cv2(path, target_frame_count, target_fps)
    else:
        # auto: try torchvision first
        try:
            return _load_with_torchvision(path, target_frame_count, target_fps)
        except ImportError:
            logger.debug("torchvision unavailable, falling back to cv2")
            return _load_with_cv2(path, target_frame_count, target_fps)


def discover_videos(
    directory: Path,
    include_subdirectories: bool = False,
) -> list[Path]:
    """Find all video files in a directory."""
    directory = Path(directory)
    if not directory.exists():
        return []

    if include_subdirectories:
        results: list[Path] = []
        for ext in SUPPORTED_VIDEO_EXTENSIONS:
            results.extend(directory.rglob(f"*{ext}"))
        return sorted(results)
    else:
        return sorted(
            p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
        )


__all__ = [
    "SUPPORTED_VIDEO_EXTENSIONS",
    "VideoFrames",
    "discover_videos",
    "is_video_file",
    "load_video",
    "supported_video_extensions",
]
