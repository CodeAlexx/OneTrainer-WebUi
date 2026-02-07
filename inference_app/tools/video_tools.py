"""
Video Tools - Lightweight video editing using FFmpeg
Crop, trim, resize, extract frames, convert formats
"""

import subprocess
import shutil
from pathlib import Path
from typing import Optional, Tuple, List
import json
import tempfile


def get_ffmpeg_path() -> str:
    """Get FFmpeg executable path."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg not found. Install with: apt install ffmpeg")
    return ffmpeg


def get_ffprobe_path() -> str:
    """Get FFprobe executable path."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("FFprobe not found. Install with: apt install ffmpeg")
    return ffprobe


def get_video_info(video_path: str) -> dict:
    """
    Get video metadata (duration, resolution, fps, codec).

    Returns:
        dict with keys: duration, width, height, fps, codec, audio_codec
    """
    cmd = [
        get_ffprobe_path(),
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFprobe failed: {result.stderr}")

    data = json.loads(result.stdout)

    # Find video stream
    video_stream = None
    audio_stream = None
    for stream in data.get("streams", []):
        if stream["codec_type"] == "video" and not video_stream:
            video_stream = stream
        elif stream["codec_type"] == "audio" and not audio_stream:
            audio_stream = stream

    if not video_stream:
        raise RuntimeError("No video stream found")

    # Parse frame rate
    fps_str = video_stream.get("r_frame_rate", "30/1")
    if "/" in fps_str:
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 30.0
    else:
        fps = float(fps_str)

    return {
        "duration": float(data.get("format", {}).get("duration", 0)),
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "fps": fps,
        "codec": video_stream.get("codec_name", "unknown"),
        "audio_codec": audio_stream.get("codec_name") if audio_stream else None,
        "bitrate": int(data.get("format", {}).get("bit_rate", 0)),
        "frames": int(video_stream.get("nb_frames", 0)) or None,
    }


def trim_video(
    input_path: str,
    output_path: str,
    start_time: float,
    end_time: Optional[float] = None,
    duration: Optional[float] = None,
    codec: str = "copy",
) -> str:
    """
    Trim video to specified time range.

    Args:
        input_path: Source video
        output_path: Output video
        start_time: Start time in seconds
        end_time: End time in seconds (optional, use duration instead)
        duration: Duration in seconds (optional, use end_time instead)
        codec: Video codec - 'copy' for fast trim, 'libx264' for re-encode

    Returns:
        Output path
    """
    cmd = [get_ffmpeg_path(), "-y", "-i", input_path, "-ss", str(start_time)]

    if end_time is not None:
        cmd.extend(["-to", str(end_time)])
    elif duration is not None:
        cmd.extend(["-t", str(duration)])

    if codec == "copy":
        cmd.extend(["-c", "copy"])
    else:
        cmd.extend(["-c:v", codec, "-c:a", "aac"])

    cmd.append(output_path)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg trim failed: {result.stderr}")

    return output_path


def crop_video(
    input_path: str,
    output_path: str,
    x: int,
    y: int,
    width: int,
    height: int,
    codec: str = "libx264",
) -> str:
    """
    Crop video to specified region.

    Args:
        input_path: Source video
        output_path: Output video
        x, y: Top-left corner of crop region
        width, height: Size of crop region
        codec: Video codec

    Returns:
        Output path
    """
    crop_filter = f"crop={width}:{height}:{x}:{y}"

    cmd = [
        get_ffmpeg_path(), "-y", "-i", input_path,
        "-vf", crop_filter,
        "-c:v", codec, "-c:a", "aac",
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg crop failed: {result.stderr}")

    return output_path


def resize_video(
    input_path: str,
    output_path: str,
    width: Optional[int] = None,
    height: Optional[int] = None,
    scale: Optional[float] = None,
    codec: str = "libx264",
) -> str:
    """
    Resize video to specified dimensions.

    Args:
        input_path: Source video
        output_path: Output video
        width: Target width (use -1 to maintain aspect ratio)
        height: Target height (use -1 to maintain aspect ratio)
        scale: Scale factor (alternative to width/height)
        codec: Video codec

    Returns:
        Output path
    """
    if scale is not None:
        scale_filter = f"scale=iw*{scale}:ih*{scale}"
    elif width and height:
        # Ensure even dimensions
        w = width if width % 2 == 0 else width + 1
        h = height if height % 2 == 0 else height + 1
        scale_filter = f"scale={w}:{h}"
    elif width:
        scale_filter = f"scale={width}:-2"  # -2 ensures even height
    elif height:
        scale_filter = f"scale=-2:{height}"
    else:
        raise ValueError("Provide width, height, or scale")

    cmd = [
        get_ffmpeg_path(), "-y", "-i", input_path,
        "-vf", scale_filter,
        "-c:v", codec, "-c:a", "aac",
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg resize failed: {result.stderr}")

    return output_path


def extract_frames(
    input_path: str,
    output_dir: str,
    fps: Optional[float] = None,
    start_time: float = 0,
    duration: Optional[float] = None,
    format: str = "png",
) -> List[str]:
    """
    Extract frames from video.

    Args:
        input_path: Source video
        output_dir: Directory for extracted frames
        fps: Frames per second to extract (None = all frames)
        start_time: Start time in seconds
        duration: Duration to extract
        format: Output format (png, jpg)

    Returns:
        List of frame paths
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    output_pattern = str(Path(output_dir) / f"frame_%06d.{format}")

    cmd = [get_ffmpeg_path(), "-y", "-i", input_path]

    if start_time > 0:
        cmd.extend(["-ss", str(start_time)])
    if duration:
        cmd.extend(["-t", str(duration)])

    if fps:
        cmd.extend(["-vf", f"fps={fps}"])

    cmd.append(output_pattern)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg extract failed: {result.stderr}")

    # Get list of extracted frames
    frames = sorted(Path(output_dir).glob(f"frame_*.{format}"))
    return [str(f) for f in frames]


def frames_to_video(
    frame_pattern: str,
    output_path: str,
    fps: float = 30,
    codec: str = "libx264",
) -> str:
    """
    Create video from image sequence.

    Args:
        frame_pattern: Pattern like '/path/to/frames/frame_%06d.png'
        output_path: Output video path
        fps: Output frame rate
        codec: Video codec

    Returns:
        Output path
    """
    cmd = [
        get_ffmpeg_path(), "-y",
        "-framerate", str(fps),
        "-i", frame_pattern,
        "-c:v", codec,
        "-pix_fmt", "yuv420p",
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg frames_to_video failed: {result.stderr}")

    return output_path


def convert_video(
    input_path: str,
    output_path: str,
    codec: str = "libx264",
    audio_codec: str = "aac",
    crf: int = 23,
    preset: str = "medium",
) -> str:
    """
    Convert video to different format/codec.

    Args:
        input_path: Source video
        output_path: Output video (format determined by extension)
        codec: Video codec (libx264, libx265, vp9, etc.)
        audio_codec: Audio codec (aac, mp3, opus, etc.)
        crf: Quality (0-51, lower = better, 23 = default)
        preset: Encoding speed (ultrafast, fast, medium, slow, veryslow)

    Returns:
        Output path
    """
    cmd = [
        get_ffmpeg_path(), "-y", "-i", input_path,
        "-c:v", codec,
        "-crf", str(crf),
        "-preset", preset,
        "-c:a", audio_codec,
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg convert failed: {result.stderr}")

    return output_path


def video_to_gif(
    input_path: str,
    output_path: str,
    fps: int = 10,
    width: int = 480,
    start_time: float = 0,
    duration: Optional[float] = None,
) -> str:
    """
    Convert video to GIF.

    Args:
        input_path: Source video
        output_path: Output GIF
        fps: Output FPS
        width: Output width (-1 for auto height)
        start_time: Start time
        duration: Duration

    Returns:
        Output path
    """
    # Generate palette for better quality
    palette_path = tempfile.mktemp(suffix=".png")

    filters = f"fps={fps},scale={width}:-1:flags=lanczos"

    # Build base command
    base_cmd = [get_ffmpeg_path(), "-y", "-i", input_path]
    if start_time > 0:
        base_cmd.extend(["-ss", str(start_time)])
    if duration:
        base_cmd.extend(["-t", str(duration)])

    # Generate palette
    palette_cmd = base_cmd + [
        "-vf", f"{filters},palettegen",
        palette_path
    ]
    subprocess.run(palette_cmd, capture_output=True)

    # Generate GIF with palette
    gif_cmd = base_cmd + [
        "-i", palette_path,
        "-lavfi", f"{filters} [x]; [x][1:v] paletteuse",
        output_path
    ]

    result = subprocess.run(gif_cmd, capture_output=True, text=True)

    # Cleanup
    Path(palette_path).unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg GIF failed: {result.stderr}")

    return output_path


def add_audio(
    video_path: str,
    audio_path: str,
    output_path: str,
    replace: bool = True,
) -> str:
    """
    Add or replace audio track in video.

    Args:
        video_path: Source video
        audio_path: Audio file to add
        output_path: Output video
        replace: Replace existing audio (True) or mix (False)

    Returns:
        Output path
    """
    if replace:
        cmd = [
            get_ffmpeg_path(), "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            output_path
        ]
    else:
        cmd = [
            get_ffmpeg_path(), "-y",
            "-i", video_path,
            "-i", audio_path,
            "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=first",
            "-c:v", "copy",
            output_path
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg add_audio failed: {result.stderr}")

    return output_path


def extract_audio(
    input_path: str,
    output_path: str,
    codec: str = "aac",
) -> str:
    """
    Extract audio from video.

    Args:
        input_path: Source video
        output_path: Output audio file
        codec: Audio codec

    Returns:
        Output path
    """
    cmd = [
        get_ffmpeg_path(), "-y", "-i", input_path,
        "-vn", "-c:a", codec,
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg extract_audio failed: {result.stderr}")

    return output_path


def concat_videos(
    input_paths: List[str],
    output_path: str,
    codec: str = "libx264",
) -> str:
    """
    Concatenate multiple videos.

    Args:
        input_paths: List of video paths
        output_path: Output video
        codec: Video codec

    Returns:
        Output path
    """
    # Create concat file
    concat_file = tempfile.mktemp(suffix=".txt")
    with open(concat_file, "w") as f:
        for path in input_paths:
            f.write(f"file '{path}'\n")

    cmd = [
        get_ffmpeg_path(), "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_file,
        "-c:v", codec,
        "-c:a", "aac",
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    Path(concat_file).unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg concat failed: {result.stderr}")

    return output_path


def speed_video(
    input_path: str,
    output_path: str,
    speed: float = 2.0,
    audio: bool = True,
) -> str:
    """
    Change video speed.

    Args:
        input_path: Source video
        output_path: Output video
        speed: Speed multiplier (2.0 = 2x faster, 0.5 = half speed)
        audio: Adjust audio speed too

    Returns:
        Output path
    """
    video_filter = f"setpts={1/speed}*PTS"

    if audio:
        audio_filter = f"atempo={speed}"
        # atempo only supports 0.5-2.0, chain for larger ranges
        if speed > 2.0:
            audio_filter = f"atempo=2.0,atempo={speed/2.0}"
        elif speed < 0.5:
            audio_filter = f"atempo=0.5,atempo={speed/0.5}"

        cmd = [
            get_ffmpeg_path(), "-y", "-i", input_path,
            "-filter_complex", f"[0:v]{video_filter}[v];[0:a]{audio_filter}[a]",
            "-map", "[v]", "-map", "[a]",
            output_path
        ]
    else:
        cmd = [
            get_ffmpeg_path(), "-y", "-i", input_path,
            "-vf", video_filter,
            "-an",
            output_path
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg speed failed: {result.stderr}")

    return output_path


# Convenience class
class VideoEditor:
    """Simple video editor wrapper."""

    def __init__(self, video_path: str):
        self.path = video_path
        self.info = get_video_info(video_path)

    @property
    def duration(self) -> float:
        return self.info["duration"]

    @property
    def size(self) -> Tuple[int, int]:
        return (self.info["width"], self.info["height"])

    @property
    def fps(self) -> float:
        return self.info["fps"]

    def trim(self, start: float, end: float = None, output: str = None) -> "VideoEditor":
        output = output or self._temp_path("mp4")
        trim_video(self.path, output, start, end)
        return VideoEditor(output)

    def crop(self, x: int, y: int, w: int, h: int, output: str = None) -> "VideoEditor":
        output = output or self._temp_path("mp4")
        crop_video(self.path, output, x, y, w, h)
        return VideoEditor(output)

    def resize(self, width: int = None, height: int = None, output: str = None) -> "VideoEditor":
        output = output or self._temp_path("mp4")
        resize_video(self.path, output, width, height)
        return VideoEditor(output)

    def to_gif(self, fps: int = 10, width: int = 480, output: str = None) -> str:
        output = output or self._temp_path("gif")
        return video_to_gif(self.path, output, fps, width)

    def extract_frames(self, output_dir: str, fps: float = None) -> List[str]:
        return extract_frames(self.path, output_dir, fps)

    def _temp_path(self, ext: str) -> str:
        return tempfile.mktemp(suffix=f".{ext}")
