"""
FFmpeg-based Video Editor Engine
Powerful video editing with filter graphs, transitions, and effects
"""

import asyncio
import json
import os
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import shutil


# ============================================================================
# Enums
# ============================================================================

class ClipType(str, Enum):
    VIDEO = "video"
    AUDIO = "audio"
    IMAGE = "image"
    TEXT = "text"
    COLOR = "color"


class TransitionType(str, Enum):
    NONE = "none"
    FADE = "fade"
    DISSOLVE = "dissolve"
    WIPE_LEFT = "wipeleft"
    WIPE_RIGHT = "wiperight"
    WIPE_UP = "wipeup"
    WIPE_DOWN = "wipedown"
    SLIDE_LEFT = "slideleft"
    SLIDE_RIGHT = "slideright"
    CIRCLE_OPEN = "circleopen"
    CIRCLE_CLOSE = "circleclose"
    FADE_BLACK = "fadeblack"
    FADE_WHITE = "fadewhite"


class EffectType(str, Enum):
    # Color
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    SATURATION = "saturation"
    HUE = "hue"
    GAMMA = "gamma"
    # Stylize
    BLUR = "blur"
    SHARPEN = "sharpen"
    DENOISE = "denoise"
    GLOW = "glow"
    VIGNETTE = "vignette"
    # Utility
    SPEED = "speed"
    REVERSE = "reverse"
    CHROMAKEY = "chromakey"
    OPACITY = "opacity"
    FLIP_H = "flip_h"
    FLIP_V = "flip_v"


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class MediaFile:
    """Imported media file with metadata."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    path: str = ""
    name: str = ""
    type: str = ""  # video, audio, image
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    codec: str = ""
    audio_codec: str = ""
    sample_rate: int = 0
    channels: int = 0
    file_size: int = 0


@dataclass
class Effect:
    """Effect applied to a clip."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: EffectType = EffectType.BRIGHTNESS
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Transition:
    """Transition between clips."""
    type: TransitionType = TransitionType.NONE
    duration: float = 0.5


@dataclass
class Clip:
    """A clip on the timeline."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: ClipType = ClipType.VIDEO
    name: str = ""

    # Source
    media_id: str = ""  # Reference to MediaFile
    source_path: str = ""
    source_in: float = 0.0  # Trim in point
    source_out: float = 0.0  # Trim out point

    # Timeline position
    track_id: str = ""
    start_time: float = 0.0

    # Transform
    position_x: float = 0.0
    position_y: float = 0.0
    scale: float = 1.0
    rotation: float = 0.0
    opacity: float = 1.0

    # For text clips
    text_content: str = ""
    font_family: str = "Arial"
    font_size: int = 48
    font_color: str = "#FFFFFF"

    # For color clips
    color: str = "#000000"

    # Effects & transitions
    effects: List[Effect] = field(default_factory=list)
    transition_in: Optional[Transition] = None
    transition_out: Optional[Transition] = None

    # Audio
    volume: float = 1.0
    muted: bool = False

    @property
    def duration(self) -> float:
        return self.source_out - self.source_in

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration


@dataclass
class Track:
    """A track in the timeline."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    type: str = "video"  # video or audio
    order: int = 0
    muted: bool = False
    locked: bool = False
    visible: bool = True
    height: int = 60


@dataclass
class Project:
    """Video editing project."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Untitled Project"

    # Canvas
    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    sample_rate: int = 48000
    background_color: str = "#000000"

    # Content
    media: List[MediaFile] = field(default_factory=list)
    tracks: List[Track] = field(default_factory=list)
    clips: List[Clip] = field(default_factory=list)

    @property
    def duration(self) -> float:
        """Calculate project duration from clips."""
        if not self.clips:
            return 0.0
        return max(c.end_time for c in self.clips)


# ============================================================================
# FFmpeg Utilities
# ============================================================================

def run_ffprobe(path: str) -> Dict[str, Any]:
    """Get media file metadata using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    return json.loads(result.stdout)


def extract_frame(input_path: str, time: float, output_path: str,
                  width: int = 640, height: int = 360) -> bool:
    """Extract a single frame from video."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(time),
        "-i", input_path,
        "-vframes", "1",
        "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
        "-f", "image2",
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, timeout=30)
    return result.returncode == 0


# ============================================================================
# Filter Graph Builder
# ============================================================================

class FilterGraphBuilder:
    """Builds FFmpeg filter graphs for clips and timeline."""

    def __init__(self, project: Project):
        self.project = project

    def build_effect_filter(self, effect: Effect) -> str:
        """Convert effect to FFmpeg filter string."""
        if not effect.enabled:
            return ""

        p = effect.params

        if effect.type == EffectType.BRIGHTNESS:
            val = p.get("value", 0)
            return f"eq=brightness={val}"

        elif effect.type == EffectType.CONTRAST:
            val = p.get("value", 1)
            return f"eq=contrast={val}"

        elif effect.type == EffectType.SATURATION:
            val = p.get("value", 1)
            return f"eq=saturation={val}"

        elif effect.type == EffectType.HUE:
            val = p.get("value", 0)
            return f"hue=h={val}"

        elif effect.type == EffectType.GAMMA:
            val = p.get("value", 1)
            return f"eq=gamma={val}"

        elif effect.type == EffectType.BLUR:
            sigma = p.get("sigma", 5)
            return f"gblur=sigma={sigma}"

        elif effect.type == EffectType.SHARPEN:
            amount = p.get("amount", 1)
            return f"unsharp=5:5:{amount}:5:5:0"

        elif effect.type == EffectType.DENOISE:
            strength = p.get("strength", 4)
            return f"hqdn3d={strength}"

        elif effect.type == EffectType.GLOW:
            amount = p.get("amount", 0.5)
            return f"gblur=sigma=20,blend=all_mode=screen:all_opacity={amount}"

        elif effect.type == EffectType.VIGNETTE:
            amount = p.get("amount", 0.5)
            return f"vignette=PI/{2 + amount * 2}"

        elif effect.type == EffectType.SPEED:
            rate = p.get("rate", 1.0)
            if rate != 1.0:
                return f"setpts={1/rate}*PTS"
            return ""

        elif effect.type == EffectType.REVERSE:
            return "reverse"

        elif effect.type == EffectType.CHROMAKEY:
            color = p.get("color", "0x00FF00")
            similarity = p.get("similarity", 0.3)
            blend = p.get("blend", 0.1)
            return f"chromakey={color}:{similarity}:{blend}"

        elif effect.type == EffectType.OPACITY:
            val = p.get("value", 1.0)
            return f"colorchannelmixer=aa={val}"

        elif effect.type == EffectType.FLIP_H:
            return "hflip"

        elif effect.type == EffectType.FLIP_V:
            return "vflip"

        return ""

    def build_clip_filters(self, clip: Clip) -> List[str]:
        """Build filter chain for a clip."""
        filters = []

        # Trim
        if clip.type in [ClipType.VIDEO, ClipType.AUDIO]:
            filters.append(f"trim=start={clip.source_in}:end={clip.source_out}")
            filters.append("setpts=PTS-STARTPTS")

        # Scale to project size for video/image
        if clip.type in [ClipType.VIDEO, ClipType.IMAGE]:
            w, h = self.project.width, self.project.height
            filters.append(f"scale={w}:{h}:force_original_aspect_ratio=decrease")
            filters.append(f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2")

        # Transform
        if clip.scale != 1.0:
            sw = int(self.project.width * clip.scale)
            sh = int(self.project.height * clip.scale)
            filters.append(f"scale={sw}:{sh}")

        if clip.rotation != 0:
            filters.append(f"rotate={clip.rotation}*PI/180:fillcolor=none")

        # Effects
        for effect in clip.effects:
            ef = self.build_effect_filter(effect)
            if ef:
                filters.append(ef)

        # Opacity (must be last for video)
        if clip.opacity < 1.0 and clip.type != ClipType.AUDIO:
            filters.append(f"format=rgba,colorchannelmixer=aa={clip.opacity}")

        return filters

    def build_audio_filters(self, clip: Clip) -> List[str]:
        """Build audio filter chain for a clip."""
        filters = []

        # Trim
        filters.append(f"atrim=start={clip.source_in}:end={clip.source_out}")
        filters.append("asetpts=PTS-STARTPTS")

        # Volume
        if clip.volume != 1.0:
            filters.append(f"volume={clip.volume}")

        # Mute
        if clip.muted:
            filters.append("volume=0")

        # Speed (affects audio too)
        for effect in clip.effects:
            if effect.type == EffectType.SPEED and effect.enabled:
                rate = effect.params.get("rate", 1.0)
                if rate != 1.0:
                    filters.append(f"atempo={rate}")

        return filters


# ============================================================================
# Video Editor Engine
# ============================================================================

class VideoEditorEngine:
    """Main video editor engine using FFmpeg."""

    def __init__(self):
        self.project: Optional[Project] = None
        self.upload_dir = Path("editor_uploads")
        self.upload_dir.mkdir(exist_ok=True)
        self.cache_dir = Path("editor_cache")
        self.cache_dir.mkdir(exist_ok=True)

        # Export state
        self.is_exporting = False
        self.export_progress = 0.0
        self.export_cancel = False

    # ========================================================================
    # Project Management
    # ========================================================================

    def new_project(self, name: str = "Untitled", width: int = 1920,
                    height: int = 1080, fps: float = 30.0) -> Project:
        """Create a new project."""
        self.project = Project(
            name=name,
            width=width,
            height=height,
            fps=fps,
            tracks=[
                Track(name="Video 1", type="video", order=0),
                Track(name="Video 2", type="video", order=1),
                Track(name="Audio 1", type="audio", order=2),
                Track(name="Audio 2", type="audio", order=3),
            ]
        )
        self._clear_cache()
        return self.project

    def get_project(self) -> Optional[Project]:
        """Get current project."""
        return self.project

    def update_project(self, updates: Dict[str, Any]) -> Optional[Project]:
        """Update project settings."""
        if not self.project:
            return None

        for key, value in updates.items():
            if hasattr(self.project, key):
                setattr(self.project, key, value)

        self._clear_cache()
        return self.project

    # ========================================================================
    # Media Management
    # ========================================================================

    def import_media(self, file_path: str) -> MediaFile:
        """Import a media file and extract metadata."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get metadata
        probe = run_ffprobe(str(path))

        media = MediaFile(
            path=str(path.absolute()),
            name=path.name,
            file_size=path.stat().st_size,
        )

        # Parse format
        fmt = probe.get("format", {})
        media.duration = float(fmt.get("duration", 0))

        # Parse streams
        for stream in probe.get("streams", []):
            codec_type = stream.get("codec_type")

            if codec_type == "video":
                media.type = "video"
                media.width = stream.get("width", 0)
                media.height = stream.get("height", 0)
                media.codec = stream.get("codec_name", "")

                # Parse FPS
                fps_str = stream.get("r_frame_rate", "30/1")
                try:
                    num, den = map(int, fps_str.split("/"))
                    media.fps = num / den if den else 30.0
                except:
                    media.fps = 30.0

            elif codec_type == "audio":
                if not media.type:
                    media.type = "audio"
                media.audio_codec = stream.get("codec_name", "")
                media.sample_rate = int(stream.get("sample_rate", 48000))
                media.channels = stream.get("channels", 2)

        # Check for image
        if not media.type or (media.duration == 0 and media.width > 0):
            media.type = "image"
            media.duration = 5.0  # Default image duration

        # Add to project
        if self.project:
            self.project.media.append(media)

        return media

    def upload_media(self, filename: str, content: bytes) -> MediaFile:
        """Upload and import media file."""
        # Save file
        file_path = self.upload_dir / filename
        with open(file_path, "wb") as f:
            f.write(content)

        # Import
        return self.import_media(str(file_path))

    def get_media(self) -> List[MediaFile]:
        """Get all imported media."""
        return self.project.media if self.project else []

    def remove_media(self, media_id: str) -> bool:
        """Remove media from project."""
        if not self.project:
            return False

        self.project.media = [m for m in self.project.media if m.id != media_id]
        return True

    # ========================================================================
    # Track Management
    # ========================================================================

    def add_track(self, name: str, track_type: str = "video") -> Track:
        """Add a new track."""
        if not self.project:
            raise ValueError("No project loaded")

        order = len(self.project.tracks)
        track = Track(name=name, type=track_type, order=order)
        self.project.tracks.append(track)
        return track

    def remove_track(self, track_id: str) -> bool:
        """Remove a track and its clips."""
        if not self.project:
            return False

        self.project.tracks = [t for t in self.project.tracks if t.id != track_id]
        self.project.clips = [c for c in self.project.clips if c.track_id != track_id]
        self._clear_cache()
        return True

    def update_track(self, track_id: str, updates: Dict[str, Any]) -> Optional[Track]:
        """Update track properties."""
        if not self.project:
            return None

        for track in self.project.tracks:
            if track.id == track_id:
                for key, value in updates.items():
                    if hasattr(track, key):
                        setattr(track, key, value)
                return track
        return None

    # ========================================================================
    # Clip Management
    # ========================================================================

    def add_clip(self, clip_data: Dict[str, Any]) -> Clip:
        """Add a clip to the timeline."""
        if not self.project:
            raise ValueError("No project loaded")

        clip = Clip(
            type=ClipType(clip_data.get("type", "video")),
            name=clip_data.get("name", ""),
            media_id=clip_data.get("media_id", ""),
            source_path=clip_data.get("source_path", ""),
            source_in=clip_data.get("source_in", 0),
            source_out=clip_data.get("source_out", 0),
            track_id=clip_data.get("track_id", ""),
            start_time=clip_data.get("start_time", 0),
            text_content=clip_data.get("text_content", ""),
            color=clip_data.get("color", "#000000"),
        )

        # Auto-detect duration from media
        if clip.source_path and clip.source_out == 0:
            try:
                probe = run_ffprobe(clip.source_path)
                duration = float(probe.get("format", {}).get("duration", 5))
                clip.source_out = duration
            except:
                clip.source_out = 5.0

        self.project.clips.append(clip)
        self._clear_cache()
        return clip

    def update_clip(self, clip_id: str, updates: Dict[str, Any]) -> Optional[Clip]:
        """Update clip properties."""
        if not self.project:
            return None

        for clip in self.project.clips:
            if clip.id == clip_id:
                for key, value in updates.items():
                    if hasattr(clip, key):
                        setattr(clip, key, value)
                self._clear_cache()
                return clip
        return None

    def remove_clip(self, clip_id: str) -> bool:
        """Remove a clip."""
        if not self.project:
            return False

        self.project.clips = [c for c in self.project.clips if c.id != clip_id]
        self._clear_cache()
        return True

    def split_clip(self, clip_id: str, split_time: float) -> Tuple[Optional[Clip], Optional[Clip]]:
        """Split a clip at the given time."""
        if not self.project:
            return None, None

        clip = next((c for c in self.project.clips if c.id == clip_id), None)
        if not clip:
            return None, None

        # Check if split point is within clip
        if split_time <= clip.start_time or split_time >= clip.end_time:
            return None, None

        # Calculate split point in source
        offset = split_time - clip.start_time
        source_split = clip.source_in + offset

        # Create second clip
        clip2 = Clip(
            type=clip.type,
            name=f"{clip.name} (2)",
            media_id=clip.media_id,
            source_path=clip.source_path,
            source_in=source_split,
            source_out=clip.source_out,
            track_id=clip.track_id,
            start_time=split_time,
            position_x=clip.position_x,
            position_y=clip.position_y,
            scale=clip.scale,
            rotation=clip.rotation,
            opacity=clip.opacity,
            volume=clip.volume,
            effects=[Effect(**asdict(e)) for e in clip.effects],
        )

        # Modify first clip
        clip.source_out = source_split
        clip.name = f"{clip.name} (1)"

        self.project.clips.append(clip2)
        self._clear_cache()
        return clip, clip2

    # ========================================================================
    # Effects Management
    # ========================================================================

    def add_effect(self, clip_id: str, effect_type: str, params: Dict[str, Any] = None) -> Optional[Effect]:
        """Add an effect to a clip."""
        if not self.project:
            return None

        clip = next((c for c in self.project.clips if c.id == clip_id), None)
        if not clip:
            return None

        effect = Effect(
            type=EffectType(effect_type),
            params=params or {}
        )
        clip.effects.append(effect)
        self._clear_cache()
        return effect

    def update_effect(self, clip_id: str, effect_id: str, updates: Dict[str, Any]) -> Optional[Effect]:
        """Update effect parameters."""
        if not self.project:
            return None

        clip = next((c for c in self.project.clips if c.id == clip_id), None)
        if not clip:
            return None

        for effect in clip.effects:
            if effect.id == effect_id:
                for key, value in updates.items():
                    if key == "params":
                        effect.params.update(value)
                    elif hasattr(effect, key):
                        setattr(effect, key, value)
                self._clear_cache()
                return effect
        return None

    def remove_effect(self, clip_id: str, effect_id: str) -> bool:
        """Remove an effect from a clip."""
        if not self.project:
            return False

        clip = next((c for c in self.project.clips if c.id == clip_id), None)
        if not clip:
            return False

        clip.effects = [e for e in clip.effects if e.id != effect_id]
        self._clear_cache()
        return True

    # ========================================================================
    # Transitions
    # ========================================================================

    def set_transition(self, clip_id: str, position: str,
                       trans_type: str, duration: float = 0.5) -> bool:
        """Set transition on clip (in or out)."""
        if not self.project:
            return False

        clip = next((c for c in self.project.clips if c.id == clip_id), None)
        if not clip:
            return False

        transition = Transition(
            type=TransitionType(trans_type),
            duration=duration
        )

        if position == "in":
            clip.transition_in = transition
        else:
            clip.transition_out = transition

        self._clear_cache()
        return True

    # ========================================================================
    # Preview
    # ========================================================================

    def get_preview_frame(self, time: float, width: int = 640, height: int = 360) -> Optional[bytes]:
        """Get a preview frame at the given time."""
        if not self.project or not self.project.clips:
            # Return black frame
            return self._generate_black_frame(width, height)

        # Find clips at this time
        active_clips = [c for c in self.project.clips
                       if c.start_time <= time < c.end_time
                       and c.type in [ClipType.VIDEO, ClipType.IMAGE]]

        if not active_clips:
            return self._generate_black_frame(width, height)

        # Use first video clip for preview (simple version)
        clip = active_clips[0]
        source_time = clip.source_in + (time - clip.start_time)

        # Check cache
        cache_key = f"{clip.id}_{source_time:.2f}_{width}_{height}"
        cache_path = self.cache_dir / f"{cache_key}.png"

        if cache_path.exists():
            return cache_path.read_bytes()

        # Extract frame
        try:
            if extract_frame(clip.source_path, source_time, str(cache_path), width, height):
                return cache_path.read_bytes()
        except Exception as e:
            print(f"Preview error: {e}")

        return self._generate_black_frame(width, height)

    def _generate_black_frame(self, width: int, height: int) -> bytes:
        """Generate a black frame."""
        cache_path = self.cache_dir / f"black_{width}_{height}.png"

        if not cache_path.exists():
            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", f"color=black:s={width}x{height}:d=1",
                "-frames:v", "1",
                str(cache_path)
            ]
            subprocess.run(cmd, capture_output=True, timeout=10)

        if cache_path.exists():
            return cache_path.read_bytes()
        return b""

    def _clear_cache(self):
        """Clear preview cache."""
        for f in self.cache_dir.glob("*.png"):
            if not f.name.startswith("black_"):
                try:
                    f.unlink()
                except:
                    pass

    # ========================================================================
    # Export
    # ========================================================================

    def export_video(self, output_path: str, format: str = "mp4",
                     quality: str = "high") -> bool:
        """Export the timeline to a video file."""
        if not self.project or not self.project.clips:
            return False

        self.is_exporting = True
        self.export_progress = 0.0
        self.export_cancel = False

        try:
            # Get video clips sorted by start time
            video_clips = sorted(
                [c for c in self.project.clips if c.type in [ClipType.VIDEO, ClipType.IMAGE]],
                key=lambda c: c.start_time
            )

            if not video_clips:
                return False

            # Build FFmpeg command
            cmd = self._build_export_command(video_clips, output_path, format, quality)

            # Run export
            process = subprocess.Popen(
                cmd,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )

            duration = self.project.duration

            # Parse progress
            for line in process.stderr:
                if self.export_cancel:
                    process.kill()
                    return False

                if "time=" in line:
                    try:
                        time_str = line.split("time=")[1].split()[0]
                        parts = time_str.split(":")
                        current_time = float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
                        self.export_progress = min(current_time / duration, 1.0) if duration > 0 else 0
                    except:
                        pass

            process.wait()
            self.export_progress = 1.0
            return process.returncode == 0

        except Exception as e:
            print(f"Export error: {e}")
            return False
        finally:
            self.is_exporting = False

    def _build_export_command(self, clips: List[Clip], output_path: str,
                               format: str, quality: str) -> List[str]:
        """Build FFmpeg export command."""
        cmd = ["ffmpeg", "-y"]

        # Add inputs
        for clip in clips:
            cmd.extend(["-i", clip.source_path])

        # Build filter graph
        filter_builder = FilterGraphBuilder(self.project)
        filter_parts = []

        for i, clip in enumerate(clips):
            # Video filters
            v_filters = filter_builder.build_clip_filters(clip)
            if v_filters:
                filter_parts.append(f"[{i}:v]{','.join(v_filters)}[v{i}]")
            else:
                filter_parts.append(f"[{i}:v]null[v{i}]")

        # Concat if multiple clips (simple version - no transitions)
        if len(clips) > 1:
            inputs = "".join(f"[v{i}]" for i in range(len(clips)))
            filter_parts.append(f"{inputs}concat=n={len(clips)}:v=1:a=0[vout]")
        else:
            filter_parts.append(f"[v0]null[vout]")

        cmd.extend(["-filter_complex", ";".join(filter_parts)])
        cmd.extend(["-map", "[vout]"])

        # Encoding settings
        if format == "mp4":
            cmd.extend(["-c:v", "libx264"])
            if quality == "high":
                cmd.extend(["-crf", "18", "-preset", "slow"])
            elif quality == "medium":
                cmd.extend(["-crf", "23", "-preset", "medium"])
            else:
                cmd.extend(["-crf", "28", "-preset", "fast"])

        cmd.extend(["-pix_fmt", "yuv420p"])
        cmd.append(output_path)

        return cmd

    def cancel_export(self):
        """Cancel ongoing export."""
        self.export_cancel = True


# ============================================================================
# Singleton Instance
# ============================================================================

_editor_instance: Optional[VideoEditorEngine] = None

def get_video_editor() -> VideoEditorEngine:
    """Get or create video editor instance."""
    global _editor_instance
    if _editor_instance is None:
        _editor_instance = VideoEditorEngine()
    return _editor_instance
