"""
Video Editor Backend - Pro Timeline Editor using Movis
Supports multi-track editing, effects, transitions, keyframes
"""

import json
import uuid
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np

import movis as mv
from movis.layer import Composition, Video, Audio, Image, Text, Rectangle
from movis.effect import GaussianBlur, DropShadow, HSLShift


# ============================================================================
# Enums & Types
# ============================================================================

class ClipType(str, Enum):
    VIDEO = "video"
    AUDIO = "audio"
    IMAGE = "image"
    TEXT = "text"
    SHAPE = "shape"
    COMPOSITION = "composition"


class TrackType(str, Enum):
    VIDEO = "video"
    AUDIO = "audio"
    EFFECT = "effect"


class TransitionType(str, Enum):
    NONE = "none"
    CROSSFADE = "crossfade"
    FADE_BLACK = "fade_black"
    WIPE_LEFT = "wipe_left"
    WIPE_RIGHT = "wipe_right"
    WIPE_UP = "wipe_up"
    WIPE_DOWN = "wipe_down"
    DISSOLVE = "dissolve"


class EffectType(str, Enum):
    BLUR = "blur"
    DROP_SHADOW = "drop_shadow"
    CHROMAKEY = "chromakey"
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    SATURATION = "saturation"
    HUE_SHIFT = "hue_shift"
    OPACITY = "opacity"
    SCALE = "scale"
    ROTATE = "rotate"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Keyframe:
    """Animation keyframe."""
    time: float  # in seconds
    value: Any
    easing: str = "linear"  # linear, ease_in, ease_out, ease_in_out


@dataclass
class Effect:
    """Effect applied to a clip."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: EffectType = EffectType.OPACITY
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)
    keyframes: Dict[str, List[Keyframe]] = field(default_factory=dict)


@dataclass
class Clip:
    """A clip on the timeline."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: ClipType = ClipType.VIDEO
    name: str = ""

    # Source
    source_path: str = ""
    source_start: float = 0.0  # in seconds
    source_end: float = 0.0

    # Timeline position
    track_id: str = ""
    start_time: float = 0.0  # in seconds
    duration: float = 0.0

    # Transform
    position: Tuple[float, float] = (0.0, 0.0)
    scale: Tuple[float, float] = (1.0, 1.0)
    rotation: float = 0.0
    opacity: float = 1.0
    anchor: Tuple[float, float] = (0.5, 0.5)

    # Transitions
    transition_in: TransitionType = TransitionType.NONE
    transition_in_duration: float = 0.0
    transition_out: TransitionType = TransitionType.NONE
    transition_out_duration: float = 0.0

    # Effects
    effects: List[Effect] = field(default_factory=list)

    # Text-specific
    text_content: str = ""
    font_family: str = "Arial"
    font_size: int = 48
    font_color: str = "#FFFFFF"

    # Shape-specific
    shape_type: str = "rectangle"
    shape_color: str = "#FFFFFF"
    shape_size: Tuple[int, int] = (100, 100)

    # Animation keyframes
    keyframes: Dict[str, List[Keyframe]] = field(default_factory=dict)


@dataclass
class Track:
    """A track in the timeline."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    type: TrackType = TrackType.VIDEO
    order: int = 0
    muted: bool = False
    locked: bool = False
    visible: bool = True
    height: int = 60  # UI height in pixels


@dataclass
class Project:
    """Video editing project."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Untitled Project"

    # Canvas settings
    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    duration: float = 300.0  # Total duration in seconds (auto-extends)
    background_color: str = "#000000"

    # Timeline
    tracks: List[Track] = field(default_factory=list)
    clips: List[Clip] = field(default_factory=list)

    # Project metadata
    created_at: str = ""
    modified_at: str = ""

    def __post_init__(self):
        if not self.tracks:
            # Create default tracks
            self.tracks = [
                Track(name="Video 1", type=TrackType.VIDEO, order=0),
                Track(name="Video 2", type=TrackType.VIDEO, order=1),
                Track(name="Audio 1", type=TrackType.AUDIO, order=2),
                Track(name="Audio 2", type=TrackType.AUDIO, order=3),
            ]


# ============================================================================
# Video Editor Engine
# ============================================================================

class VideoEditorEngine:
    """Main video editor engine using Movis."""

    def __init__(self):
        self.project: Optional[Project] = None
        self.composition: Optional[Composition] = None
        self.preview_cache: Dict[int, np.ndarray] = {}
        self.is_rendering = False
        self.render_progress = 0.0
        self.projects_dir = Path("/home/alex/OneTrainer/inference_app/projects")
        self.projects_dir.mkdir(exist_ok=True)
        self.exports_dir = Path("/home/alex/OneTrainer/inference_app/exports")
        self.exports_dir.mkdir(exist_ok=True)

    def new_project(self, name: str = "Untitled", width: int = 1920,
                    height: int = 1080, fps: float = 30.0) -> Project:
        """Create a new project."""
        from datetime import datetime
        now = datetime.now().isoformat()

        self.project = Project(
            name=name,
            width=width,
            height=height,
            fps=fps,
            created_at=now,
            modified_at=now,
        )
        self._rebuild_composition()
        return self.project

    def save_project(self, path: str = None) -> str:
        """Save project to file."""
        if not self.project:
            raise ValueError("No project loaded")

        if path is None:
            path = str(self.projects_dir / f"{self.project.id}.json")

        # Convert to dict for JSON serialization
        data = self._project_to_dict(self.project)

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        return path

    def load_project(self, path: str) -> Project:
        """Load project from file."""
        with open(path, 'r') as f:
            data = json.load(f)

        self.project = self._dict_to_project(data)
        self._rebuild_composition()
        return self.project

    def _project_to_dict(self, project: Project) -> Dict:
        """Convert project to serializable dict."""
        return {
            'id': project.id,
            'name': project.name,
            'width': project.width,
            'height': project.height,
            'fps': project.fps,
            'duration': project.duration,
            'background_color': project.background_color,
            'tracks': [asdict(t) for t in project.tracks],
            'clips': [self._clip_to_dict(c) for c in project.clips],
            'created_at': project.created_at,
            'modified_at': project.modified_at,
        }

    def _clip_to_dict(self, clip: Clip) -> Dict:
        """Convert clip to serializable dict."""
        d = asdict(clip)
        d['type'] = clip.type.value
        d['transition_in'] = clip.transition_in.value
        d['transition_out'] = clip.transition_out.value
        d['effects'] = [self._effect_to_dict(e) for e in clip.effects]
        return d

    def _effect_to_dict(self, effect: Effect) -> Dict:
        """Convert effect to serializable dict."""
        d = asdict(effect)
        d['type'] = effect.type.value
        return d

    def _dict_to_project(self, data: Dict) -> Project:
        """Convert dict to Project."""
        tracks = [Track(**t) for t in data.get('tracks', [])]
        clips = [self._dict_to_clip(c) for c in data.get('clips', [])]

        return Project(
            id=data.get('id', str(uuid.uuid4())),
            name=data.get('name', 'Untitled'),
            width=data.get('width', 1920),
            height=data.get('height', 1080),
            fps=data.get('fps', 30.0),
            duration=data.get('duration', 60.0),
            background_color=data.get('background_color', '#000000'),
            tracks=tracks,
            clips=clips,
            created_at=data.get('created_at', ''),
            modified_at=data.get('modified_at', ''),
        )

    def _dict_to_clip(self, data: Dict) -> Clip:
        """Convert dict to Clip."""
        data['type'] = ClipType(data.get('type', 'video'))
        data['transition_in'] = TransitionType(data.get('transition_in', 'none'))
        data['transition_out'] = TransitionType(data.get('transition_out', 'none'))
        data['effects'] = [self._dict_to_effect(e) for e in data.get('effects', [])]
        data['position'] = tuple(data.get('position', (0, 0)))
        data['scale'] = tuple(data.get('scale', (1, 1)))
        data['anchor'] = tuple(data.get('anchor', (0.5, 0.5)))
        data['shape_size'] = tuple(data.get('shape_size', (100, 100)))
        return Clip(**data)

    def _dict_to_effect(self, data: Dict) -> Effect:
        """Convert dict to Effect."""
        data['type'] = EffectType(data.get('type', 'opacity'))
        return Effect(**data)

    # ========================================================================
    # Track Management
    # ========================================================================

    def add_track(self, name: str, track_type: TrackType) -> Track:
        """Add a new track."""
        if not self.project:
            raise ValueError("No project loaded")

        order = max([t.order for t in self.project.tracks], default=-1) + 1
        track = Track(name=name, type=track_type, order=order)
        self.project.tracks.append(track)
        return track

    def remove_track(self, track_id: str) -> bool:
        """Remove a track and its clips."""
        if not self.project:
            return False

        self.project.tracks = [t for t in self.project.tracks if t.id != track_id]
        self.project.clips = [c for c in self.project.clips if c.track_id != track_id]
        self._rebuild_composition()
        return True

    def reorder_tracks(self, track_ids: List[str]) -> bool:
        """Reorder tracks by ID list."""
        if not self.project:
            return False

        track_map = {t.id: t for t in self.project.tracks}
        for i, tid in enumerate(track_ids):
            if tid in track_map:
                track_map[tid].order = i

        self.project.tracks.sort(key=lambda t: t.order)
        self._rebuild_composition()
        return True

    # ========================================================================
    # Clip Management
    # ========================================================================

    def add_clip(self, clip: Clip) -> Clip:
        """Add a clip to the timeline."""
        if not self.project:
            raise ValueError("No project loaded")

        # Auto-detect duration for media files
        if clip.type in [ClipType.VIDEO, ClipType.AUDIO] and clip.source_path:
            clip.duration = self._get_media_duration(clip.source_path)
            clip.source_end = clip.duration

        self.project.clips.append(clip)

        # Auto-extend project duration if clip extends beyond
        clip_end = clip.start_time + clip.duration
        if clip_end > self.project.duration:
            self.project.duration = clip_end + 10  # Add 10s buffer

        self._rebuild_composition()
        return clip

    def remove_clip(self, clip_id: str) -> bool:
        """Remove a clip from the timeline."""
        if not self.project:
            return False

        self.project.clips = [c for c in self.project.clips if c.id != clip_id]
        self._rebuild_composition()
        return True

    def update_clip(self, clip_id: str, updates: Dict[str, Any]) -> Optional[Clip]:
        """Update clip properties."""
        if not self.project:
            return None

        for clip in self.project.clips:
            if clip.id == clip_id:
                for key, value in updates.items():
                    if hasattr(clip, key):
                        setattr(clip, key, value)
                self._rebuild_composition()
                return clip
        return None

    def split_clip(self, clip_id: str, split_time: float) -> Tuple[Optional[Clip], Optional[Clip]]:
        """Split a clip at the specified time."""
        if not self.project:
            return None, None

        clip = next((c for c in self.project.clips if c.id == clip_id), None)
        if not clip:
            return None, None

        # Check if split_time is within clip bounds
        relative_time = split_time - clip.start_time
        if relative_time <= 0 or relative_time >= clip.duration:
            return None, None

        # Create second clip
        clip2 = Clip(
            type=clip.type,
            name=f"{clip.name} (2)",
            source_path=clip.source_path,
            source_start=clip.source_start + relative_time,
            source_end=clip.source_end,
            track_id=clip.track_id,
            start_time=split_time,
            duration=clip.duration - relative_time,
            position=clip.position,
            scale=clip.scale,
            rotation=clip.rotation,
            opacity=clip.opacity,
        )

        # Update first clip
        clip.duration = relative_time
        clip.source_end = clip.source_start + relative_time
        clip.name = f"{clip.name} (1)"

        self.project.clips.append(clip2)
        self._rebuild_composition()
        return clip, clip2

    def _get_media_duration(self, path: str) -> float:
        """Get duration of a media file."""
        try:
            import subprocess
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', path
            ], capture_output=True, text=True)
            return float(result.stdout.strip())
        except:
            return 5.0  # Default duration

    # ========================================================================
    # Effect Management
    # ========================================================================

    def add_effect(self, clip_id: str, effect_type: EffectType,
                   params: Dict[str, Any] = None) -> Optional[Effect]:
        """Add effect to a clip."""
        if not self.project:
            return None

        clip = next((c for c in self.project.clips if c.id == clip_id), None)
        if not clip:
            return None

        effect = Effect(
            type=effect_type,
            params=params or self._get_default_effect_params(effect_type)
        )
        clip.effects.append(effect)
        self._rebuild_composition()
        return effect

    def remove_effect(self, clip_id: str, effect_id: str) -> bool:
        """Remove effect from a clip."""
        if not self.project:
            return False

        clip = next((c for c in self.project.clips if c.id == clip_id), None)
        if not clip:
            return False

        clip.effects = [e for e in clip.effects if e.id != effect_id]
        self._rebuild_composition()
        return True

    def _get_default_effect_params(self, effect_type: EffectType) -> Dict[str, Any]:
        """Get default parameters for an effect."""
        defaults = {
            EffectType.BLUR: {"radius": 5.0},
            EffectType.DROP_SHADOW: {"offset": (5, 5), "blur": 10, "color": "#000000", "opacity": 0.5},
            EffectType.CHROMAKEY: {"color": "#00FF00", "threshold": 0.3},
            EffectType.BRIGHTNESS: {"value": 0.0},
            EffectType.CONTRAST: {"value": 1.0},
            EffectType.SATURATION: {"value": 1.0},
            EffectType.HUE_SHIFT: {"degrees": 0},
            EffectType.OPACITY: {"value": 1.0},
            EffectType.SCALE: {"x": 1.0, "y": 1.0},
            EffectType.ROTATE: {"degrees": 0},
        }
        return defaults.get(effect_type, {})

    # ========================================================================
    # Composition Building
    # ========================================================================

    def _rebuild_composition(self):
        """Rebuild the Movis composition from project data."""
        if not self.project:
            return

        # Clear preview cache
        self.preview_cache.clear()

        # Create new composition
        self.composition = Composition(
            size=(self.project.width, self.project.height),
            duration=self.project.duration,
        )

        # Parse background color
        bg_color = self._parse_color(self.project.background_color)

        # Add background
        self.composition.add_layer(
            Rectangle(
                size=(self.project.width, self.project.height),
                color=bg_color,
            ),
            name="background",
        )

        # Sort tracks by order (lower order = higher layer priority)
        sorted_tracks = sorted(self.project.tracks, key=lambda t: t.order, reverse=True)

        for track in sorted_tracks:
            if track.muted or not track.visible:
                continue

            # Get clips for this track, sorted by start time
            track_clips = sorted(
                [c for c in self.project.clips if c.track_id == track.id],
                key=lambda c: c.start_time
            )

            for clip in track_clips:
                self._add_clip_to_composition(clip)

    def _add_clip_to_composition(self, clip: Clip):
        """Add a clip to the composition."""
        layer = None

        try:
            if clip.type == ClipType.VIDEO:
                layer = Video(clip.source_path)
            elif clip.type == ClipType.AUDIO:
                layer = Audio(clip.source_path)
            elif clip.type == ClipType.IMAGE:
                layer = Image(clip.source_path)
            elif clip.type == ClipType.TEXT:
                layer = Text(
                    clip.text_content,
                    font_family=clip.font_family,
                    font_size=clip.font_size,
                    color=self._parse_color(clip.font_color),
                )
            elif clip.type == ClipType.SHAPE:
                if clip.shape_type == "rectangle":
                    layer = Rectangle(
                        size=clip.shape_size,
                        color=self._parse_color(clip.shape_color),
                    )

            if layer is None:
                return

            # Add to composition with timing
            layer_item = self.composition.add_layer(
                layer,
                name=clip.name or clip.id,
                offset=clip.start_time,
                start_time=clip.source_start,
                end_time=clip.source_start + clip.duration,
            )

            # Apply transform
            if clip.position != (0, 0):
                layer_item.position = clip.position
            if clip.scale != (1, 1):
                layer_item.scale = clip.scale
            if clip.rotation != 0:
                layer_item.rotation = clip.rotation
            if clip.opacity != 1.0:
                layer_item.opacity = clip.opacity

            # Apply effects
            for effect in clip.effects:
                if not effect.enabled:
                    continue
                self._apply_effect(layer_item, effect)

        except Exception as e:
            print(f"Failed to add clip {clip.id}: {e}")

    def _apply_effect(self, layer_item, effect: Effect):
        """Apply an effect to a layer."""
        try:
            if effect.type == EffectType.BLUR:
                radius = effect.params.get('radius', 5.0)
                layer_item.add_effect(GaussianBlur(radius=radius))
            elif effect.type == EffectType.DROP_SHADOW:
                offset = effect.params.get('offset', (5, 5))
                blur = effect.params.get('blur', 10)
                layer_item.add_effect(DropShadow(offset=offset, radius=blur))
            elif effect.type == EffectType.CHROMAKEY:
                # Chromakey not available in movis, use HSL shift as alternative
                hue = effect.params.get('hue_shift', 0)
                saturation = effect.params.get('saturation', 1.0)
                lightness = effect.params.get('lightness', 1.0)
                layer_item.add_effect(HSLShift(h=hue, s=saturation, l=lightness))
        except Exception as e:
            print(f"Failed to apply effect {effect.type}: {e}")

    def _parse_color(self, color: str) -> Tuple[int, int, int]:
        """Parse hex color to RGB tuple."""
        if color.startswith('#'):
            color = color[1:]
        return tuple(int(color[i:i+2], 16) for i in (0, 2, 4))

    # ========================================================================
    # Preview & Rendering
    # ========================================================================

    def get_preview_frame(self, time: float, width: int = None,
                          height: int = None, draft: bool = True) -> Optional[np.ndarray]:
        """Get a preview frame at the specified time."""
        if not self.composition:
            return None

        # Use cache if available
        frame_num = int(time * self.project.fps)
        if frame_num in self.preview_cache:
            return self.preview_cache[frame_num]

        try:
            # Render frame at current time
            frame = self.composition(time)

            # Resize if needed
            if width and height:
                from PIL import Image as PILImage
                img = PILImage.fromarray(frame)
                img = img.resize((width, height), PILImage.Resampling.LANCZOS)
                frame = np.array(img)

            # Cache frame
            self.preview_cache[frame_num] = frame

            return frame

        except Exception as e:
            print(f"Failed to render preview frame: {e}")
            return None

    def export_video(self, output_path: str, start_time: float = 0,
                     end_time: float = None, callback=None) -> bool:
        """Export the project to a video file."""
        if not self.composition or not self.project:
            return False

        if end_time is None:
            end_time = self.project.duration

        self.is_rendering = True
        self.render_progress = 0.0

        try:
            # Calculate total frames
            total_frames = int((end_time - start_time) * self.project.fps)

            # Export using movis
            self.composition.write_video(
                output_path,
                start_time=start_time,
                end_time=end_time,
            )

            self.render_progress = 1.0
            return True

        except Exception as e:
            print(f"Export failed: {e}")
            return False
        finally:
            self.is_rendering = False

    def export_frame(self, time: float, output_path: str) -> bool:
        """Export a single frame as image."""
        frame = self.get_preview_frame(time, draft=False)
        if frame is None:
            return False

        try:
            from PIL import Image as PILImage
            img = PILImage.fromarray(frame)
            img.save(output_path)
            return True
        except Exception as e:
            print(f"Failed to export frame: {e}")
            return False

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def get_timeline_data(self) -> Dict:
        """Get timeline data for UI rendering."""
        if not self.project:
            return {}

        return {
            'duration': self.project.duration,
            'fps': self.project.fps,
            'width': self.project.width,
            'height': self.project.height,
            'tracks': [asdict(t) for t in self.project.tracks],
            'clips': [self._clip_to_dict(c) for c in self.project.clips],
        }

    def import_media(self, file_path: str) -> Dict[str, Any]:
        """Import a media file and return its metadata."""
        import subprocess

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get media info using ffprobe
        try:
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', file_path
            ], capture_output=True, text=True)
            info = json.loads(result.stdout)

            # Determine media type
            has_video = any(s['codec_type'] == 'video' for s in info.get('streams', []))
            has_audio = any(s['codec_type'] == 'audio' for s in info.get('streams', []))

            if has_video:
                video_stream = next(s for s in info['streams'] if s['codec_type'] == 'video')
                return {
                    'type': 'video',
                    'path': file_path,
                    'duration': float(info['format'].get('duration', 0)),
                    'width': int(video_stream.get('width', 0)),
                    'height': int(video_stream.get('height', 0)),
                    'fps': eval(video_stream.get('r_frame_rate', '30/1')),
                }
            elif has_audio:
                return {
                    'type': 'audio',
                    'path': file_path,
                    'duration': float(info['format'].get('duration', 0)),
                }
            else:
                # Assume image
                return {
                    'type': 'image',
                    'path': file_path,
                }

        except Exception as e:
            print(f"Failed to get media info: {e}")
            return {'type': 'unknown', 'path': file_path}


# Singleton instance
_editor_instance = None

def get_video_editor() -> VideoEditorEngine:
    """Get the video editor instance."""
    global _editor_instance
    if _editor_instance is None:
        _editor_instance = VideoEditorEngine()
    return _editor_instance
