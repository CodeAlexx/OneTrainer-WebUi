# FFmpeg Video Editor - Implementation Plan

## Overview
A powerful yet intuitive video editor built on FFmpeg, designed for editing AI-generated video clips with professional features and simple controls.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    React Frontend (UI)                       │
├─────────────────────────────────────────────────────────────┤
│  Timeline │ Preview │ Effects Panel │ Properties │ Media Bin │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI Backend                            │
├─────────────────────────────────────────────────────────────┤
│  Project Manager │ Timeline Engine │ Render Queue            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   FFmpeg Engine                              │
├─────────────────────────────────────────────────────────────┤
│  Filter Graphs │ Frame Extract │ Encode/Decode │ Streaming   │
└─────────────────────────────────────────────────────────────┘
```

---

## UI Layout

```
┌──────────────────────────────────────────────────────────────────────┐
│ [+] [📁] [💾] [↩️] [↪️] │ [✂️] [🗑️] │ [🧲 Snap] │ [🔍 -] 100% [🔍 +] │ [▶ Export] │
├──────────────────────────────────────────────────────────────────────┤
│ Selected: clip.mp4 │ Type: Video │ 1920x1080 │ 24fps │ Duration: 5.2s │ [Replace] │
├────────────────────────────────────┬─────────────────────────────────┤
│                                    │  EFFECTS              [+ Add]  │
│         PREVIEW PLAYER             │  ┌─────────────────────────┐   │
│                                    │  │ ☑ Color Correction      │   │
│      ┌──────────────────────┐      │  │   Brightness: [====○==] │   │
│      │                      │      │  │   Contrast:   [===○===] │   │
│      │                      │      │  │   Saturation: [====○==] │   │
│      │    Video Preview     │      │  └─────────────────────────┘   │
│      │                      │      │  ┌─────────────────────────┐   │
│      │                      │      │  │ ☑ Speed                 │   │
│      └──────────────────────┘      │  │   Rate: [=○==========] 0.5x│ │
│                                    │  └─────────────────────────┘   │
│  [⏮] [⏪] [ ▶ ] [⏩] [⏭]  00:05.20 │─────────────────────────────────┤
├────────────────────────────────────┤  PROPERTIES                    │
│         MEDIA BIN                  │  ┌─────────────────────────┐   │
│  ┌─────┐ ┌─────┐ ┌─────┐          │  │ Position X: [0    ]     │   │
│  │ 📹  │ │ 📹  │ │ 🖼️  │          │  │ Position Y: [0    ]     │   │
│  │clip1│ │clip2│ │img1 │          │  │ Scale:      [100  ] %   │   │
│  └─────┘ └─────┘ └─────┘          │  │ Rotation:   [0    ] °   │   │
│  [+ Import Media]                  │  │ Opacity:    [100  ] %   │   │
│                                    │  └─────────────────────────┘   │
├────────────────────────────────────┴─────────────────────────────────┤
│ TIMELINE                                                    [Zoom: ══○══] │
├──────────────────────────────────────────────────────────────────────┤
│  00:00    00:05    00:10    00:15    00:20    00:25    00:30        │
│  │        │        │        │        │        │        │            │
├──────────────────────────────────────────────────────────────────────┤
│ V1 │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░│                                     │
│    │    clip1.mp4    │ fade │  clip2.mp4  │                          │
├──────────────────────────────────────────────────────────────────────┤
│ V2 │                    │▒▒▒▒▒▒▒▒▒▒▒▒▒│                              │
│    │                    │  overlay.png │                              │
├──────────────────────────────────────────────────────────────────────┤
│ A1 │████████████████████████████████████│                            │
│    │         background_music.mp3       │                            │
├──────────────────────────────────────────────────────────────────────┤
│ A2 │          │▓▓▓▓▓▓▓▓│                                             │
│    │          │ sfx.wav│                                             │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Core Features

### 1. Timeline
- **Multi-track**: Unlimited video + audio tracks
- **Drag & drop**: Move clips, resize handles for trim
- **Snapping**: Snap to playhead, clip edges, markers
- **Zoom**: Smooth zoom from 10% to 400%
- **Scroll**: Horizontal scroll with mousewheel
- **Waveforms**: Audio waveform display
- **Thumbnails**: Video thumbnail strip on clips

### 2. Clips
- **Types**: Video, Audio, Image, Text, Color (solid)
- **Trim**: Drag edges to trim in/out points
- **Split**: Split at playhead (S key)
- **Slip/Slide**: Alt+drag to slip content
- **Copy/Paste**: Duplicate clips
- **Link/Unlink**: Link video+audio together

### 3. Transitions (via FFmpeg xfade)
- Fade (in/out/cross)
- Dissolve
- Wipe (left/right/up/down)
- Slide
- Circle/Radial
- Custom (shader-based)

### 4. Effects (via FFmpeg filters)

**Color:**
- Brightness/Contrast/Saturation
- Color curves
- LUT support
- White balance
- Hue shift

**Transform:**
- Scale/Position/Rotation
- Crop
- Flip/Mirror
- Keyframe animation

**Stylize:**
- Blur (gaussian, motion, radial)
- Sharpen
- Denoise
- Glow
- Vignette

**Utility:**
- Speed/Reverse
- Chromakey (green screen)
- Stabilization
- Opacity

### 5. Text
- Rich text with fonts
- Position/scale/rotation
- Animated (fade, slide, typewriter)
- Subtitles (SRT import)

### 6. Audio
- Volume control per clip
- Fade in/out
- Waveform visualization
- Mute/solo tracks
- Audio sync

---

## Backend Implementation

### Data Models

```python
@dataclass
class Project:
    id: str
    name: str
    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    sample_rate: int = 48000
    tracks: List[Track]
    clips: List[Clip]

@dataclass
class Clip:
    id: str
    type: ClipType  # video, audio, image, text, color
    track_id: str

    # Source
    source_path: str
    source_in: float  # trim in point
    source_out: float  # trim out point

    # Timeline
    start_time: float
    duration: float

    # Transform
    position: Tuple[float, float]
    scale: float
    rotation: float
    opacity: float

    # Effects
    effects: List[Effect]

    # Transitions
    transition_in: Optional[Transition]
    transition_out: Optional[Transition]

@dataclass
class Effect:
    type: str  # brightness, blur, speed, etc.
    params: Dict[str, Any]
    keyframes: Dict[str, List[Keyframe]]  # animated params

@dataclass
class Transition:
    type: str  # fade, wipe, dissolve
    duration: float
    params: Dict[str, Any]
```

### FFmpeg Filter Graph Builder

```python
class FFmpegFilterBuilder:
    def build_clip_filters(self, clip: Clip) -> str:
        """Build filter chain for a single clip."""
        filters = []

        # Trim
        filters.append(f"trim=start={clip.source_in}:end={clip.source_out}")
        filters.append("setpts=PTS-STARTPTS")

        # Transform
        if clip.scale != 1.0:
            w = int(self.project.width * clip.scale)
            h = int(self.project.height * clip.scale)
            filters.append(f"scale={w}:{h}")

        if clip.rotation != 0:
            filters.append(f"rotate={clip.rotation}*PI/180")

        # Effects
        for effect in clip.effects:
            filters.append(self.build_effect(effect))

        # Opacity
        if clip.opacity < 1.0:
            filters.append(f"colorchannelmixer=aa={clip.opacity}")

        return ",".join(filters)

    def build_effect(self, effect: Effect) -> str:
        """Convert effect to FFmpeg filter."""
        if effect.type == "brightness":
            return f"eq=brightness={effect.params['value']}"
        elif effect.type == "blur":
            return f"gblur=sigma={effect.params['radius']}"
        elif effect.type == "speed":
            return f"setpts={1/effect.params['rate']}*PTS"
        # ... more effects

    def build_transition(self, clip1: Clip, clip2: Clip, trans: Transition) -> str:
        """Build xfade transition between clips."""
        return f"xfade=transition={trans.type}:duration={trans.duration}:offset={clip1.end_time - trans.duration}"
```

### Preview Engine

```python
class PreviewEngine:
    def __init__(self, project: Project):
        self.project = project
        self.cache = {}  # frame cache

    async def get_frame(self, time: float, width: int = 640) -> bytes:
        """Extract single frame at time."""
        cache_key = f"{time:.2f}_{width}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Build filter graph for this frame
        filter_graph = self.build_composite_filter(time)

        cmd = [
            "ffmpeg", "-y",
            *self.build_inputs(),
            "-filter_complex", filter_graph,
            "-ss", str(time),
            "-frames:v", "1",
            "-f", "image2pipe",
            "-vcodec", "png",
            "-"
        ]

        result = await asyncio.subprocess.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE
        )
        frame = await result.stdout.read()

        self.cache[cache_key] = frame
        return frame
```

### Export Engine

```python
class ExportEngine:
    async def export(self, project: Project, output_path: str,
                     format: str = "mp4", quality: str = "high"):
        """Export full timeline to video file."""

        filter_graph = self.build_full_filter_graph(project)

        cmd = [
            "ffmpeg", "-y",
            *self.build_all_inputs(project),
            "-filter_complex", filter_graph,
            "-map", "[vout]",
            "-map", "[aout]",
            *self.get_encoding_params(format, quality),
            output_path
        ]

        process = await asyncio.subprocess.create_subprocess_exec(
            *cmd,
            stderr=asyncio.subprocess.PIPE
        )

        # Parse progress from stderr
        async for line in process.stderr:
            if b"time=" in line:
                self.parse_progress(line)
```

---

## Frontend Components

### Timeline Component
```typescript
interface TimelineProps {
  project: Project;
  currentTime: number;
  zoom: number;
  onTimeChange: (time: number) => void;
  onClipMove: (clipId: string, newStart: number, newTrack: string) => void;
  onClipTrim: (clipId: string, newIn: number, newOut: number) => void;
}

// Features:
// - Virtual scrolling for performance
// - Smooth zoom with pinch/scroll
// - Drag handles for trim
// - Snap guides
// - Multi-select with shift/ctrl
// - Context menu (right-click)
```

### Preview Component
```typescript
interface PreviewProps {
  project: Project;
  currentTime: number;
  isPlaying: boolean;
  onPlayPause: () => void;
  onSeek: (time: number) => void;
}

// Features:
// - Frame-accurate preview
// - Playback at various speeds
// - Full-screen mode
// - Safe area guides
// - Pixel aspect ratio correction
```

### Effects Panel
```typescript
interface EffectsPanelProps {
  clip: Clip;
  onAddEffect: (type: string) => void;
  onUpdateEffect: (effectId: string, params: any) => void;
  onRemoveEffect: (effectId: string) => void;
  onReorderEffects: (newOrder: string[]) => void;
}

// Features:
// - Drag to reorder effects
// - Enable/disable toggle
// - Preset system
// - Copy/paste effects
// - Keyframe editor
```

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Space | Play/Pause |
| J/K/L | Reverse/Stop/Forward |
| I/O | Set In/Out points |
| S | Split at playhead |
| Del | Delete selected |
| Ctrl+C/V | Copy/Paste |
| Ctrl+Z/Y | Undo/Redo |
| +/- | Zoom in/out |
| Home/End | Go to start/end |
| Left/Right | Frame step |
| Shift+Left/Right | 1 second step |

---

## API Endpoints

```
POST   /api/editor/project/new          Create project
GET    /api/editor/project              Get current project
PUT    /api/editor/project              Update project settings

POST   /api/editor/media/import         Import media file
GET    /api/editor/media                List imported media
DELETE /api/editor/media/{id}           Remove media

POST   /api/editor/clip                 Add clip to timeline
PUT    /api/editor/clip/{id}            Update clip
DELETE /api/editor/clip/{id}            Remove clip
POST   /api/editor/clip/{id}/split      Split clip

POST   /api/editor/effect/{clip_id}     Add effect to clip
PUT    /api/editor/effect/{id}          Update effect
DELETE /api/editor/effect/{id}          Remove effect

GET    /api/editor/preview/{time}       Get preview frame
WS     /api/editor/preview/stream       Stream preview frames

POST   /api/editor/export               Start export
GET    /api/editor/export/progress      Get export progress
POST   /api/editor/export/cancel        Cancel export
```

---

## Implementation Phases

### Phase 1: Core (MVP)
- [ ] FFmpeg backend engine
- [ ] Basic timeline UI
- [ ] Import video/audio/image
- [ ] Trim clips
- [ ] Preview frames
- [ ] Basic export

### Phase 2: Effects
- [ ] Color correction
- [ ] Transform (scale/rotate/position)
- [ ] Speed control
- [ ] Blur/sharpen
- [ ] Opacity

### Phase 3: Transitions
- [ ] Fade transitions
- [ ] Wipe transitions
- [ ] Dissolve
- [ ] Transition UI

### Phase 4: Audio
- [ ] Audio waveforms
- [ ] Volume control
- [ ] Audio fade
- [ ] Mute/solo

### Phase 5: Polish
- [ ] Undo/redo
- [ ] Keyboard shortcuts
- [ ] Presets
- [ ] Thumbnails on clips
- [ ] Performance optimization

---

## Tech Stack

**Backend:**
- Python 3.11+
- FastAPI
- FFmpeg (subprocess)
- asyncio for non-blocking

**Frontend:**
- React 18
- TypeScript
- Tailwind CSS
- Zustand (state)
- React-DnD (drag/drop)

**FFmpeg Filters Used:**
- `trim`, `setpts` - trimming
- `scale`, `rotate`, `overlay` - transform
- `eq`, `colorbalance`, `lut3d` - color
- `gblur`, `unsharp`, `hqdn3d` - stylize
- `xfade` - transitions
- `amix`, `volume` - audio
