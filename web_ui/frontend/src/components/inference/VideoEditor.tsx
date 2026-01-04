/**
 * VideoEditor - Pro Timeline Video Editor
 * Full-featured video editing with timeline, tracks, effects, and export
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Play, Pause, SkipBack, SkipForward, Volume2, VolumeX,
  Scissors, Plus, Trash2, Film, Music,
  Download, Save, FolderOpen,
  ZoomIn, ZoomOut, Magnet,
  Eye, EyeOff, Undo2, Redo2
} from 'lucide-react';

const API_BASE = '/api/editor';

// Types
interface Track {
  id: string;
  name: string;
  type: 'video' | 'audio';
  order: number;
  muted: boolean;
  locked: boolean;
  visible: boolean;
  height: number;
}

interface Transition {
  type: string;
  duration: number;
}

interface Clip {
  id: string;
  type: 'video' | 'audio' | 'image' | 'text' | 'color';
  name: string;
  source_path: string;
  media_id: string;
  track_id: string;
  // Trim points
  source_in: number;
  source_out: number;
  // Timeline
  start_time: number;
  duration: number;
  end_time: number;
  // Transform
  position_x: number;
  position_y: number;
  scale: number;
  rotation: number;
  opacity: number;
  // Effects & transitions
  effects: Effect[];
  transition_in?: Transition;
  transition_out?: Transition;
  // Audio
  volume: number;
  muted: boolean;
  // Text
  text_content?: string;
  font_family?: string;
  font_size?: number;
  font_color?: string;
  // Color
  color?: string;
}

interface Effect {
  id: string;
  type: string;
  enabled: boolean;
  params: Record<string, any>;
}

interface Project {
  id: string;
  name: string;
  width: number;
  height: number;
  fps: number;
  duration: number;
  tracks: Track[];
  clips: Clip[];
}

// Available effects
const AVAILABLE_EFFECTS = [
  { type: 'brightness', name: 'Brightness', category: 'color', defaultParams: { value: 0 } },
  { type: 'contrast', name: 'Contrast', category: 'color', defaultParams: { value: 1 } },
  { type: 'saturation', name: 'Saturation', category: 'color', defaultParams: { value: 1 } },
  { type: 'hue', name: 'Hue Shift', category: 'color', defaultParams: { value: 0 } },
  { type: 'gamma', name: 'Gamma', category: 'color', defaultParams: { value: 1 } },
  { type: 'blur', name: 'Blur', category: 'stylize', defaultParams: { radius: 5 } },
  { type: 'sharpen', name: 'Sharpen', category: 'stylize', defaultParams: { amount: 1 } },
  { type: 'denoise', name: 'Denoise', category: 'stylize', defaultParams: { strength: 4 } },
  { type: 'vignette', name: 'Vignette', category: 'stylize', defaultParams: { amount: 0.3 } },
  { type: 'speed', name: 'Speed', category: 'utility', defaultParams: { rate: 1 } },
  { type: 'chromakey', name: 'Green Screen', category: 'utility', defaultParams: { color: '#00FF00', similarity: 0.3 } },
  { type: 'flip_h', name: 'Flip Horizontal', category: 'utility', defaultParams: {} },
  { type: 'flip_v', name: 'Flip Vertical', category: 'utility', defaultParams: {} },
];

const AVAILABLE_TRANSITIONS = [
  { type: 'fade', name: 'Fade' },
  { type: 'dissolve', name: 'Dissolve' },
  { type: 'wipeleft', name: 'Wipe Left' },
  { type: 'wiperight', name: 'Wipe Right' },
  { type: 'wipeup', name: 'Wipe Up' },
  { type: 'wipedown', name: 'Wipe Down' },
  { type: 'slideleft', name: 'Slide Left' },
  { type: 'slideright', name: 'Slide Right' },
  { type: 'circleopen', name: 'Circle Open' },
  { type: 'circleclose', name: 'Circle Close' },
  { type: 'fadeblack', name: 'Fade to Black' },
  { type: 'fadewhite', name: 'Fade to White' },
];

// Timeline constants
const TRACK_HEIGHT = 60;
const TIMELINE_HEADER_HEIGHT = 30;
const PIXELS_PER_SECOND_DEFAULT = 50;

export const VideoEditor: React.FC = () => {
  // Project state
  const [project, setProject] = useState<Project | null>(null);
  const [tracks, setTracks] = useState<Track[]>([]);
  const [clips, setClips] = useState<Clip[]>([]);

  // Playback state
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);

  // Timeline state
  const [zoom, setZoom] = useState(1);
  const [scrollLeft] = useState(0);
  const [selectedClipId, setSelectedClipId] = useState<string | null>(null);
  const [selectedTrackId, setSelectedTrackId] = useState<string | null>(null);
  const [snapEnabled, setSnapEnabled] = useState(true);

  // In/Out points for selection
  const [inPoint, setInPoint] = useState<number | null>(null);
  const [outPoint, setOutPoint] = useState<number | null>(null);

  // Drag state
  const [isDragging, setIsDragging] = useState(false);
  const [dragClipId, setDragClipId] = useState<string | null>(null);
  const [dragStartX, setDragStartX] = useState(0);
  const [dragStartTime, setDragStartTime] = useState(0);

  // UI state
  const [previewFrame, setPreviewFrame] = useState<string | null>(null);
  const [showEffectsPanel, setShowEffectsPanel] = useState(true);
  const [isExporting, setIsExporting] = useState(false);
  const [exportProgress, setExportProgress] = useState(0);
  const [jumpTimeInput, setJumpTimeInput] = useState('');

  // Undo/Redo history
  const [history, setHistory] = useState<{ clips: Clip[]; tracks: Track[] }[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const maxHistory = 50;

  // Save state to history
  const saveToHistory = useCallback((newClips: Clip[], newTracks: Track[]) => {
    setHistory(prev => {
      // Remove any redo states
      const trimmed = prev.slice(0, historyIndex + 1);
      // Add new state
      const updated = [...trimmed, { clips: JSON.parse(JSON.stringify(newClips)), tracks: JSON.parse(JSON.stringify(newTracks)) }];
      // Limit history size
      if (updated.length > maxHistory) {
        return updated.slice(-maxHistory);
      }
      return updated;
    });
    setHistoryIndex(prev => Math.min(prev + 1, maxHistory - 1));
  }, [historyIndex]);

  // Undo
  const undo = useCallback(() => {
    if (historyIndex > 0) {
      const prevState = history[historyIndex - 1];
      setClips(prevState.clips);
      setTracks(prevState.tracks);
      setHistoryIndex(prev => prev - 1);
    }
  }, [history, historyIndex]);

  // Redo
  const redo = useCallback(() => {
    if (historyIndex < history.length - 1) {
      const nextState = history[historyIndex + 1];
      setClips(nextState.clips);
      setTracks(nextState.tracks);
      setHistoryIndex(prev => prev + 1);
    }
  }, [history, historyIndex]);

  const canUndo = historyIndex > 0;
  const canRedo = historyIndex < history.length - 1;

  // Refs
  const timelineRef = useRef<HTMLDivElement>(null);
  const timelineRulerRef = useRef<HTMLDivElement>(null);
  const playbackRef = useRef<number | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Sync timeline ruler scroll with content
  const handleTimelineScroll = (e: React.UIEvent<HTMLDivElement>) => {
    if (timelineRulerRef.current) {
      timelineRulerRef.current.scrollLeft = e.currentTarget.scrollLeft;
    }
  };

  const pixelsPerSecond = PIXELS_PER_SECOND_DEFAULT * zoom;

  // Initialize project
  useEffect(() => {
    createNewProject();
  }, []);

  // Keyboard shortcuts for undo/redo and in/out points
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
        e.preventDefault();
        undo();
      } else if ((e.ctrlKey || e.metaKey) && (e.key === 'y' || (e.key === 'z' && e.shiftKey))) {
        e.preventDefault();
        redo();
      } else if (e.key === 'i' || e.key === 'I') {
        // Set In point
        setInPoint(currentTime);
      } else if (e.key === 'o' || e.key === 'O') {
        // Set Out point
        setOutPoint(currentTime);
      } else if (e.key === ' ' && !e.target) {
        // Space to play/pause
        e.preventDefault();
        setIsPlaying(prev => !prev);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [undo, redo, currentTime]);

  // Mouse wheel scrubbing on timeline
  const handleTimelineWheel = useCallback((e: React.WheelEvent) => {
    // Only scrub if middle mouse button is pressed or Ctrl is held
    if (e.buttons === 4 || e.ctrlKey) {
      e.preventDefault();
      const delta = e.deltaY > 0 ? 0.5 : -0.5; // 0.5 seconds per scroll tick
      setCurrentTime(prev => Math.max(0, Math.min(prev + delta, project?.duration || 60)));
    }
  }, [project?.duration]);

  // Save initial state to history when project loads
  useEffect(() => {
    if (clips.length > 0 || tracks.length > 0) {
      if (history.length === 0) {
        saveToHistory(clips, tracks);
      }
    }
  }, [clips, tracks, history.length, saveToHistory]);

  // Playback loop
  useEffect(() => {
    if (isPlaying && project) {
      const intervalId = window.setInterval(() => {
        setCurrentTime(prev => {
          const next = prev + 1 / project.fps;
          // Stop at out point if selection is active
          if (outPoint !== null && inPoint !== null && next >= outPoint) {
            // Clear interval immediately and stop
            clearInterval(intervalId);
            setIsPlaying(false);
            return outPoint;
          }
          if (next >= project.duration) {
            // Clear interval immediately and stop
            clearInterval(intervalId);
            setIsPlaying(false);
            return 0;
          }
          return next;
        });
      }, 1000 / project.fps);
      playbackRef.current = intervalId;

      return () => {
        clearInterval(intervalId);
        playbackRef.current = null;
      };
    } else {
      // Clear any existing interval when not playing
      if (playbackRef.current) {
        clearInterval(playbackRef.current);
        playbackRef.current = null;
      }
    }
  }, [isPlaying, project, inPoint, outPoint]);

  // Update preview frame
  useEffect(() => {
    if (project) {
      fetchPreviewFrame(currentTime);
    }
  }, [currentTime, project, clips]);

  // API Functions
  const createNewProject = async () => {
    const res = await fetch(`${API_BASE}/project/new`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: 'New Project', width: 1920, height: 1080, fps: 30 })
    });
    const data = await res.json();
    if (data.success) {
      setProject(data.project);
      setTracks(data.project.tracks);
      setClips(data.project.clips);
    }
  };

  const fetchPreviewFrame = async (time: number) => {
    try {
      const res = await fetch(`${API_BASE}/preview/${time}?width=640&height=360&draft=true`);
      const data = await res.json();
      if (data.success) {
        setPreviewFrame(data.frame);
      }
    } catch (e) {
      // Ignore preview errors
    }
  };

  const addClip = async (type: string, trackId: string, startTime: number, sourcePath?: string) => {
    const res = await fetch(`${API_BASE}/clip/add`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        type,
        name: `${type.charAt(0).toUpperCase() + type.slice(1)} Clip`,
        source_path: sourcePath || '',
        track_id: trackId,
        start_time: startTime,
        duration: 5,
        text_content: type === 'text' ? 'Sample Text' : '',
      })
    });
    const data = await res.json();
    if (data.success) {
      setClips(prev => [...prev, data.clip]);
    }
  };

  const updateClip = async (clipId: string, updates: Partial<Clip>) => {
    // Save current state to history before update
    saveToHistory(clips, tracks);

    const res = await fetch(`${API_BASE}/clip/${clipId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ updates })
    });
    const data = await res.json();
    if (data.success) {
      setClips(prev => prev.map(c => c.id === clipId ? data.clip : c));
    }
  };

  const deleteClip = async (clipId: string) => {
    // Save current state to history before delete
    saveToHistory(clips, tracks);

    const res = await fetch(`${API_BASE}/clip/${clipId}`, { method: 'DELETE' });
    const data = await res.json();
    if (data.success) {
      setClips(prev => prev.filter(c => c.id !== clipId));
      if (selectedClipId === clipId) setSelectedClipId(null);
    }
  };

  // Jump to specific time (format: mm:ss or hh:mm:ss)
  const jumpToTime = (timeStr: string) => {
    const parts = timeStr.split(':').map(Number);
    let seconds = 0;
    if (parts.length === 3) {
      // hh:mm:ss
      seconds = parts[0] * 3600 + parts[1] * 60 + parts[2];
    } else if (parts.length === 2) {
      // mm:ss
      seconds = parts[0] * 60 + parts[1];
    } else if (parts.length === 1) {
      // just seconds
      seconds = parts[0];
    }
    setCurrentTime(Math.max(0, Math.min(seconds, project?.duration || 60)));
  };

  // State for replace mode
  const [isReplaceMode, setIsReplaceMode] = useState(false);

  // Import media file (always adds new clip)
  const handleImportMedia = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file || !project) return;

    // Determine file type
    const isVideo = file.type.startsWith('video/');
    const isAudio = file.type.startsWith('audio/');
    const isImage = file.type.startsWith('image/');

    if (!isVideo && !isAudio && !isImage) {
      alert('Unsupported file type. Please select a video, audio, or image file.');
      return;
    }

    // Upload file to server
    const formData = new FormData();
    formData.append('file', file);

    try {
      const uploadRes = await fetch('/api/editor/upload', {
        method: 'POST',
        body: formData
      });
      const uploadData = await uploadRes.json();

      if (uploadData.success) {
        // Check if we're in replace mode
        if (isReplaceMode && selectedClipId) {
          const duration = uploadData.metadata?.duration || 5;
          await updateClip(selectedClipId, {
            source_path: uploadData.file_path,
            name: file.name,
            duration: duration,
          });
          // Update project duration if needed
          const selectedClip = clips.find(c => c.id === selectedClipId);
          if (selectedClip) {
            const clipEnd = selectedClip.start_time + duration;
            if (clipEnd > project.duration) {
              setProject(prev => prev ? { ...prev, duration: clipEnd + 10 } : prev);
            }
          }
          setIsReplaceMode(false);
        } else {
          // Always add new clip
          const trackType = isAudio ? 'audio' : 'video';
          const targetTrack = tracks.find(t => t.type === trackType);

          if (!targetTrack) {
            alert(`No ${trackType} track available. Create a project first.`);
            return;
          }

          const clipType = isVideo ? 'video' : isAudio ? 'audio' : 'image';
          const duration = uploadData.metadata?.duration || 5;

          const res = await fetch(`${API_BASE}/clip/add`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              type: clipType,
              name: file.name,
              source_path: uploadData.file_path,
              track_id: targetTrack.id,
              start_time: currentTime,
              duration: duration,
            })
          });
          const data = await res.json();
          if (data.success) {
            setClips(prev => [...prev, data.clip]);
            // Update project duration if needed
            const clipEnd = currentTime + duration;
            if (clipEnd > project.duration) {
              setProject(prev => prev ? { ...prev, duration: clipEnd + 10 } : prev);
            }
          } else {
            alert('Failed to add clip: ' + (data.error || 'Unknown error'));
          }
        }
      } else {
        alert('Failed to upload file: ' + (uploadData.error || 'Unknown error'));
      }
    } catch (e) {
      alert('Error uploading file: ' + e);
      console.error(e);
    }

    // Reset file input and replace mode
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    setIsReplaceMode(false);
  };

  const openImportDialog = () => {
    setIsReplaceMode(false);
    fileInputRef.current?.click();
  };

  const splitClip = async (clipId: string) => {
    const res = await fetch(`${API_BASE}/clip/${clipId}/split?split_time=${currentTime}`, {
      method: 'POST'
    });
    const data = await res.json();
    if (data.success) {
      setClips(prev => {
        const filtered = prev.filter(c => c.id !== clipId);
        return [...filtered, ...data.clips];
      });
    }
  };

  const addTrack = async (type: 'video' | 'audio') => {
    const name = type === 'video' ? `Video ${tracks.filter(t => t.type === 'video').length + 1}` :
                                    `Audio ${tracks.filter(t => t.type === 'audio').length + 1}`;
    const res = await fetch(`${API_BASE}/track/add?name=${name}&track_type=${type}`, {
      method: 'POST'
    });
    const data = await res.json();
    if (data.success) {
      setTracks(prev => [...prev, { ...data.track, height: TRACK_HEIGHT, visible: true, muted: false, locked: false }]);
    }
  };

  const exportVideo = async () => {
    setIsExporting(true);
    setExportProgress(0);

    try {
      const res = await fetch(`${API_BASE}/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          output_name: project?.name || 'export',
          format: 'mp4',
          quality: 'high'
        })
      });
      const data = await res.json();

      // Poll progress
      const pollProgress = async () => {
        const progressRes = await fetch(`${API_BASE}/export/progress`);
        const progressData = await progressRes.json();
        setExportProgress(progressData.progress);
        if (progressData.is_rendering) {
          setTimeout(pollProgress, 500);
        } else {
          setIsExporting(false);
          if (data.success) {
            alert(`Exported to: ${data.path}`);
          } else {
            alert(`Export failed: ${data.error || 'Unknown error'}`);
          }
        }
      };
      pollProgress();
    } catch (e) {
      setIsExporting(false);
      alert(`Export error: ${e}`);
    }
  };

  // Timeline interaction handlers
  const handleTimelineClick = (e: React.MouseEvent) => {
    if (!timelineRef.current) return;
    const rect = timelineRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left + scrollLeft;
    const time = x / pixelsPerSecond;
    setCurrentTime(Math.max(0, Math.min(time, project?.duration || 60)));
  };

  const handleClipMouseDown = (e: React.MouseEvent, clip: Clip) => {
    e.stopPropagation();
    setSelectedClipId(clip.id);
    setIsDragging(true);
    setDragClipId(clip.id);
    setDragStartX(e.clientX);
    setDragStartTime(clip.start_time);
  };

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!isDragging || !dragClipId) return;

    const deltaX = e.clientX - dragStartX;
    const deltaTime = deltaX / pixelsPerSecond;
    let newTime = Math.max(0, dragStartTime + deltaTime);

    // Snap to other clips
    if (snapEnabled) {
      const snapThreshold = 0.1; // seconds
      clips.forEach(c => {
        if (c.id !== dragClipId) {
          if (Math.abs(newTime - c.start_time) < snapThreshold) {
            newTime = c.start_time;
          }
          if (Math.abs(newTime - (c.start_time + c.duration)) < snapThreshold) {
            newTime = c.start_time + c.duration;
          }
        }
      });
    }

    setClips(prev => prev.map(c =>
      c.id === dragClipId ? { ...c, start_time: newTime } : c
    ));
  }, [isDragging, dragClipId, dragStartX, dragStartTime, pixelsPerSecond, snapEnabled, clips]);

  const handleMouseUp = useCallback(() => {
    if (isDragging && dragClipId) {
      const clip = clips.find(c => c.id === dragClipId);
      if (clip) {
        updateClip(dragClipId, { start_time: clip.start_time });
      }
    }
    setIsDragging(false);
    setDragClipId(null);
  }, [isDragging, dragClipId, clips]);

  useEffect(() => {
    if (isDragging) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
      return () => {
        window.removeEventListener('mousemove', handleMouseMove);
        window.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isDragging, handleMouseMove, handleMouseUp]);

  // Format time display
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    const frames = Math.floor((seconds % 1) * (project?.fps || 30));
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}:${frames.toString().padStart(2, '0')}`;
  };

  // Get clip color based on type
  const getClipColor = (type: string) => {
    const colors: Record<string, string> = {
      video: 'bg-blue-600',
      audio: 'bg-green-600',
      image: 'bg-purple-600',
      text: 'bg-yellow-600',
      shape: 'bg-pink-600',
    };
    return colors[type] || 'bg-gray-600';
  };

  const selectedClip = clips.find(c => c.id === selectedClipId);

  return (
    <div className="h-full flex flex-col bg-gray-950 text-gray-200">
      {/* Hidden file input for import */}
      <input
        ref={fileInputRef}
        type="file"
        accept="video/*,audio/*,image/*"
        onChange={handleImportMedia}
        className="hidden"
      />

      {/* Toolbar */}
      <div className="flex items-center gap-2 px-4 py-2 bg-gray-900 border-b border-gray-800">
        <button onClick={createNewProject} className="p-2 hover:bg-gray-800 rounded" title="New Project">
          <Plus className="w-4 h-4" />
        </button>
        <button onClick={openImportDialog} className="p-2 hover:bg-gray-800 rounded" title="Import Media">
          <FolderOpen className="w-4 h-4" />
        </button>
        <button className="p-2 hover:bg-gray-800 rounded" title="Save Project">
          <Save className="w-4 h-4" />
        </button>

        <div className="w-px h-6 bg-gray-700 mx-2" />

        <button
          onClick={undo}
          disabled={!canUndo}
          className="p-2 hover:bg-gray-800 rounded disabled:opacity-30 disabled:cursor-not-allowed"
          title="Undo (Ctrl+Z)"
        >
          <Undo2 className="w-4 h-4" />
        </button>
        <button
          onClick={redo}
          disabled={!canRedo}
          className="p-2 hover:bg-gray-800 rounded disabled:opacity-30 disabled:cursor-not-allowed"
          title="Redo (Ctrl+Y)"
        >
          <Redo2 className="w-4 h-4" />
        </button>

        <div className="w-px h-6 bg-gray-700 mx-2" />

        <button
          onClick={() => selectedClipId && splitClip(selectedClipId)}
          disabled={!selectedClipId}
          className="p-2 hover:bg-gray-800 rounded disabled:opacity-50"
          title="Split Clip (S)"
        >
          <Scissors className="w-4 h-4" />
        </button>
        <button
          onClick={() => selectedClipId && deleteClip(selectedClipId)}
          disabled={!selectedClipId}
          className="p-2 hover:bg-gray-800 rounded disabled:opacity-50 text-red-400"
          title="Delete Clip (Del)"
        >
          <Trash2 className="w-4 h-4" />
        </button>

        <div className="w-px h-6 bg-gray-700 mx-2" />

        <button
          onClick={() => setSnapEnabled(!snapEnabled)}
          className={`p-2 rounded ${snapEnabled ? 'bg-amber-600 text-black' : 'hover:bg-gray-800'}`}
          title="Snap to Clips"
        >
          <Magnet className="w-4 h-4" />
        </button>

        <div className="flex-1" />

        <div className="flex items-center gap-1">
          <button onClick={() => setZoom(Math.max(0.25, zoom - 0.25))} className="p-1 hover:bg-gray-800 rounded">
            <ZoomOut className="w-4 h-4" />
          </button>
          <span className="text-xs w-12 text-center">{Math.round(zoom * 100)}%</span>
          <button onClick={() => setZoom(Math.min(4, zoom + 0.25))} className="p-1 hover:bg-gray-800 rounded">
            <ZoomIn className="w-4 h-4" />
          </button>
        </div>

        <div className="w-px h-6 bg-gray-700 mx-2" />

        <button
          onClick={exportVideo}
          disabled={isExporting}
          className="px-3 py-1.5 bg-amber-600 hover:bg-amber-500 text-black rounded flex items-center gap-1.5 disabled:opacity-50"
        >
          <Download className="w-4 h-4" />
          {isExporting ? `${Math.round(exportProgress * 100)}%` : 'Export'}
        </button>
      </div>

      {/* Clip Info Bar */}
      {selectedClip && (
        <div className="flex items-center gap-3 px-4 py-2 bg-gray-800 border-b border-gray-700 text-sm">
          <span className="font-medium text-amber-400">{selectedClip.name || 'Untitled'}</span>
          <span className="text-gray-600">|</span>
          <span className="px-2 py-0.5 rounded bg-gray-700 text-xs uppercase">{selectedClip.type}</span>
          <span className="text-gray-600">|</span>
          <span className="text-gray-400">
            <span className="text-gray-500">Duration:</span> <span className="text-white font-mono">{selectedClip.duration.toFixed(2)}s</span>
          </span>
          <span className="text-gray-600">|</span>
          <span className="text-gray-400">
            <span className="text-gray-500">Start:</span> <span className="text-white font-mono">{selectedClip.start_time.toFixed(2)}s</span>
          </span>
          {selectedClip.scale !== 1 && (
            <>
              <span className="text-gray-600">|</span>
              <span className="text-gray-400">
                <span className="text-gray-500">Scale:</span> <span className="text-white">{Math.round((selectedClip.scale || 1) * 100)}%</span>
              </span>
            </>
          )}
          {selectedClip.effects && selectedClip.effects.length > 0 && (
            <>
              <span className="text-gray-600">|</span>
              <span className="text-purple-400">{selectedClip.effects.length} effect{selectedClip.effects.length > 1 ? 's' : ''}</span>
            </>
          )}
          {selectedClip.source_path && (
            <>
              <span className="text-gray-600">|</span>
              <span className="text-gray-400 truncate max-w-xs" title={selectedClip.source_path}>
                {selectedClip.source_path.split('/').pop()}
              </span>
            </>
          )}
          <div className="flex-1" />
          <button
            onClick={() => {
              setIsReplaceMode(true);
              fileInputRef.current?.click();
            }}
            className="px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded text-xs"
            title="Replace clip media"
          >
            Replace
          </button>
          <button
            onClick={() => setSelectedClipId(null)}
            className="px-2 py-1 bg-gray-600 hover:bg-gray-500 rounded text-xs ml-1"
            title="Deselect clip"
          >
            Deselect
          </button>
        </div>
      )}

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Preview panel */}
        <div className="w-1/2 flex flex-col border-r border-gray-800">
          {/* Preview */}
          <div className="flex-1 min-h-0 flex items-center justify-center bg-black">
            {previewFrame ? (
              <img
                src={previewFrame}
                alt="Preview"
                className="max-w-full max-h-full object-contain"
              />
            ) : (
              <div className="text-gray-600">No preview available</div>
            )}
          </div>

          {/* Playback controls */}
          <div className="flex-shrink-0 flex items-center gap-4 px-4 py-3 bg-gray-900 border-t border-gray-800">
            <button onClick={() => setCurrentTime(0)} className="p-2 hover:bg-gray-800 rounded">
              <SkipBack className="w-5 h-5" />
            </button>
            <button
              onClick={() => setIsPlaying(!isPlaying)}
              className="p-3 bg-amber-600 hover:bg-amber-500 text-black rounded-full"
            >
              {isPlaying ? <Pause className="w-6 h-6" /> : <Play className="w-6 h-6" />}
            </button>
            <button
              onClick={() => setCurrentTime(project?.duration || 60)}
              className="p-2 hover:bg-gray-800 rounded"
            >
              <SkipForward className="w-5 h-5" />
            </button>

            <span className="font-mono text-sm">{formatTime(currentTime)}</span>
            <span className="text-gray-500">/</span>
            <span className="font-mono text-sm text-gray-500">{formatTime(project?.duration || 0)}</span>

            {/* Jump to time input */}
            <div className="flex items-center gap-1 ml-2">
              <input
                type="text"
                placeholder="mm:ss"
                value={jumpTimeInput}
                onChange={e => setJumpTimeInput(e.target.value)}
                onKeyDown={e => {
                  if (e.key === 'Enter') {
                    jumpToTime(jumpTimeInput);
                    setJumpTimeInput('');
                  }
                }}
                className="w-16 px-2 py-1 bg-gray-800 border border-gray-700 rounded text-xs font-mono text-center"
                title="Jump to time (mm:ss or hh:mm:ss)"
              />
              <button
                onClick={() => {
                  jumpToTime(jumpTimeInput);
                  setJumpTimeInput('');
                  setIsPlaying(true);
                }}
                className="px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded text-xs"
                title="Go & Play"
              >
                Go
              </button>
            </div>

            {/* In/Out point controls */}
            <div className="flex items-center gap-1 ml-2 border-l border-gray-700 pl-2">
              <button
                onClick={() => setInPoint(currentTime)}
                className={`px-2 py-1 rounded text-xs ${inPoint !== null ? 'bg-green-700 text-white' : 'bg-gray-700 hover:bg-gray-600'}`}
                title="Set In Point (I)"
              >
                In{inPoint !== null && `: ${inPoint.toFixed(1)}s`}
              </button>
              <button
                onClick={() => setOutPoint(currentTime)}
                className={`px-2 py-1 rounded text-xs ${outPoint !== null ? 'bg-red-700 text-white' : 'bg-gray-700 hover:bg-gray-600'}`}
                title="Set Out Point (O)"
              >
                Out{outPoint !== null && `: ${outPoint.toFixed(1)}s`}
              </button>
              {inPoint !== null && outPoint !== null && outPoint > inPoint && (
                <>
                  <span className="text-xs text-gray-400">
                    {(outPoint - inPoint).toFixed(1)}s
                  </span>
                  <button
                    onClick={() => {
                      setCurrentTime(inPoint);
                      setIsPlaying(true);
                    }}
                    className="px-2 py-1 bg-amber-600 hover:bg-amber-500 text-black rounded text-xs"
                    title="Play Selection"
                  >
                    Play
                  </button>
                  <button
                    onClick={() => {
                      setInPoint(null);
                      setOutPoint(null);
                    }}
                    className="px-1 py-1 bg-gray-700 hover:bg-gray-600 rounded text-xs"
                    title="Clear Selection"
                  >
                    ✕
                  </button>
                </>
              )}
            </div>

            <div className="flex-1" />

            <button
              onClick={() => setIsMuted(!isMuted)}
              className="p-2 hover:bg-gray-800 rounded"
            >
              {isMuted ? <VolumeX className="w-5 h-5" /> : <Volume2 className="w-5 h-5" />}
            </button>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={isMuted ? 0 : volume}
              onChange={e => setVolume(Number(e.target.value))}
              className="w-20"
            />
          </div>
        </div>

        {/* Right panel - Clip properties / Effects */}
        <div className="w-1/2 flex flex-col">
          {/* Tabs */}
          <div className="flex bg-gray-900 border-b border-gray-800">
            <button
              onClick={() => setShowEffectsPanel(false)}
              className={`px-4 py-2 text-sm ${!showEffectsPanel ? 'bg-gray-800 text-amber-400' : 'text-gray-400'}`}
            >
              Properties
            </button>
            <button
              onClick={() => setShowEffectsPanel(true)}
              className={`px-4 py-2 text-sm ${showEffectsPanel ? 'bg-gray-800 text-amber-400' : 'text-gray-400'}`}
            >
              Effects
            </button>
          </div>

          {/* Panel content */}
          <div className="flex-1 overflow-y-auto p-4">
            {selectedClip ? (
              showEffectsPanel ? (
                // Effects panel
                <div className="space-y-4">
                  <h3 className="text-sm font-medium text-gray-300">Effects for: {selectedClip.name}</h3>

                  {/* Applied effects */}
                  {selectedClip.effects.length === 0 ? (
                    <p className="text-sm text-gray-500">No effects applied</p>
                  ) : (
                    selectedClip.effects.map(effect => {
                      const effectDef = AVAILABLE_EFFECTS.find(e => e.type === effect.type);
                      return (
                        <div key={effect.id} className="p-3 bg-gray-800 rounded">
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-2">
                              <input
                                type="checkbox"
                                checked={effect.enabled}
                                onChange={() => {
                                  const updatedEffects = selectedClip.effects.map(e =>
                                    e.id === effect.id ? { ...e, enabled: !e.enabled } : e
                                  );
                                  updateClip(selectedClip.id, { effects: updatedEffects });
                                }}
                                className="w-4 h-4"
                              />
                              <span className={`text-sm font-medium ${effect.enabled ? '' : 'text-gray-500'}`}>
                                {effectDef?.name || effect.type}
                              </span>
                            </div>
                            <button
                              onClick={() => {
                                const updatedEffects = selectedClip.effects.filter(e => e.id !== effect.id);
                                updateClip(selectedClip.id, { effects: updatedEffects });
                              }}
                              className="text-red-400 hover:text-red-300"
                            >
                              <Trash2 className="w-4 h-4" />
                            </button>
                          </div>

                          {/* Effect parameters */}
                          {effect.enabled && (
                            <div className="space-y-2 pl-6">
                              {Object.entries(effect.params).map(([key, value]) => (
                                <div key={key}>
                                  <label className="block text-xs text-gray-400 mb-1 capitalize">{key}</label>
                                  {typeof value === 'number' ? (
                                    <div className="flex items-center gap-2">
                                      <input
                                        type="range"
                                        min={key === 'rate' ? 0.1 : key.includes('radius') || key.includes('strength') ? 0 : -1}
                                        max={key === 'rate' ? 4 : key.includes('radius') || key.includes('strength') ? 20 : 2}
                                        step={key === 'rate' ? 0.1 : 0.05}
                                        value={value}
                                        onChange={e => {
                                          const updatedEffects = selectedClip.effects.map(ef =>
                                            ef.id === effect.id
                                              ? { ...ef, params: { ...ef.params, [key]: Number(e.target.value) } }
                                              : ef
                                          );
                                          updateClip(selectedClip.id, { effects: updatedEffects });
                                        }}
                                        className="flex-1"
                                      />
                                      <span className="text-xs w-12 text-right">{Number(value).toFixed(2)}</span>
                                    </div>
                                  ) : typeof value === 'string' && value.startsWith('#') ? (
                                    <input
                                      type="color"
                                      value={value}
                                      onChange={e => {
                                        const updatedEffects = selectedClip.effects.map(ef =>
                                          ef.id === effect.id
                                            ? { ...ef, params: { ...ef.params, [key]: e.target.value } }
                                            : ef
                                        );
                                        updateClip(selectedClip.id, { effects: updatedEffects });
                                      }}
                                      className="w-full h-6 bg-gray-700 border border-gray-600 rounded cursor-pointer"
                                    />
                                  ) : null}
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      );
                    })
                  )}

                  {/* Add effect dropdown */}
                  <div className="relative">
                    <select
                      onChange={e => {
                        if (e.target.value) {
                          const effectDef = AVAILABLE_EFFECTS.find(ef => ef.type === e.target.value);
                          if (effectDef) {
                            const newEffect: Effect = {
                              id: `effect-${Date.now()}`,
                              type: effectDef.type,
                              enabled: true,
                              params: { ...effectDef.defaultParams },
                            };
                            updateClip(selectedClip.id, {
                              effects: [...selectedClip.effects, newEffect],
                            });
                          }
                          e.target.value = '';
                        }
                      }}
                      className="w-full py-2 px-3 bg-gray-800 border border-dashed border-gray-600 rounded text-sm text-gray-400 hover:border-gray-500 hover:text-gray-300"
                      defaultValue=""
                    >
                      <option value="">+ Add Effect...</option>
                      <optgroup label="Color">
                        {AVAILABLE_EFFECTS.filter(e => e.category === 'color').map(e => (
                          <option key={e.type} value={e.type}>{e.name}</option>
                        ))}
                      </optgroup>
                      <optgroup label="Stylize">
                        {AVAILABLE_EFFECTS.filter(e => e.category === 'stylize').map(e => (
                          <option key={e.type} value={e.type}>{e.name}</option>
                        ))}
                      </optgroup>
                      <optgroup label="Utility">
                        {AVAILABLE_EFFECTS.filter(e => e.category === 'utility').map(e => (
                          <option key={e.type} value={e.type}>{e.name}</option>
                        ))}
                      </optgroup>
                    </select>
                  </div>

                  {/* Transitions section */}
                  <div className="pt-4 border-t border-gray-700">
                    <h4 className="text-xs text-gray-500 uppercase tracking-wide mb-2">Transitions</h4>
                    <div className="grid grid-cols-2 gap-2">
                      <div>
                        <label className="block text-xs text-gray-400 mb-1">In</label>
                        <select
                          value={selectedClip.transition_in?.type || 'none'}
                          onChange={e => {
                            const type = e.target.value;
                            updateClip(selectedClip.id, {
                              transition_in: type === 'none' ? undefined : { type, duration: 0.5 },
                            });
                          }}
                          className="w-full px-2 py-1 bg-gray-800 border border-gray-700 rounded text-sm"
                        >
                          <option value="none">None</option>
                          {AVAILABLE_TRANSITIONS.map(t => (
                            <option key={t.type} value={t.type}>{t.name}</option>
                          ))}
                        </select>
                      </div>
                      <div>
                        <label className="block text-xs text-gray-400 mb-1">Out</label>
                        <select
                          value={selectedClip.transition_out?.type || 'none'}
                          onChange={e => {
                            const type = e.target.value;
                            updateClip(selectedClip.id, {
                              transition_out: type === 'none' ? undefined : { type, duration: 0.5 },
                            });
                          }}
                          className="w-full px-2 py-1 bg-gray-800 border border-gray-700 rounded text-sm"
                        >
                          <option value="none">None</option>
                          {AVAILABLE_TRANSITIONS.map(t => (
                            <option key={t.type} value={t.type}>{t.name}</option>
                          ))}
                        </select>
                      </div>
                    </div>
                    {(selectedClip.transition_in || selectedClip.transition_out) && (
                      <div className="mt-2">
                        <label className="block text-xs text-gray-400 mb-1">Transition Duration</label>
                        <input
                          type="range"
                          min="0.1"
                          max="2"
                          step="0.1"
                          value={selectedClip.transition_in?.duration || selectedClip.transition_out?.duration || 0.5}
                          onChange={e => {
                            const duration = Number(e.target.value);
                            updateClip(selectedClip.id, {
                              transition_in: selectedClip.transition_in ? { ...selectedClip.transition_in, duration } : undefined,
                              transition_out: selectedClip.transition_out ? { ...selectedClip.transition_out, duration } : undefined,
                            });
                          }}
                          className="w-full"
                        />
                        <span className="text-xs text-gray-500">
                          {(selectedClip.transition_in?.duration || selectedClip.transition_out?.duration || 0.5).toFixed(1)}s
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              ) : (
                // Properties panel
                <div className="space-y-4">
                  <h3 className="text-sm font-medium text-gray-300">Clip Properties</h3>
                  <div className="space-y-3">
                    <div>
                      <label className="block text-xs text-gray-400 mb-1">Name</label>
                      <input
                        type="text"
                        value={selectedClip.name}
                        onChange={e => updateClip(selectedClip.id, { name: e.target.value })}
                        className="w-full px-2 py-1 bg-gray-800 border border-gray-700 rounded text-sm"
                      />
                    </div>

                    {/* Timeline */}
                    <div className="pt-2 border-t border-gray-700">
                      <label className="block text-xs text-gray-500 mb-2 uppercase tracking-wide">Timeline</label>
                      <div className="grid grid-cols-2 gap-2">
                        <div>
                          <label className="block text-xs text-gray-400 mb-1">Start Time</label>
                          <input
                            type="number"
                            value={selectedClip.start_time.toFixed(2)}
                            onChange={e => updateClip(selectedClip.id, { start_time: Number(e.target.value) })}
                            className="w-full px-2 py-1 bg-gray-800 border border-gray-700 rounded text-sm"
                            step="0.1"
                          />
                        </div>
                        <div>
                          <label className="block text-xs text-gray-400 mb-1">Duration</label>
                          <input
                            type="number"
                            value={selectedClip.duration.toFixed(2)}
                            readOnly
                            className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-sm text-gray-400"
                          />
                        </div>
                      </div>
                      {(selectedClip.type === 'video' || selectedClip.type === 'audio') && (
                        <div className="grid grid-cols-2 gap-2 mt-2">
                          <div>
                            <label className="block text-xs text-gray-400 mb-1">Trim In</label>
                            <input
                              type="number"
                              value={selectedClip.source_in?.toFixed(2) || 0}
                              onChange={e => updateClip(selectedClip.id, { source_in: Number(e.target.value) })}
                              className="w-full px-2 py-1 bg-gray-800 border border-gray-700 rounded text-sm"
                              step="0.1"
                              min="0"
                            />
                          </div>
                          <div>
                            <label className="block text-xs text-gray-400 mb-1">Trim Out</label>
                            <input
                              type="number"
                              value={selectedClip.source_out?.toFixed(2) || selectedClip.duration}
                              onChange={e => updateClip(selectedClip.id, { source_out: Number(e.target.value) })}
                              className="w-full px-2 py-1 bg-gray-800 border border-gray-700 rounded text-sm"
                              step="0.1"
                            />
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Transform */}
                    {(selectedClip.type === 'video' || selectedClip.type === 'image') && (
                      <div className="pt-2 border-t border-gray-700">
                        <label className="block text-xs text-gray-500 mb-2 uppercase tracking-wide">Transform</label>
                        <div className="grid grid-cols-2 gap-2">
                          <div>
                            <label className="block text-xs text-gray-400 mb-1">Position X</label>
                            <input
                              type="number"
                              value={selectedClip.position_x || 0}
                              onChange={e => updateClip(selectedClip.id, { position_x: Number(e.target.value) })}
                              className="w-full px-2 py-1 bg-gray-800 border border-gray-700 rounded text-sm"
                              step="1"
                            />
                          </div>
                          <div>
                            <label className="block text-xs text-gray-400 mb-1">Position Y</label>
                            <input
                              type="number"
                              value={selectedClip.position_y || 0}
                              onChange={e => updateClip(selectedClip.id, { position_y: Number(e.target.value) })}
                              className="w-full px-2 py-1 bg-gray-800 border border-gray-700 rounded text-sm"
                              step="1"
                            />
                          </div>
                        </div>
                        <div className="grid grid-cols-2 gap-2 mt-2">
                          <div>
                            <label className="block text-xs text-gray-400 mb-1">Scale %</label>
                            <input
                              type="number"
                              value={Math.round((selectedClip.scale || 1) * 100)}
                              onChange={e => updateClip(selectedClip.id, { scale: Number(e.target.value) / 100 })}
                              className="w-full px-2 py-1 bg-gray-800 border border-gray-700 rounded text-sm"
                              step="1"
                              min="1"
                              max="500"
                            />
                          </div>
                          <div>
                            <label className="block text-xs text-gray-400 mb-1">Rotation °</label>
                            <input
                              type="number"
                              value={selectedClip.rotation || 0}
                              onChange={e => updateClip(selectedClip.id, { rotation: Number(e.target.value) })}
                              className="w-full px-2 py-1 bg-gray-800 border border-gray-700 rounded text-sm"
                              step="1"
                            />
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Opacity & Audio */}
                    <div className="pt-2 border-t border-gray-700">
                      <label className="block text-xs text-gray-500 mb-2 uppercase tracking-wide">Mix</label>
                      <div>
                        <label className="block text-xs text-gray-400 mb-1">Opacity: {Math.round((selectedClip.opacity || 1) * 100)}%</label>
                        <input
                          type="range"
                          min="0"
                          max="1"
                          step="0.05"
                          value={selectedClip.opacity || 1}
                          onChange={e => updateClip(selectedClip.id, { opacity: Number(e.target.value) })}
                          className="w-full"
                        />
                      </div>
                      {(selectedClip.type === 'video' || selectedClip.type === 'audio') && (
                        <div className="mt-2">
                          <label className="block text-xs text-gray-400 mb-1">Volume: {Math.round((selectedClip.volume || 1) * 100)}%</label>
                          <input
                            type="range"
                            min="0"
                            max="2"
                            step="0.05"
                            value={selectedClip.volume || 1}
                            onChange={e => updateClip(selectedClip.id, { volume: Number(e.target.value) })}
                            className="w-full"
                          />
                        </div>
                      )}
                    </div>

                    {/* Text Properties */}
                    {selectedClip.type === 'text' && (
                      <div className="pt-2 border-t border-gray-700">
                        <label className="block text-xs text-gray-500 mb-2 uppercase tracking-wide">Text</label>
                        <div>
                          <label className="block text-xs text-gray-400 mb-1">Content</label>
                          <textarea
                            value={selectedClip.text_content || ''}
                            onChange={e => updateClip(selectedClip.id, { text_content: e.target.value })}
                            className="w-full px-2 py-1 bg-gray-800 border border-gray-700 rounded text-sm h-20"
                          />
                        </div>
                        <div className="grid grid-cols-2 gap-2 mt-2">
                          <div>
                            <label className="block text-xs text-gray-400 mb-1">Font Size</label>
                            <input
                              type="number"
                              value={selectedClip.font_size || 48}
                              onChange={e => updateClip(selectedClip.id, { font_size: Number(e.target.value) })}
                              className="w-full px-2 py-1 bg-gray-800 border border-gray-700 rounded text-sm"
                            />
                          </div>
                          <div>
                            <label className="block text-xs text-gray-400 mb-1">Color</label>
                            <input
                              type="color"
                              value={selectedClip.font_color || '#FFFFFF'}
                              onChange={e => updateClip(selectedClip.id, { font_color: e.target.value })}
                              className="w-full h-8 bg-gray-800 border border-gray-700 rounded cursor-pointer"
                            />
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )
            ) : (
              <div className="text-center text-gray-500 py-8">
                Select a clip to edit properties
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Timeline */}
      <div className="h-64 flex flex-col bg-gray-900 border-t border-gray-800">
        {/* Timeline header */}
        <div className="flex border-b border-gray-700">
          {/* Track controls header */}
          <div className="w-48 flex-shrink-0 bg-gray-850 border-r border-gray-700 flex items-center px-2">
            <button
              onClick={() => addTrack('video')}
              className="p-1 hover:bg-gray-700 rounded text-blue-400"
              title="Add Video Track"
            >
              <Film className="w-4 h-4" />
            </button>
            <button
              onClick={() => addTrack('audio')}
              className="p-1 hover:bg-gray-700 rounded text-green-400 ml-1"
              title="Add Audio Track"
            >
              <Music className="w-4 h-4" />
            </button>
          </div>

          {/* Time ruler */}
          <div
            ref={timelineRulerRef}
            className="flex-1 overflow-x-auto overflow-y-hidden scrollbar-hide"
            style={{ height: TIMELINE_HEADER_HEIGHT }}
            onClick={handleTimelineClick}
          >
            <div
              className="relative h-full"
              style={{ width: (project?.duration || 60) * pixelsPerSecond, minWidth: '100%' }}
            >
              {/* Time markers */}
              {Array.from({ length: Math.ceil((project?.duration || 60) + 1) }, (_, i) => (
                <div
                  key={i}
                  className="absolute top-0 h-full border-l border-gray-700 text-xs text-gray-500 pl-1"
                  style={{ left: i * pixelsPerSecond }}
                >
                  {i}s
                </div>
              ))}

              {/* Playhead in ruler */}
              <div
                className="absolute top-0 h-full w-0.5 bg-red-500 z-10"
                style={{ left: currentTime * pixelsPerSecond }}
              />
            </div>
          </div>
        </div>

        {/* Tracks */}
        <div className="flex-1 flex overflow-hidden">
          {/* Track controls */}
          <div className="w-48 flex-shrink-0 bg-gray-850 border-r border-gray-700 overflow-y-auto">
            {tracks.map(track => (
              <div
                key={track.id}
                className={`flex items-center gap-1 px-2 border-b border-gray-700 ${
                  selectedTrackId === track.id ? 'bg-gray-800' : ''
                }`}
                style={{ height: TRACK_HEIGHT }}
                onClick={() => setSelectedTrackId(track.id)}
              >
                {track.type === 'video' ? (
                  <Film className="w-4 h-4 text-blue-400" />
                ) : (
                  <Music className="w-4 h-4 text-green-400" />
                )}
                <span className="text-xs flex-1 truncate">{track.name}</span>
                <button
                  onClick={e => { e.stopPropagation(); /* toggle mute */ }}
                  className="p-1 hover:bg-gray-700 rounded"
                >
                  {track.muted ? <VolumeX className="w-3 h-3" /> : <Volume2 className="w-3 h-3" />}
                </button>
                <button
                  onClick={e => { e.stopPropagation(); /* toggle visibility */ }}
                  className="p-1 hover:bg-gray-700 rounded"
                >
                  {track.visible ? <Eye className="w-3 h-3" /> : <EyeOff className="w-3 h-3" />}
                </button>
              </div>
            ))}
          </div>

          {/* Timeline content */}
          <div
            ref={timelineRef}
            className="flex-1 overflow-auto relative"
            onClick={handleTimelineClick}
            onScroll={handleTimelineScroll}
            onWheel={handleTimelineWheel}
          >
            <div
              className="relative"
              style={{
                width: (project?.duration || 60) * pixelsPerSecond,
                height: tracks.length * TRACK_HEIGHT,
              }}
            >
              {/* Track backgrounds */}
              {tracks.map((track, i) => (
                <div
                  key={track.id}
                  className={`absolute left-0 right-0 border-b border-gray-700 ${
                    i % 2 === 0 ? 'bg-gray-850' : 'bg-gray-900'
                  }`}
                  style={{ top: i * TRACK_HEIGHT, height: TRACK_HEIGHT }}
                  onDoubleClick={() => addClip('video', track.id, currentTime)}
                />
              ))}

              {/* In/Out Selection region */}
              {inPoint !== null && outPoint !== null && outPoint > inPoint && (
                <div
                  className="absolute top-0 bottom-0 bg-amber-500/20 border-l-2 border-r-2 border-amber-500 pointer-events-none z-5"
                  style={{
                    left: inPoint * pixelsPerSecond,
                    width: (outPoint - inPoint) * pixelsPerSecond,
                  }}
                >
                  <div className="absolute -top-0 left-0 bg-green-600 text-white text-xs px-1 rounded-br">IN</div>
                  <div className="absolute -top-0 right-0 bg-red-600 text-white text-xs px-1 rounded-bl">OUT</div>
                </div>
              )}

              {/* Clips */}
              {clips.map(clip => {
                const trackIndex = tracks.findIndex(t => t.id === clip.track_id);
                if (trackIndex === -1) return null;

                return (
                  <div
                    key={clip.id}
                    className={`absolute rounded cursor-pointer ${getClipColor(clip.type)} ${
                      selectedClipId === clip.id ? 'ring-2 ring-amber-400' : ''
                    }`}
                    style={{
                      left: clip.start_time * pixelsPerSecond,
                      width: clip.duration * pixelsPerSecond,
                      top: trackIndex * TRACK_HEIGHT + 4,
                      height: TRACK_HEIGHT - 8,
                    }}
                    onClick={e => { e.stopPropagation(); setSelectedClipId(clip.id); }}
                    onMouseDown={e => handleClipMouseDown(e, clip)}
                  >
                    <div className="px-2 py-1 text-xs truncate text-white font-medium">
                      {clip.name || clip.type}
                    </div>
                    {clip.type === 'text' && (
                      <div className="px-2 text-xs truncate text-white/70">
                        {clip.text_content}
                      </div>
                    )}
                  </div>
                );
              })}

              {/* Playhead */}
              <div
                className="absolute top-0 bottom-0 w-0.5 bg-red-500 z-20 pointer-events-none"
                style={{ left: currentTime * pixelsPerSecond }}
              >
                <div className="absolute -top-1 -left-2 w-4 h-4 bg-red-500 rotate-45 transform origin-center" />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default VideoEditor;
