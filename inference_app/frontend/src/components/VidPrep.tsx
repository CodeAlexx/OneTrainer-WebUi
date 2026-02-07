import { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import {
  Play, Pause, Scissors, ChevronDown, ChevronRight,
  RefreshCw, Download, Trash2, Plus, Settings, Film, Maximize2,
  Check, Loader2, SkipBack, SkipForward, Crop, FileVideo, FileImage
} from 'lucide-react';

const API = axios.create({ baseURL: '/api' });

// Model presets with their native FPS and recommended settings
const MODEL_PRESETS = {
  wan: {
    name: 'Wan 2.1/2.2',
    fps: 16,
    resolutions: [
      { label: 'Low (480x272)', width: 480, height: 272, frames: 65 },
      { label: 'Medium (640x360)', width: 640, height: 360, frames: 37 },
      { label: 'High (848x480)', width: 848, height: 480, frames: 21 },
      { label: '720p (1280x720)', width: 1280, height: 720, frames: 17 },
    ],
    frameRule: 'N*4+1',
    validFrames: [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81],
  },
  hunyuan: {
    name: 'HunyuanVideo',
    fps: 24,
    resolutions: [
      { label: 'Low (480x270)', width: 480, height: 270, frames: 49 },
      { label: 'Medium (640x360)', width: 640, height: 360, frames: 97 },
      { label: 'Paper (960x544)', width: 960, height: 544, frames: 129 },
    ],
    frameRule: 'N*4+1',
    validFrames: [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125, 129, 133, 137, 141, 145],
  },
  framepack: {
    name: 'FramePack',
    fps: 30,
    resolutions: [
      { label: 'Standard (512x512)', width: 512, height: 512, frames: 25 },
      { label: 'Wide (768x432)', width: 768, height: 432, frames: 25 },
    ],
    frameRule: 'flexible',
    validFrames: [9, 13, 17, 21, 25, 33, 41, 49],
  },
};

interface VideoFile {
  name: string;
  path: string;
  duration: number;
  fps: number;
  width: number;
  height: number;
  frames: number;
  size: number;
  thumbnail?: string;
}

interface CropRegion {
  x: number;
  y: number;
  width: number;
  height: number;
}

interface VideoRange {
  id: string;
  start: number;
  end: number;
  caption: string;
  crop?: CropRegion;
  useCrop: boolean;
}

interface ExportSettings {
  exportCropped: boolean;
  exportUncropped: boolean;
  exportFirstFrame: boolean;
  maxLongestEdge: number;
  useMaxEdge: boolean;
}

interface ProcessingJob {
  id: string;
  video: string;
  rangeId: string;
  status: 'pending' | 'processing' | 'done' | 'error';
  progress: number;
  error?: string;
  outputPath?: string;
}

export function VidPrep() {
  // Folder state
  const [inputFolder, setInputFolder] = useState('');
  const [outputFolder, setOutputFolder] = useState('');
  const [videos, setVideos] = useState<VideoFile[]>([]);
  const [selectedVideo, setSelectedVideo] = useState<VideoFile | null>(null);
  const [isScanning, setIsScanning] = useState(false);

  // Model preset
  const [modelPreset, setModelPreset] = useState<keyof typeof MODEL_PRESETS>('wan');
  const preset = MODEL_PRESETS[modelPreset];

  // Processing settings
  const [targetFps, setTargetFps] = useState(preset.fps);
  const [targetWidth, setTargetWidth] = useState(preset.resolutions[1].width);
  const [targetHeight, setTargetHeight] = useState(preset.resolutions[1].height);
  const [targetFrames, setTargetFrames] = useState(preset.resolutions[1].frames);
  const [enableBucket, setEnableBucket] = useState(true);
  const [bucketNoUpscale, setBucketNoUpscale] = useState(true);

  // Export settings
  const [exportSettings, setExportSettings] = useState<ExportSettings>({
    exportCropped: true,
    exportUncropped: false,
    exportFirstFrame: false,
    maxLongestEdge: 1280,
    useMaxEdge: false,
  });

  // Range editing - multiple ranges per video
  const [ranges, setRanges] = useState<VideoRange[]>([]);
  const [selectedRangeId, setSelectedRangeId] = useState<string | null>(null);
  const selectedRange = ranges.find(r => r.id === selectedRangeId) || null;

  // Video playback
  const [previewTime, setPreviewTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const videoRef = useRef<HTMLVideoElement>(null);
  const playbackRef = useRef<number | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Crop tool state
  const [showCropTool, setShowCropTool] = useState(false);
  const [cropDrag, setCropDrag] = useState<{ mode: 'move' | 'resize' | null; startX: number; startY: number; startCrop: CropRegion } | null>(null);
  const [tempCrop, setTempCrop] = useState<CropRegion | null>(null);

  // Processing
  const [jobs, setJobs] = useState<ProcessingJob[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);

  // UI state
  const [showSettings, setShowSettings] = useState(false);
  const [expandedSection, setExpandedSection] = useState<string>('videos');

  // Update settings when preset changes
  useEffect(() => {
    const p = MODEL_PRESETS[modelPreset];
    setTargetFps(p.fps);
    setTargetWidth(p.resolutions[1].width);
    setTargetHeight(p.resolutions[1].height);
    setTargetFrames(p.resolutions[1].frames);
  }, [modelPreset]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

      switch (e.key) {
        case ' ':
          e.preventDefault();
          setIsPlaying(p => !p);
          break;
        case 'j':
          // Reverse play / slow down
          setPlaybackSpeed(s => Math.max(-2, s - 0.5));
          break;
        case 'k':
          // Pause
          setIsPlaying(false);
          break;
        case 'l':
          // Forward play / speed up
          setPlaybackSpeed(s => Math.min(2, s + 0.5));
          setIsPlaying(true);
          break;
        case 'ArrowLeft':
          e.preventDefault();
          // Step back 1 frame
          if (selectedVideo && videoRef.current) {
            const frameDuration = 1 / selectedVideo.fps;
            seekTo(Math.max(0, previewTime - frameDuration));
          }
          break;
        case 'ArrowRight':
          e.preventDefault();
          // Step forward 1 frame
          if (selectedVideo && videoRef.current) {
            const frameDuration = 1 / selectedVideo.fps;
            seekTo(Math.min(selectedVideo.duration, previewTime + frameDuration));
          }
          break;
        case 'i':
        case 'I':
          // Set in point (start of range)
          if (selectedRange) {
            updateRange(selectedRange.id, { start: previewTime });
          }
          break;
        case 'o':
        case 'O':
          // Set out point (end of range)
          if (selectedRange) {
            updateRange(selectedRange.id, { end: previewTime });
          }
          break;
        case 'n':
        case 'N':
          // Add new range
          addRange();
          break;
        case 'Delete':
        case 'Backspace':
          // Delete selected range
          if (selectedRange && !e.target) {
            removeRange(selectedRange.id);
          }
          break;
        case 'Home':
          // Go to start
          seekTo(0);
          break;
        case 'End':
          // Go to end
          if (selectedVideo) seekTo(selectedVideo.duration);
          break;
        case '[':
          // Go to in point
          if (selectedRange) seekTo(selectedRange.start);
          break;
        case ']':
          // Go to out point
          if (selectedRange) seekTo(selectedRange.end);
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedVideo, selectedRange, previewTime]);

  // Scan folder for videos
  const scanFolder = async () => {
    if (!inputFolder) return;
    setIsScanning(true);
    try {
      const r = await API.post('/vidprep/scan', { folder: inputFolder });
      setVideos(r.data.videos || []);
      if (r.data.videos?.length > 0) {
        setSelectedVideo(r.data.videos[0]);
        setRanges([]);
        setSelectedRangeId(null);
      }
    } catch (e: any) {
      console.error('Scan failed:', e);
    } finally {
      setIsScanning(false);
    }
  };

  // Video playback
  useEffect(() => {
    if (isPlaying && videoRef.current && selectedVideo) {
      videoRef.current.playbackRate = Math.abs(playbackSpeed);
      const interval = window.setInterval(() => {
        if (videoRef.current) {
          const newTime = playbackSpeed >= 0
            ? videoRef.current.currentTime
            : Math.max(0, videoRef.current.currentTime - (2 * Math.abs(playbackSpeed) / 30));

          setPreviewTime(newTime);

          if (videoRef.current.ended || newTime <= 0) {
            setIsPlaying(false);
          }
        }
      }, 1000 / 30);
      playbackRef.current = interval;

      if (playbackSpeed >= 0) {
        videoRef.current.play();
      }

      return () => {
        clearInterval(interval);
        playbackRef.current = null;
      };
    } else if (!isPlaying && videoRef.current) {
      videoRef.current.pause();
      if (playbackRef.current) {
        clearInterval(playbackRef.current);
        playbackRef.current = null;
      }
    }
  }, [isPlaying, selectedVideo, playbackSpeed]);

  // Seek video
  const seekTo = useCallback((time: number) => {
    if (videoRef.current && selectedVideo) {
      const clampedTime = Math.max(0, Math.min(selectedVideo.duration, time));
      videoRef.current.currentTime = clampedTime;
      setPreviewTime(clampedTime);
    }
  }, [selectedVideo]);

  // Add range from current position
  const addRange = useCallback(() => {
    if (!selectedVideo) return;
    const start = previewTime;
    const duration = targetFrames / targetFps;
    const end = Math.min(start + duration, selectedVideo.duration);
    const newRange: VideoRange = {
      id: `range-${Date.now()}`,
      start,
      end,
      caption: '',
      useCrop: false,
    };
    setRanges(prev => [...prev, newRange]);
    setSelectedRangeId(newRange.id);
  }, [selectedVideo, previewTime, targetFrames, targetFps]);

  // Update range
  const updateRange = (id: string, updates: Partial<VideoRange>) => {
    setRanges(ranges.map(r => r.id === id ? { ...r, ...updates } : r));
  };

  // Remove range
  const removeRange = (id: string) => {
    setRanges(ranges.filter(r => r.id !== id));
    if (selectedRangeId === id) {
      const remaining = ranges.filter(r => r.id !== id);
      setSelectedRangeId(remaining.length > 0 ? remaining[0].id : null);
    }
  };

  // Initialize crop for selected range
  const initCrop = () => {
    if (!selectedVideo || !selectedRange) return;

    const crop: CropRegion = selectedRange.crop || {
      x: 0,
      y: 0,
      width: selectedVideo.width,
      height: selectedVideo.height,
    };
    setTempCrop(crop);
    setShowCropTool(true);
  };

  // Apply crop to range
  const applyCrop = () => {
    if (tempCrop && selectedRange) {
      updateRange(selectedRange.id, { crop: tempCrop, useCrop: true });
    }
    setShowCropTool(false);
    setTempCrop(null);
  };

  // Cancel crop
  const cancelCrop = () => {
    setShowCropTool(false);
    setTempCrop(null);
  };

  // Handle crop drag
  const handleCropMouseDown = (e: React.MouseEvent, mode: 'move' | 'resize') => {
    if (!tempCrop) return;
    e.preventDefault();
    setCropDrag({
      mode,
      startX: e.clientX,
      startY: e.clientY,
      startCrop: { ...tempCrop },
    });
  };

  const handleCropMouseMove = useCallback((e: MouseEvent) => {
    if (!cropDrag || !tempCrop || !videoRef.current || !selectedVideo) return;

    const videoRect = videoRef.current.getBoundingClientRect();
    const scaleX = selectedVideo.width / videoRect.width;
    const scaleY = selectedVideo.height / videoRect.height;

    const dx = (e.clientX - cropDrag.startX) * scaleX;
    const dy = (e.clientY - cropDrag.startY) * scaleY;

    if (cropDrag.mode === 'move') {
      setTempCrop({
        ...tempCrop,
        x: Math.max(0, Math.min(selectedVideo.width - tempCrop.width, cropDrag.startCrop.x + dx)),
        y: Math.max(0, Math.min(selectedVideo.height - tempCrop.height, cropDrag.startCrop.y + dy)),
      });
    } else if (cropDrag.mode === 'resize') {
      const newWidth = Math.max(64, Math.min(selectedVideo.width - tempCrop.x, cropDrag.startCrop.width + dx));
      const newHeight = Math.max(64, Math.min(selectedVideo.height - tempCrop.y, cropDrag.startCrop.height + dy));
      setTempCrop({
        ...tempCrop,
        width: newWidth,
        height: newHeight,
      });
    }
  }, [cropDrag, tempCrop, selectedVideo]);

  const handleCropMouseUp = useCallback(() => {
    setCropDrag(null);
  }, []);

  useEffect(() => {
    if (cropDrag) {
      window.addEventListener('mousemove', handleCropMouseMove);
      window.addEventListener('mouseup', handleCropMouseUp);
      return () => {
        window.removeEventListener('mousemove', handleCropMouseMove);
        window.removeEventListener('mouseup', handleCropMouseUp);
      };
    }
  }, [cropDrag, handleCropMouseMove, handleCropMouseUp]);

  // Set crop to match target aspect ratio
  const setCropToAspect = () => {
    if (!selectedVideo || !tempCrop) return;

    const targetAspect = targetWidth / targetHeight;
    const videoAspect = selectedVideo.width / selectedVideo.height;

    let cropWidth: number, cropHeight: number;

    if (targetAspect > videoAspect) {
      // Target is wider - constrain by width
      cropWidth = selectedVideo.width;
      cropHeight = cropWidth / targetAspect;
    } else {
      // Target is taller - constrain by height
      cropHeight = selectedVideo.height;
      cropWidth = cropHeight * targetAspect;
    }

    setTempCrop({
      x: (selectedVideo.width - cropWidth) / 2,
      y: (selectedVideo.height - cropHeight) / 2,
      width: cropWidth,
      height: cropHeight,
    });
  };

  // Process videos
  const processVideos = async () => {
    if (ranges.length === 0) {
      alert('Add at least one range to process');
      return;
    }
    if (!outputFolder) {
      alert('Set output folder first');
      return;
    }

    setIsProcessing(true);
    const newJobs: ProcessingJob[] = ranges.map(r => ({
      id: `job-${r.id}`,
      video: selectedVideo?.name || '',
      rangeId: r.id,
      status: 'pending',
      progress: 0,
    }));
    setJobs(newJobs);

    try {
      const response = await API.post('/vidprep/process', {
        input_folder: inputFolder,
        output_folder: outputFolder,
        video_path: selectedVideo?.path,
        ranges: ranges.map(r => ({
          id: r.id,
          start: r.start,
          end: r.end,
          caption: r.caption,
          crop: r.useCrop ? r.crop : null,
        })),
        settings: {
          target_fps: targetFps,
          target_width: targetWidth,
          target_height: targetHeight,
          target_frames: targetFrames,
          enable_bucket: enableBucket,
          bucket_no_upscale: bucketNoUpscale,
          export_cropped: exportSettings.exportCropped,
          export_uncropped: exportSettings.exportUncropped,
          export_first_frame: exportSettings.exportFirstFrame,
          max_longest_edge: exportSettings.useMaxEdge ? exportSettings.maxLongestEdge : null,
        },
      });

      // Update jobs with results
      if (response.data.results) {
        setJobs(jobs.map(j => {
          const result = response.data.results.find((r: any) => r.range_id === j.rangeId);
          if (result) {
            return {
              ...j,
              status: result.success ? 'done' : 'error',
              progress: 100,
              error: result.error,
              outputPath: result.output_path,
            };
          }
          return j;
        }));
      }
    } catch (e: any) {
      console.error('Processing failed:', e);
      setJobs(jobs.map(j => ({ ...j, status: 'error', error: e.message })));
    } finally {
      setIsProcessing(false);
    }
  };

  // Generate TOML config
  const generateConfig = () => {
    const toml = `# Dataset config for ${MODEL_PRESETS[modelPreset].name}
# Generated by OneTrainer VidPrep

[general]
resolution = [${targetWidth}, ${targetHeight}]
enable_bucket = ${enableBucket}
bucket_no_upscale = ${bucketNoUpscale}

[[datasets]]
video_directory = "${outputFolder}"
caption_extension = ".txt"
target_frames = [${targetFrames}]
frame_extraction = "full"
source_fps = ${targetFps}.0
`;
    const blob = new Blob([toml], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'dataset_config.toml';
    a.click();
    URL.revokeObjectURL(url);
  };

  const formatTime = (t: number) => {
    const mins = Math.floor(t / 60);
    const secs = Math.floor(t % 60);
    const ms = Math.floor((t % 1) * 1000);
    return `${mins}:${secs.toString().padStart(2, '0')}.${ms.toString().padStart(3, '0')}`;
  };

  const formatTimeWithFrames = (t: number, fps: number) => {
    const mins = Math.floor(t / 60);
    const secs = Math.floor(t % 60);
    const frame = Math.floor((t % 1) * fps);
    return `${mins}:${secs.toString().padStart(2, '0')}:${frame.toString().padStart(2, '0')}`;
  };

  // Calculate crop overlay position for display
  const getCropOverlayStyle = () => {
    if (!tempCrop || !videoRef.current || !selectedVideo) return {};

    const videoRect = videoRef.current.getBoundingClientRect();
    const scaleX = videoRect.width / selectedVideo.width;
    const scaleY = videoRect.height / selectedVideo.height;

    return {
      left: tempCrop.x * scaleX,
      top: tempCrop.y * scaleY,
      width: tempCrop.width * scaleX,
      height: tempCrop.height * scaleY,
    };
  };

  return (
    <div className="flex-1 flex flex-col bg-gray-950 overflow-hidden">
      {/* Top toolbar */}
      <div className="flex items-center gap-2 px-4 py-2 bg-gray-900 border-b border-gray-800">
        {/* Model preset selector */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-400">Model:</span>
          <select
            value={modelPreset}
            onChange={e => setModelPreset(e.target.value as keyof typeof MODEL_PRESETS)}
            className="px-2 py-1 text-sm bg-gray-800 border border-gray-700 rounded text-gray-300"
          >
            {Object.entries(MODEL_PRESETS).map(([key, p]) => (
              <option key={key} value={key}>{p.name} ({p.fps}fps)</option>
            ))}
          </select>
        </div>

        <div className="w-px h-6 bg-gray-700" />

        {/* Resolution preset */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-400">Resolution:</span>
          <select
            value={`${targetWidth}x${targetHeight}`}
            onChange={e => {
              const [w, h] = e.target.value.split('x').map(Number);
              setTargetWidth(w);
              setTargetHeight(h);
              const res = preset.resolutions.find(r => r.width === w && r.height === h);
              if (res) setTargetFrames(res.frames);
            }}
            className="px-2 py-1 text-sm bg-gray-800 border border-gray-700 rounded text-gray-300"
          >
            {preset.resolutions.map(r => (
              <option key={r.label} value={`${r.width}x${r.height}`}>{r.label}</option>
            ))}
          </select>
        </div>

        <div className="w-px h-6 bg-gray-700" />

        {/* Frames */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-400">Frames:</span>
          <select
            value={targetFrames}
            onChange={e => setTargetFrames(Number(e.target.value))}
            className="px-2 py-1 text-sm bg-gray-800 border border-gray-700 rounded text-gray-300"
          >
            {preset.validFrames.map(f => (
              <option key={f} value={f}>{f} ({(f / targetFps).toFixed(2)}s)</option>
            ))}
          </select>
          <span className="text-xs text-gray-500">({preset.frameRule})</span>
        </div>

        <div className="flex-1" />

        {/* Keyboard shortcuts hint */}
        <div className="text-xs text-gray-500">
          Space: Play | J/K/L: Speed | I/O: In/Out | ←/→: Frame
        </div>

        <button
          onClick={() => setShowSettings(!showSettings)}
          className={`p-2 rounded ${showSettings ? 'bg-gray-700 text-amber-400' : 'text-gray-400 hover:text-gray-200'}`}
          title="Settings"
        >
          <Settings className="w-4 h-4" />
        </button>
      </div>

      {/* Settings panel (collapsible) */}
      {showSettings && (
        <div className="px-4 py-3 bg-gray-900/50 border-b border-gray-800">
          <div className="grid grid-cols-6 gap-4">
            <div>
              <label className="block text-xs text-gray-400 mb-1">Target FPS</label>
              <input
                type="number"
                value={targetFps}
                onChange={e => setTargetFps(Number(e.target.value))}
                className="w-full px-2 py-1 text-sm bg-gray-800 border border-gray-700 rounded text-gray-300"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Width</label>
              <input
                type="number"
                value={targetWidth}
                onChange={e => setTargetWidth(Number(e.target.value))}
                className="w-full px-2 py-1 text-sm bg-gray-800 border border-gray-700 rounded text-gray-300"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Height</label>
              <input
                type="number"
                value={targetHeight}
                onChange={e => setTargetHeight(Number(e.target.value))}
                className="w-full px-2 py-1 text-sm bg-gray-800 border border-gray-700 rounded text-gray-300"
              />
            </div>
            <div className="flex flex-col justify-end gap-1">
              <label className="flex items-center gap-2 text-xs text-gray-400">
                <input
                  type="checkbox"
                  checked={enableBucket}
                  onChange={e => setEnableBucket(e.target.checked)}
                  className="accent-amber-500"
                />
                Enable Bucket
              </label>
              <label className="flex items-center gap-2 text-xs text-gray-400">
                <input
                  type="checkbox"
                  checked={bucketNoUpscale}
                  onChange={e => setBucketNoUpscale(e.target.checked)}
                  className="accent-amber-500"
                />
                No Upscale
              </label>
            </div>
            <div className="col-span-2">
              <label className="block text-xs text-gray-400 mb-1">Max Longest Edge (optional)</label>
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={exportSettings.useMaxEdge}
                  onChange={e => setExportSettings({ ...exportSettings, useMaxEdge: e.target.checked })}
                  className="accent-amber-500"
                />
                <input
                  type="number"
                  value={exportSettings.maxLongestEdge}
                  onChange={e => setExportSettings({ ...exportSettings, maxLongestEdge: Number(e.target.value) })}
                  disabled={!exportSettings.useMaxEdge}
                  className="flex-1 px-2 py-1 text-sm bg-gray-800 border border-gray-700 rounded text-gray-300 disabled:opacity-50"
                />
              </div>
            </div>
          </div>

          {/* Export options */}
          <div className="mt-3 pt-3 border-t border-gray-700 flex items-center gap-6">
            <span className="text-xs text-gray-400">Export:</span>
            <label className="flex items-center gap-2 text-xs text-gray-300">
              <input
                type="checkbox"
                checked={exportSettings.exportCropped}
                onChange={e => setExportSettings({ ...exportSettings, exportCropped: e.target.checked })}
                className="accent-amber-500"
              />
              <FileVideo className="w-3 h-3" /> Cropped Clips
            </label>
            <label className="flex items-center gap-2 text-xs text-gray-300">
              <input
                type="checkbox"
                checked={exportSettings.exportUncropped}
                onChange={e => setExportSettings({ ...exportSettings, exportUncropped: e.target.checked })}
                className="accent-amber-500"
              />
              <FileVideo className="w-3 h-3" /> Uncropped Clips
            </label>
            <label className="flex items-center gap-2 text-xs text-gray-300">
              <input
                type="checkbox"
                checked={exportSettings.exportFirstFrame}
                onChange={e => setExportSettings({ ...exportSettings, exportFirstFrame: e.target.checked })}
                className="accent-amber-500"
              />
              <FileImage className="w-3 h-3" /> First Frame Images
            </label>
          </div>
        </div>
      )}

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left panel - File browser */}
        <div className="w-72 bg-gray-900 border-r border-gray-800 flex flex-col">
          {/* Input folder */}
          <div className="p-2 border-b border-gray-800">
            <label className="block text-xs text-gray-400 mb-1">Input Folder</label>
            <div className="flex gap-1">
              <input
                type="text"
                value={inputFolder}
                onChange={e => setInputFolder(e.target.value)}
                placeholder="/path/to/videos"
                className="flex-1 px-2 py-1 text-xs bg-gray-800 border border-gray-700 rounded text-gray-300"
              />
              <button
                onClick={scanFolder}
                disabled={isScanning}
                className="p-1.5 bg-amber-500 hover:bg-amber-400 text-black rounded disabled:opacity-50"
                title="Scan folder"
              >
                {isScanning ? <Loader2 className="w-4 h-4 animate-spin" /> : <RefreshCw className="w-4 h-4" />}
              </button>
            </div>
          </div>

          {/* Output folder */}
          <div className="p-2 border-b border-gray-800">
            <label className="block text-xs text-gray-400 mb-1">Output Folder</label>
            <input
              type="text"
              value={outputFolder}
              onChange={e => setOutputFolder(e.target.value)}
              placeholder="/path/to/output"
              className="w-full px-2 py-1 text-xs bg-gray-800 border border-gray-700 rounded text-gray-300"
            />
          </div>

          {/* Video list */}
          <div className="flex-1 overflow-y-auto">
            <div
              className="flex items-center px-2 py-1.5 bg-gray-800/50 cursor-pointer hover:bg-gray-800"
              onClick={() => setExpandedSection(expandedSection === 'videos' ? '' : 'videos')}
            >
              {expandedSection === 'videos' ? <ChevronDown className="w-4 h-4 text-gray-400" /> : <ChevronRight className="w-4 h-4 text-gray-400" />}
              <Film className="w-4 h-4 text-gray-400 ml-1" />
              <span className="ml-2 text-sm text-gray-300">Videos ({videos.length})</span>
            </div>
            {expandedSection === 'videos' && (
              <div className="p-1 space-y-1">
                {videos.map(v => (
                  <div
                    key={v.path}
                    onClick={() => {
                      setSelectedVideo(v);
                      setRanges([]);
                      setSelectedRangeId(null);
                      setPreviewTime(0);
                      setShowCropTool(false);
                    }}
                    className={`p-2 rounded cursor-pointer ${selectedVideo?.path === v.path ? 'bg-amber-500/20 border border-amber-500/50' : 'bg-gray-800/50 hover:bg-gray-800 border border-transparent'}`}
                  >
                    <div className="text-xs text-gray-300 truncate">{v.name}</div>
                    <div className="text-xs text-gray-500 mt-0.5">
                      {v.width}x{v.height} • {v.fps.toFixed(2)}fps • {formatTime(v.duration)}
                    </div>
                  </div>
                ))}
                {videos.length === 0 && (
                  <div className="p-4 text-center text-xs text-gray-500">
                    Set input folder and scan
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Actions */}
          <div className="p-2 border-t border-gray-800 space-y-2">
            <button
              onClick={generateConfig}
              className="w-full px-3 py-1.5 text-xs bg-gray-700 hover:bg-gray-600 text-gray-300 rounded flex items-center justify-center gap-1"
            >
              <Download className="w-3 h-3" /> Export TOML Config
            </button>
            <button
              onClick={processVideos}
              disabled={isProcessing || ranges.length === 0}
              className="w-full px-3 py-2 bg-amber-500 hover:bg-amber-400 text-black font-medium rounded disabled:opacity-50 flex items-center justify-center gap-1"
            >
              {isProcessing ? <Loader2 className="w-4 h-4 animate-spin" /> : <Scissors className="w-4 h-4" />}
              Process {ranges.length} Range{ranges.length !== 1 ? 's' : ''}
            </button>
          </div>
        </div>

        {/* Center - Video preview */}
        <div className="flex-1 flex flex-col bg-black" ref={containerRef}>
          {/* Video preview */}
          <div className="flex-1 min-h-0 flex items-center justify-center relative">
            {selectedVideo ? (
              <>
                <video
                  ref={videoRef}
                  src={`/api/vidprep/video?path=${encodeURIComponent(selectedVideo.path)}`}
                  className="max-w-full max-h-full object-contain"
                  onTimeUpdate={e => !isPlaying && setPreviewTime((e.target as HTMLVideoElement).currentTime)}
                  onLoadedMetadata={() => setPreviewTime(0)}
                />

                {/* Crop overlay */}
                {showCropTool && tempCrop && videoRef.current && (
                  <div className="absolute inset-0 pointer-events-none">
                    {/* Darkened areas outside crop */}
                    <div className="absolute inset-0 bg-black/60" />

                    {/* Crop box */}
                    <div
                      className="absolute border-2 border-amber-400 bg-transparent pointer-events-auto cursor-move"
                      style={getCropOverlayStyle()}
                      onMouseDown={e => handleCropMouseDown(e, 'move')}
                    >
                      {/* Clear area inside crop */}
                      <div className="absolute inset-0 bg-black/0" style={{ boxShadow: '0 0 0 9999px rgba(0,0,0,0.6)' }} />

                      {/* Resize handle */}
                      <div
                        className="absolute bottom-0 right-0 w-4 h-4 bg-amber-400 cursor-se-resize"
                        onMouseDown={e => { e.stopPropagation(); handleCropMouseDown(e, 'resize'); }}
                      />

                      {/* Crop dimensions */}
                      <div className="absolute -top-6 left-0 text-xs text-amber-400 bg-black/80 px-1 rounded">
                        {Math.round(tempCrop.width)}x{Math.round(tempCrop.height)}
                      </div>
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div className="text-gray-600">Select a video to preview</div>
            )}
          </div>

          {/* Timeline with range markers */}
          {selectedVideo && (
            <div className="px-4 py-2 bg-gray-900/80 border-t border-gray-800">
              {/* Range markers on timeline */}
              <div className="relative h-8 mb-2">
                <div className="absolute inset-x-0 top-1/2 h-1 bg-gray-700 rounded" />

                {/* Range markers */}
                {ranges.map(r => {
                  const left = (r.start / selectedVideo.duration) * 100;
                  const width = ((r.end - r.start) / selectedVideo.duration) * 100;
                  return (
                    <div
                      key={r.id}
                      onClick={() => {
                        setSelectedRangeId(r.id);
                        seekTo(r.start);
                      }}
                      className={`absolute top-1 h-6 rounded cursor-pointer transition-all ${
                        selectedRangeId === r.id
                          ? 'bg-amber-500 ring-2 ring-amber-300'
                          : 'bg-green-600 hover:bg-green-500'
                      }`}
                      style={{ left: `${left}%`, width: `${Math.max(width, 0.5)}%` }}
                    >
                      <div className="absolute -top-4 left-0 text-xs text-gray-400 whitespace-nowrap">
                        {ranges.indexOf(r) + 1}
                      </div>
                    </div>
                  );
                })}

                {/* Current position indicator */}
                <div
                  className="absolute top-0 w-0.5 h-full bg-red-500 z-10"
                  style={{ left: `${(previewTime / selectedVideo.duration) * 100}%` }}
                >
                  <div className="absolute -top-1 -left-1.5 w-3 h-3 bg-red-500 rounded-full" />
                </div>
              </div>

              {/* Seek bar */}
              <input
                type="range"
                min={0}
                max={selectedVideo.duration}
                step={1 / selectedVideo.fps}
                value={previewTime}
                onChange={e => seekTo(Number(e.target.value))}
                className="w-full h-1 bg-gray-700 rounded appearance-none cursor-pointer accent-amber-500"
              />
            </div>
          )}

          {/* Playback controls */}
          {selectedVideo && (
            <div className="flex-shrink-0 flex items-center gap-2 px-4 py-3 bg-gray-900 border-t border-gray-800">
              {/* Transport controls */}
              <button
                onClick={() => seekTo(0)}
                className="p-1.5 bg-gray-800 hover:bg-gray-700 rounded"
                title="Go to start (Home)"
              >
                <SkipBack className="w-4 h-4" />
              </button>

              <button
                onClick={() => {
                  if (selectedVideo) {
                    seekTo(Math.max(0, previewTime - 1 / selectedVideo.fps));
                  }
                }}
                className="p-1.5 bg-gray-800 hover:bg-gray-700 rounded"
                title="Previous frame (←)"
              >
                <ChevronDown className="w-4 h-4 -rotate-90" />
              </button>

              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className="p-2 bg-gray-800 hover:bg-gray-700 rounded"
                title="Play/Pause (Space)"
              >
                {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
              </button>

              <button
                onClick={() => {
                  if (selectedVideo) {
                    seekTo(Math.min(selectedVideo.duration, previewTime + 1 / selectedVideo.fps));
                  }
                }}
                className="p-1.5 bg-gray-800 hover:bg-gray-700 rounded"
                title="Next frame (→)"
              >
                <ChevronDown className="w-4 h-4 rotate-90" />
              </button>

              <button
                onClick={() => seekTo(selectedVideo.duration)}
                className="p-1.5 bg-gray-800 hover:bg-gray-700 rounded"
                title="Go to end (End)"
              >
                <SkipForward className="w-4 h-4" />
              </button>

              {/* Speed indicator */}
              <div className="text-xs text-gray-500 w-12 text-center">
                {playbackSpeed}x
              </div>

              {/* Time display */}
              <div className="text-sm font-mono text-gray-400 ml-2">
                {formatTimeWithFrames(previewTime, selectedVideo.fps)} / {formatTimeWithFrames(selectedVideo.duration, selectedVideo.fps)}
              </div>

              <div className="flex-1" />

              {/* Range controls */}
              {selectedRange && (
                <>
                  <button
                    onClick={() => updateRange(selectedRange.id, { start: previewTime })}
                    className="px-2 py-1 bg-blue-600 hover:bg-blue-500 text-white text-xs rounded"
                    title="Set In point (I)"
                  >
                    Set In
                  </button>
                  <button
                    onClick={() => updateRange(selectedRange.id, { end: previewTime })}
                    className="px-2 py-1 bg-blue-600 hover:bg-blue-500 text-white text-xs rounded"
                    title="Set Out point (O)"
                  >
                    Set Out
                  </button>
                  <div className="w-px h-6 bg-gray-700" />
                </>
              )}

              <button
                onClick={addRange}
                className="px-3 py-1.5 bg-green-600 hover:bg-green-500 text-white text-sm rounded flex items-center gap-1"
                title="Add new range (N)"
              >
                <Plus className="w-4 h-4" /> Add Range
              </button>

              <div className="text-xs text-gray-500 ml-2">
                {selectedVideo.width}x{selectedVideo.height} @ {selectedVideo.fps.toFixed(2)}fps
              </div>
            </div>
          )}
        </div>

        {/* Right panel - Ranges */}
        <div className="w-80 bg-gray-900 border-l border-gray-800 flex flex-col">
          <div className="px-3 py-2 border-b border-gray-800 flex items-center justify-between">
            <span className="text-sm text-gray-300 font-medium">Ranges ({ranges.length})</span>
            <button
              onClick={() => setRanges([])}
              disabled={ranges.length === 0}
              className="text-xs text-red-400 hover:text-red-300 disabled:opacity-50"
            >
              Clear All
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-2 space-y-2">
            {ranges.map((range, i) => (
              <div
                key={range.id}
                onClick={() => {
                  setSelectedRangeId(range.id);
                  seekTo(range.start);
                }}
                className={`p-3 rounded border ${
                  selectedRangeId === range.id
                    ? 'border-amber-500 bg-amber-500/10'
                    : 'border-gray-700 bg-gray-800/50 hover:bg-gray-800'
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-300 font-medium">Range {i + 1}</span>
                  <div className="flex items-center gap-1">
                    {/* Crop button */}
                    <button
                      onClick={e => {
                        e.stopPropagation();
                        setSelectedRangeId(range.id);
                        initCrop();
                      }}
                      className={`p-1 rounded ${range.useCrop ? 'text-amber-400 bg-amber-500/20' : 'text-gray-400 hover:text-gray-300'}`}
                      title="Crop region"
                    >
                      <Crop className="w-3 h-3" />
                    </button>
                    <button
                      onClick={e => { e.stopPropagation(); removeRange(range.id); }}
                      className="p-1 text-red-400 hover:text-red-300"
                      title="Delete range"
                    >
                      <Trash2 className="w-3 h-3" />
                    </button>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-2 text-xs mb-2">
                  <div>
                    <label className="text-gray-500">In (I)</label>
                    <input
                      type="number"
                      step="0.001"
                      value={range.start.toFixed(3)}
                      onChange={e => updateRange(range.id, { start: Number(e.target.value) })}
                      onClick={e => e.stopPropagation()}
                      className="w-full px-1.5 py-0.5 bg-gray-700 border border-gray-600 rounded text-gray-300"
                    />
                  </div>
                  <div>
                    <label className="text-gray-500">Out (O)</label>
                    <input
                      type="number"
                      step="0.001"
                      value={range.end.toFixed(3)}
                      onChange={e => updateRange(range.id, { end: Number(e.target.value) })}
                      onClick={e => e.stopPropagation()}
                      className="w-full px-1.5 py-0.5 bg-gray-700 border border-gray-600 rounded text-gray-300"
                    />
                  </div>
                </div>

                <div className="text-xs mb-2">
                  <label className="text-gray-500">Caption</label>
                  <textarea
                    value={range.caption}
                    onChange={e => updateRange(range.id, { caption: e.target.value })}
                    onClick={e => e.stopPropagation()}
                    placeholder="Describe this clip..."
                    className="w-full px-1.5 py-1 bg-gray-700 border border-gray-600 rounded text-gray-300 resize-none h-16"
                  />
                </div>

                {/* Range stats */}
                <div className="flex items-center justify-between text-xs text-gray-500">
                  <span>
                    {Math.round((range.end - range.start) * targetFps)} frames
                  </span>
                  <span>{(range.end - range.start).toFixed(3)}s</span>
                  {range.useCrop && range.crop && (
                    <span className="text-amber-400">
                      {Math.round(range.crop.width)}x{Math.round(range.crop.height)}
                    </span>
                  )}
                </div>
              </div>
            ))}

            {ranges.length === 0 && (
              <div className="p-4 text-center text-xs text-gray-500">
                Press N or click "Add Range" to create ranges
              </div>
            )}
          </div>

          {/* Crop tool panel */}
          {showCropTool && selectedRange && (
            <div className="p-3 border-t border-gray-700 bg-gray-800/50">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-300">Crop Tool</span>
                <div className="flex items-center gap-1">
                  <button
                    onClick={setCropToAspect}
                    className="px-2 py-1 text-xs bg-gray-700 hover:bg-gray-600 text-gray-300 rounded"
                    title="Set to target aspect ratio"
                  >
                    <Maximize2 className="w-3 h-3 inline mr-1" />
                    Fit Aspect
                  </button>
                </div>
              </div>

              {tempCrop && (
                <div className="grid grid-cols-4 gap-2 text-xs mb-2">
                  <div>
                    <label className="text-gray-500">X</label>
                    <input
                      type="number"
                      value={Math.round(tempCrop.x)}
                      onChange={e => setTempCrop({ ...tempCrop, x: Number(e.target.value) })}
                      className="w-full px-1 py-0.5 bg-gray-700 border border-gray-600 rounded text-gray-300"
                    />
                  </div>
                  <div>
                    <label className="text-gray-500">Y</label>
                    <input
                      type="number"
                      value={Math.round(tempCrop.y)}
                      onChange={e => setTempCrop({ ...tempCrop, y: Number(e.target.value) })}
                      className="w-full px-1 py-0.5 bg-gray-700 border border-gray-600 rounded text-gray-300"
                    />
                  </div>
                  <div>
                    <label className="text-gray-500">W</label>
                    <input
                      type="number"
                      value={Math.round(tempCrop.width)}
                      onChange={e => setTempCrop({ ...tempCrop, width: Number(e.target.value) })}
                      className="w-full px-1 py-0.5 bg-gray-700 border border-gray-600 rounded text-gray-300"
                    />
                  </div>
                  <div>
                    <label className="text-gray-500">H</label>
                    <input
                      type="number"
                      value={Math.round(tempCrop.height)}
                      onChange={e => setTempCrop({ ...tempCrop, height: Number(e.target.value) })}
                      className="w-full px-1 py-0.5 bg-gray-700 border border-gray-600 rounded text-gray-300"
                    />
                  </div>
                </div>
              )}

              <div className="flex items-center gap-2">
                <button
                  onClick={applyCrop}
                  className="flex-1 px-2 py-1.5 bg-amber-500 hover:bg-amber-400 text-black text-sm rounded flex items-center justify-center gap-1"
                >
                  <Check className="w-4 h-4" /> Apply
                </button>
                <button
                  onClick={cancelCrop}
                  className="flex-1 px-2 py-1.5 bg-gray-700 hover:bg-gray-600 text-gray-300 text-sm rounded"
                >
                  Cancel
                </button>
              </div>
            </div>
          )}

          {/* Output info */}
          <div className="p-3 border-t border-gray-800 bg-gray-800/50">
            <div className="text-xs text-gray-400 space-y-1">
              <div className="flex justify-between">
                <span>Output FPS:</span>
                <span className="text-amber-400">{targetFps}</span>
              </div>
              <div className="flex justify-between">
                <span>Output Size:</span>
                <span className="text-amber-400">{targetWidth}x{targetHeight}</span>
              </div>
              <div className="flex justify-between">
                <span>Target Frames:</span>
                <span className="text-amber-400">{targetFrames} ({(targetFrames / targetFps).toFixed(2)}s)</span>
              </div>
              <div className="flex justify-between">
                <span>Aspect Ratio:</span>
                <span className="text-amber-400">{(targetWidth / targetHeight).toFixed(3)}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
