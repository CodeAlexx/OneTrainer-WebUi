import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { ChevronRight, ChevronDown, X, Plus, Trash2, Paintbrush, ImageIcon, Wand2, Film, Scissors } from 'lucide-react';
import { MaskEditor } from './components/MaskEditor';
import { VideoEditor } from './components/VideoEditor';
import { VidPrep } from './components/VidPrep';

const API = axios.create({ baseURL: '/api' });

// Types
type ModelType = 'flux_dev' | 'flux_schnell' | 'sdxl' | 'sd_35' | 'z_image' | 'qwen_image' | 'lumina_2' | 'omnigen_2' | 'wan_t2v' | string;
type GenerationMode = 'txt2img' | 'img2img' | 'inpaint';

interface LoRA { path: string; weight: number; enabled: boolean; }
interface GeneratedImage { id: string; path: string; thumbnail: string; prompt: string; seed: number; }

// Collapsible Section Component
const Section = ({ title, children, defaultOpen = false, toggle, enabled, onToggle }: {
  title: string; children?: React.ReactNode; defaultOpen?: boolean;
  toggle?: boolean; enabled?: boolean; onToggle?: (v: boolean) => void;
}) => {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border-b border-gray-700">
      <div className="flex items-center px-2 py-1.5 hover:bg-gray-800 cursor-pointer select-none"
        onClick={() => setOpen(!open)}>
        {open ? <ChevronDown className="w-4 h-4 text-gray-400" /> : <ChevronRight className="w-4 h-4 text-gray-400" />}
        <span className="ml-1 text-sm text-gray-300 flex-1">{title}</span>
        {toggle && (
          <div className="ml-2" onClick={e => { e.stopPropagation(); onToggle?.(!enabled); }}>
            <div className={`w-8 h-4 rounded-full relative transition-colors ${enabled ? 'bg-amber-500' : 'bg-gray-600'}`}>
              <div className={`absolute top-0.5 w-3 h-3 rounded-full bg-white transition-all ${enabled ? 'left-4' : 'left-0.5'}`} />
            </div>
          </div>
        )}
      </div>
      {open && <div className="px-3 py-2 space-y-2 bg-gray-900/50">{children}</div>}
    </div>
  );
};

// Slider Component
const Slider = ({ label, value, onChange, min, max, step = 1 }: {
  label: string; value: number; onChange: (v: number) => void; min: number; max: number; step?: number;
}) => (
  <div className="flex items-center gap-2">
    <span className="text-xs text-gray-400 w-20 shrink-0">{label}</span>
    <input type="range" min={min} max={max} step={step} value={value} onChange={e => onChange(Number(e.target.value))}
      className="flex-1 h-1 bg-gray-700 rounded appearance-none cursor-pointer accent-amber-500" />
    <input type="number" value={value} onChange={e => onChange(Number(e.target.value))}
      className="w-14 px-1 py-0.5 text-xs bg-gray-800 border border-gray-700 rounded text-right text-gray-300" />
  </div>
);

// Select Component
const Select = ({ label, value, onChange, options }: {
  label: string; value: string; onChange: (v: string) => void; options: string[] | { value: string; label: string }[];
}) => (
  <div className="flex items-center gap-2">
    <span className="text-xs text-gray-400 w-20 shrink-0">{label}</span>
    <select value={value} onChange={e => onChange(e.target.value)}
      className="flex-1 px-2 py-1 text-xs bg-gray-800 border border-gray-700 rounded text-gray-300">
      {options.map(o => typeof o === 'string' ?
        <option key={o} value={o}>{o}</option> :
        <option key={o.value} value={o.value}>{o.label}</option>
      )}
    </select>
  </div>
);

// Model options with paths
const MODEL_OPTIONS: { value: string; label: string; path: string; category?: string }[] = [
  // Image models
  { value: 'flux_dev', label: 'FLUX Dev', path: '/home/alex/SwarmUI/Models/diffusion_models/flux1-dev.safetensors', category: 'Image' },
  { value: 'flux_schnell', label: 'FLUX Schnell', path: '/home/alex/SwarmUI/Models/diffusion_models/uncensoredFemalesFLUX4step_nf4Schnell4step.safetensors', category: 'Image' },
  { value: 'sdxl', label: 'SDXL', path: '/home/alex/SwarmUI/Models/diffusion_models/lustifySDXLNSFW_ggwpV7.safetensors', category: 'Image' },
  { value: 'sd_35', label: 'SD 3.5', path: '/home/alex/SwarmUI/Models/diffusion_models/sd3.5_large.safetensors', category: 'Image' },
  { value: 'z_image', label: 'Z-Image', path: '/home/alex/SwarmUI/Models/diffusion_models/z_image_de_turbo_v1_bf16.safetensors', category: 'Image' },
  { value: 'z_image_turbo', label: 'Z-Image Turbo', path: '/home/alex/SwarmUI/Models/diffusion_models/z_image_turbo_bf16.safetensors', category: 'Image' },
  { value: 'qwen_image', label: 'Qwen Image', path: '/home/alex/SwarmUI/Models/diffusion_models/qwen_image_fp8_e4m3fn.safetensors', category: 'Image' },
  { value: 'qwen_image_edit', label: 'Qwen-Edit', path: 'Qwen/Qwen-Image-Edit', category: 'Image' },
  { value: 'lumina_2', label: 'Lumina 2', path: 'Alpha-VLLM/Lumina-Image-2.0', category: 'Image' },
  { value: 'omnigen_2', label: 'OmniGen 2', path: 'BAAI/OmniGen2', category: 'Image' },
  // Kandinsky 5
  { value: 'kandinsky_5', label: 'Kandinsky 5 T2I', path: 'kandinskylab/Kandinsky-5.0-T2I-Lite', category: 'Image' },
  { value: 'kandinsky_5_video', label: 'Kandinsky 5 T2V Lite', path: 'kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s', category: 'Video' },
  { value: 'kandinsky_5_video_pro', label: 'Kandinsky 5 T2V Pro', path: '/home/alex/OneTrainer/models/kandinsky-5-video-pro/model/kandinsky5pro_t2v_sft_5s.safetensors', category: 'Video' },
  // Wan 2.2 Video models
  { value: 'wan_t2v_high', label: 'Wan 2.2 T2V (High)', path: '/home/alex/SwarmUI/Models/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors', category: 'Video' },
  { value: 'wan_t2v_low', label: 'Wan 2.2 T2V (Low)', path: '/home/alex/SwarmUI/Models/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors', category: 'Video' },
  { value: 'wan_i2v_high', label: 'Wan 2.2 I2V (High)', path: '/home/alex/SwarmUI/Models/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors', category: 'Video' },
  { value: 'wan_i2v_low', label: 'Wan 2.2 I2V (Low)', path: '/home/alex/SwarmUI/Models/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors', category: 'Video' },
  { value: 'wan_vace', label: 'Wan 2.1 VACE', path: '/home/alex/SwarmUI/Models/diffusion_models/wan2.1_vace_14B_fp16.safetensors', category: 'Video' },
  // Hunyuan Video (placeholder - need weights)
  // { value: 'hunyuan_video', label: 'Hunyuan Video', path: '', category: 'Video' },
];

// Get path for model type
const getModelPath = (modelType: string): string => {
  const model = MODEL_OPTIONS.find(m => m.value === modelType);
  return model?.path || modelType;
};

const SAMPLER_OPTIONS = ['euler', 'euler_a', 'dpm_2m', 'dpm_2m_karras', 'ddim', 'unipc', 'heun'];
const RESOLUTION_OPTIONS = ['512x512', '768x768', '1024x1024', '1280x720', '720x1280', '1536x1536'];

// Video resolution presets (Kandinsky uses specific sizes)
const VIDEO_RESOLUTION_OPTIONS = [
  { value: '512x512', label: '512x512 (Square)' },
  { value: '768x512', label: '768x512 (Landscape)' },
  { value: '512x768', label: '512x768 (Portrait)' },
  { value: '1024x1024', label: '1024x1024 (Square HD)' },
  { value: '1280x768', label: '1280x768 (Landscape HD)' },
  { value: '768x1280', label: '768x1280 (Portrait HD)' },
];

// Check if model is a video model
const isVideoModel = (model: string) => {
  const videoModels = ['kandinsky_5_video', 'kandinsky_5_video_pro', 'wan_t2v_high', 'wan_t2v_low', 'wan_i2v_high', 'wan_i2v_low', 'wan_vace', 'hunyuan_video'];
  return videoModels.includes(model);
};

export default function App() {
  // Model state
  const [modelType, setModelType] = useState<ModelType>('flux_dev');
  const [modelPath, setModelPath] = useState('');
  const [precision] = useState('bf16');
  const [modelLoaded, setModelLoaded] = useState(false);

  // Generation mode & params
  const [mode, setMode] = useState<GenerationMode>('txt2img');
  const [prompt, setPrompt] = useState('');
  const [negPrompt, setNegPrompt] = useState('');
  const [seed, setSeed] = useState(-1);
  const [steps, setSteps] = useState(20);
  const [cfg, setCfg] = useState(7);
  const [sampler, setSampler] = useState('euler');
  const [resolution, setResolution] = useState('1024x1024');
  const [images, setImages] = useState(1);

  // Optional features
  const [variationSeed, setVariationSeed] = useState(false);
  const [initImage, setInitImage] = useState<string | null>(null);
  const [maskImage, setMaskImage] = useState<string | null>(null);
  const [showMaskEditor, setShowMaskEditor] = useState(false);
  const [initStrength, setInitStrength] = useState(0.75);
  const [refineEnabled, setRefineEnabled] = useState(false);
  const [refineScale, setRefineScale] = useState(2);
  const [cnEnabled, setCnEnabled] = useState(false);
  const [numFrames, setNumFrames] = useState(16);
  const [videoDuration, setVideoDuration] = useState(5); // seconds (for Kandinsky)
  const [videoFps, setVideoFps] = useState(24);
  const [videoResolution, setVideoResolution] = useState('768x512');
  const [freeU, setFreeU] = useState(false);

  // LoRAs
  const [loras, setLoras] = useState<LoRA[]>([]);

  // UI state
  const [isGenerating, setIsGenerating] = useState(false);
  const [isLoadingModel, setIsLoadingModel] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState('');
  const [currentStep, setCurrentStep] = useState(0);
  const [totalSteps, setTotalSteps] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [gallery, setGallery] = useState<GeneratedImage[]>([]);
  const [selectedImage, setSelectedImage] = useState<GeneratedImage | null>(null);
  const [bottomTab, setBottomTab] = useState<'history' | 'models' | 'loras'>('history');
  const [mainTab, setMainTab] = useState<'generate' | 'vidprep' | 'editor'>('generate');
  const [showAdvanced, setShowAdvanced] = useState(false);

  const promptRef = useRef<HTMLTextAreaElement>(null);

  // Poll status
  useEffect(() => {
    const poll = async () => {
      try {
        const r = await API.get('/status');
        setModelLoaded(r.data.model_info?.loaded || false);
        setIsGenerating(r.data.is_generating);
        setCurrentStep(r.data.current_step || 0);
        setTotalSteps(r.data.total_steps || 0);
      } catch {}
    };
    poll();
    const i = setInterval(poll, 500); // Poll faster for better UX
    return () => clearInterval(i);
  }, []);

  // Load gallery
  useEffect(() => {
    API.get('/gallery').then(r => setGallery(r.data.images || [])).catch(() => {});
  }, []);

  // Parse resolution (use video resolution for video models)
  const activeResolution = isVideoModel(modelType) ? videoResolution : resolution;
  const [width, height] = activeResolution.split('x').map(Number);

  // Generate
  const generate = async () => {
    if (!modelPath && !modelType) { setError('Select a model'); return; }
    if (!prompt.trim()) { setError('Enter a prompt'); return; }
    if ((mode === 'img2img' || mode === 'inpaint') && !initImage) {
      setError('Select an init image for img2img/inpaint mode'); return;
    }
    if (mode === 'inpaint' && !maskImage) {
      setError('Create a mask for inpainting mode'); return;
    }

    setError(null);

    // Show loading message if model not loaded
    const selectedModel = MODEL_OPTIONS.find(m => m.value === modelType);
    if (!modelLoaded) {
      setIsLoadingModel(true);
      setLoadingMessage(`Loading ${selectedModel?.label || modelType}...`);
    }

    try {
      const r = await API.post('/generate', {
        model_path: modelPath || getModelPath(modelType), model_type: modelType, precision,
        mode: mode,
        prompt, negative_prompt: negPrompt,
        width, height, steps, cfg_scale: cfg, sampler, seed,
        batch_count: images,
        init_image: (mode === 'img2img' || mode === 'inpaint') ? initImage : undefined,
        mask_image: mode === 'inpaint' ? maskImage : undefined,
        strength: initStrength,
        free_u: freeU,
        loras: loras.filter(l => l.enabled),
        num_frames: isVideoModel(modelType) ? numFrames : undefined,
        video_duration: isVideoModel(modelType) ? videoDuration : undefined,
        video_fps: isVideoModel(modelType) ? videoFps : undefined,
        enable_hires: refineEnabled, hires_scale: refineScale,
      });

      if (r.data.images?.length) {
        const newImages = r.data.images;
        setGallery(prev => [...newImages, ...prev]);
        // Force new object reference to trigger re-render
        setSelectedImage({ ...newImages[0] });
      }
    } catch (e: any) {
      setError(e.response?.data?.detail || 'Generation failed');
    } finally {
      setIsGenerating(false);
      setIsLoadingModel(false);
      setLoadingMessage('');
    }
  };

  // Use selected image as init image
  const useAsInitImage = async () => {
    if (!selectedImage) return;
    try {
      const response = await fetch(`/api/gallery/${selectedImage.id}`);
      const blob = await response.blob();
      const reader = new FileReader();
      reader.onload = () => {
        setInitImage(reader.result as string);
        if (mode === 'txt2img') setMode('img2img');
      };
      reader.readAsDataURL(blob);
    } catch (e) {
      console.error('Failed to load image:', e);
    }
  };

  // Open mask editor
  const openMaskEditor = () => {
    if (!initImage) {
      setError('Select an init image first');
      return;
    }
    setShowMaskEditor(true);
  };

  const cancel = () => API.post('/generate/cancel');
  const randomSeed = () => setSeed(Math.floor(Math.random() * 2147483647));
  const recycleSeed = () => selectedImage && setSeed(selectedImage.seed);

  const addLora = () => setLoras([...loras, { path: '', weight: 1, enabled: true }]);
  const removeLora = (i: number) => setLoras(loras.filter((_, idx) => idx !== i));

  return (
    <div className="h-screen flex flex-col bg-gray-950 text-gray-200 overflow-hidden">
      {/* Mask Editor Modal */}
      {showMaskEditor && initImage && (
        <MaskEditor
          image={initImage}
          mask={maskImage}
          onMaskChange={(mask) => {
            setMaskImage(mask);
            if (mask && mode !== 'inpaint') setMode('inpaint');
          }}
          onClose={() => setShowMaskEditor(false)}
        />
      )}

      {/* Top tabs */}
      <div className="flex bg-gray-900 border-b border-gray-800">
        {/* Mode selector - Generate tab */}
        <div className="flex items-center border-r border-gray-800">
          <button
            onClick={() => { setMainTab('generate'); setMode('txt2img'); }}
            className={`px-4 py-2 text-sm flex items-center gap-1.5 ${mainTab === 'generate' && mode === 'txt2img' ? 'bg-gray-800 text-amber-400 border-b-2 border-amber-500' : 'text-gray-400 hover:text-gray-200'}`}
          >
            <Wand2 className="w-4 h-4" /> txt2img
          </button>
          <button
            onClick={() => { setMainTab('generate'); setMode('img2img'); }}
            className={`px-4 py-2 text-sm flex items-center gap-1.5 ${mainTab === 'generate' && mode === 'img2img' ? 'bg-gray-800 text-amber-400 border-b-2 border-amber-500' : 'text-gray-400 hover:text-gray-200'}`}
          >
            <ImageIcon className="w-4 h-4" /> img2img
          </button>
          <button
            onClick={() => { setMainTab('generate'); setMode('inpaint'); }}
            className={`px-4 py-2 text-sm flex items-center gap-1.5 ${mainTab === 'generate' && mode === 'inpaint' ? 'bg-gray-800 text-amber-400 border-b-2 border-amber-500' : 'text-gray-400 hover:text-gray-200'}`}
          >
            <Paintbrush className="w-4 h-4" /> Inpaint
          </button>
        </div>
        {/* Vid Prep Tab */}
        <button
          onClick={() => setMainTab('vidprep')}
          className={`px-4 py-2 text-sm flex items-center gap-1.5 ${mainTab === 'vidprep' ? 'bg-gray-800 text-amber-400 border-b-2 border-amber-500' : 'text-gray-400 hover:text-gray-200'}`}
        >
          <Scissors className="w-4 h-4" /> Vid Prep
        </button>
        {/* Video Editor Tab */}
        <button
          onClick={() => setMainTab('editor')}
          className={`px-4 py-2 text-sm flex items-center gap-1.5 ${mainTab === 'editor' ? 'bg-gray-800 text-amber-400 border-b-2 border-amber-500' : 'text-gray-400 hover:text-gray-200'}`}
        >
          <Film className="w-4 h-4" /> Video Editor
        </button>
        <button className="px-4 py-2 text-sm text-gray-400 hover:text-gray-200">Models</button>
        <button className="px-4 py-2 text-sm text-gray-400 hover:text-gray-200">Settings</button>
        <div className="flex-1" />
        <div className="px-4 py-2 text-xs text-gray-500">OneTrainer Inference</div>
      </div>

      {/* Vid Prep Mode */}
      {mainTab === 'vidprep' ? (
        <VidPrep />
      ) : mainTab === 'editor' ? (
        <VideoEditor />
      ) : (
      <>
      {/* Main area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left sidebar - Parameters */}
        <div className="w-72 bg-gray-900 border-r border-gray-800 overflow-y-auto">
          {/* Filter */}
          <div className="p-2 border-b border-gray-800">
            <input type="text" placeholder="Filter parameters..."
              className="w-full px-2 py-1 text-xs bg-gray-800 border border-gray-700 rounded text-gray-300 placeholder-gray-500" />
          </div>

          {/* Core Parameters */}
          <Section title="Core Parameters" defaultOpen>
            <Slider label="Images" value={images} onChange={setImages} min={1} max={16} />
            <div className="flex items-center gap-2">
              <span className="text-xs text-gray-400 w-20">Seed</span>
              <input type="number" value={seed} onChange={e => setSeed(Number(e.target.value))}
                className="flex-1 px-2 py-1 text-xs bg-gray-800 border border-gray-700 rounded text-gray-300" />
              <button onClick={randomSeed} className="p-1 bg-amber-500 rounded text-black text-xs" title="Random">🎲</button>
              <button onClick={recycleSeed} className="p-1 bg-amber-600 rounded text-black text-xs" title="Recycle">♻️</button>
            </div>
            <Slider label="Steps" value={steps} onChange={setSteps} min={1} max={150} />
            <Slider label="CFG Scale" value={cfg} onChange={setCfg} min={1} max={30} step={0.5} />
          </Section>

          <Section title="Variation Seed" toggle enabled={variationSeed} onToggle={setVariationSeed}>
            {variationSeed && <Slider label="Strength" value={0.5} onChange={() => {}} min={0} max={1} step={0.05} />}
          </Section>

          <Section title="Resolution" defaultOpen>
            <Select label="Preset" value={resolution} onChange={setResolution} options={RESOLUTION_OPTIONS} />
          </Section>

          <Section title="Sampling" defaultOpen>
            <Select label="Sampler" value={sampler} onChange={setSampler} options={SAMPLER_OPTIONS} />
          </Section>

          <Section title="Init Image" defaultOpen={mode !== 'txt2img'}>
            {/* Image upload / display */}
            {initImage ? (
              <div className="space-y-2">
                <div className="relative">
                  <img src={initImage} alt="Init" className="w-full rounded border border-gray-700" />
                  <button
                    onClick={() => { setInitImage(null); setMaskImage(null); }}
                    className="absolute top-1 right-1 p-1 bg-red-600 hover:bg-red-500 rounded text-white"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </div>
                {/* Mask preview for inpaint mode */}
                {mode === 'inpaint' && (
                  <div className="space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-gray-400">Mask</span>
                      <button
                        onClick={openMaskEditor}
                        className="px-2 py-1 text-xs bg-purple-600 hover:bg-purple-500 text-white rounded flex items-center gap-1"
                      >
                        <Paintbrush className="w-3 h-3" />
                        {maskImage ? 'Edit Mask' : 'Create Mask'}
                      </button>
                    </div>
                    {maskImage && (
                      <img src={maskImage} alt="Mask" className="w-full rounded border border-gray-700 opacity-70" />
                    )}
                  </div>
                )}
                <Slider label="Strength" value={initStrength} onChange={setInitStrength} min={0} max={1} step={0.05} />
              </div>
            ) : (
              <div className="space-y-2">
                <label className="block border border-dashed border-gray-600 rounded p-4 text-center text-xs text-gray-500 cursor-pointer hover:border-gray-500 hover:text-gray-400">
                  <input
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) {
                        const reader = new FileReader();
                        reader.onload = () => setInitImage(reader.result as string);
                        reader.readAsDataURL(file);
                      }
                    }}
                  />
                  Drop image here or click to upload
                </label>
                {selectedImage && (
                  <button
                    onClick={useAsInitImage}
                    className="w-full px-2 py-1.5 text-xs bg-gray-700 hover:bg-gray-600 text-gray-300 rounded flex items-center justify-center gap-1"
                  >
                    <ImageIcon className="w-3 h-3" />
                    Use selected image
                  </button>
                )}
              </div>
            )}
          </Section>

          <Section title="Refine / Upscale" toggle enabled={refineEnabled} onToggle={setRefineEnabled}>
            {refineEnabled && <Slider label="Scale" value={refineScale} onChange={setRefineScale} min={1} max={4} step={0.5} />}
          </Section>

          <Section title="ControlNet" toggle enabled={cnEnabled} onToggle={setCnEnabled}>
            {cnEnabled && <Select label="Model" value="" onChange={() => {}} options={['canny', 'depth', 'pose']} />}
          </Section>

          <Section title="VIDEO SETTINGS" defaultOpen={isVideoModel(modelType)}>
            <Slider label="Duration (s)" value={videoDuration} onChange={setVideoDuration} min={1} max={10} />
            <Slider label="Frames" value={numFrames} onChange={setNumFrames} min={4} max={64} />
            <div className="flex items-center gap-2">
              <span className="text-xs text-gray-400 w-20 shrink-0">Resolution</span>
              <select value={videoResolution} onChange={e => setVideoResolution(e.target.value)}
                className="flex-1 px-2 py-1 text-xs bg-gray-800 border border-gray-700 rounded text-gray-300">
                {VIDEO_RESOLUTION_OPTIONS.map(o => (
                  <option key={o.value} value={o.value}>{o.label}</option>
                ))}
              </select>
            </div>
            <Slider label="FPS" value={videoFps} onChange={setVideoFps} min={8} max={30} />
            <p className="text-xs text-gray-500 mt-1">Kandinsky: ~{Math.floor(videoDuration * 6 + 1)} frames at 24fps internal</p>
          </Section>

          <Section title="FreeU" toggle enabled={freeU} onToggle={setFreeU} />

          {/* Advanced toggle */}
          <div className="p-2 border-t border-gray-800">
            <label className="flex items-center gap-2 text-xs text-gray-400 cursor-pointer">
              <input type="checkbox" checked={showAdvanced} onChange={e => setShowAdvanced(e.target.checked)}
                className="accent-amber-500" />
              Display Advanced Options
            </label>
          </div>

          {showAdvanced && (
            <>
              <Section title="LoRAs">
                {loras.map((l, i) => (
                  <div key={i} className="flex items-center gap-1">
                    <input type="text" value={l.path} placeholder="LoRA path..."
                      onChange={e => { const u = [...loras]; u[i].path = e.target.value; setLoras(u); }}
                      className="flex-1 px-1 py-0.5 text-xs bg-gray-800 border border-gray-700 rounded" />
                    <input type="number" value={l.weight} step={0.1}
                      onChange={e => { const u = [...loras]; u[i].weight = Number(e.target.value); setLoras(u); }}
                      className="w-12 px-1 py-0.5 text-xs bg-gray-800 border border-gray-700 rounded" />
                    <button onClick={() => removeLora(i)} className="p-0.5 text-red-400 hover:text-red-300">
                      <Trash2 className="w-3 h-3" />
                    </button>
                  </div>
                ))}
                <button onClick={addLora} className="flex items-center gap-1 text-xs text-amber-500 hover:text-amber-400">
                  <Plus className="w-3 h-3" /> Add LoRA
                </button>
              </Section>
            </>
          )}
        </div>

        {/* Center - Preview */}
        <div className="flex-1 flex flex-col bg-gray-950">
          {/* Image preview area */}
          <div className="flex-1 flex items-center justify-center p-4 overflow-hidden relative">
            {selectedImage ? (
              <div className="relative group max-w-full max-h-full">
                <img
                  key={selectedImage.id}
                  src={`/api/gallery/${selectedImage.id}`}
                  alt=""
                  className="max-w-full max-h-full object-contain rounded"
                />
                {/* Quick action buttons - show on hover */}
                <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                  <button
                    onClick={useAsInitImage}
                    className="px-3 py-1.5 bg-gray-800/90 hover:bg-gray-700 text-white text-sm rounded flex items-center gap-1.5 shadow-lg"
                    title="Use as init image for img2img"
                  >
                    <ImageIcon className="w-4 h-4" /> img2img
                  </button>
                  <button
                    onClick={() => {
                      useAsInitImage();
                      setTimeout(() => setShowMaskEditor(true), 100);
                    }}
                    className="px-3 py-1.5 bg-purple-700/90 hover:bg-purple-600 text-white text-sm rounded flex items-center gap-1.5 shadow-lg"
                    title="Inpaint this image"
                  >
                    <Paintbrush className="w-4 h-4" /> Inpaint
                  </button>
                </div>
              </div>
            ) : (
              <div className="text-center text-gray-600">
                <p className="text-lg mb-2">Welcome to OneTrainer Inference</p>
                <p className="text-sm">Select a model and generate images</p>
              </div>
            )}
          </div>

          {/* Status bar - Loading / Progress */}
          {(isLoadingModel || isGenerating) && (
            <div className="px-4 py-2 bg-gray-900/80 border-t border-gray-800">
              <div className="flex items-center gap-3">
                {/* Spinner */}
                <div className="w-4 h-4 border-2 border-amber-500 border-t-transparent rounded-full animate-spin" />

                {/* Status text */}
                <span className="text-sm text-amber-400 font-medium">
                  {isLoadingModel && !isGenerating ? loadingMessage :
                   isGenerating && currentStep > 0 ? `Step ${currentStep}/${totalSteps}` :
                   isGenerating ? 'Starting generation...' : ''}
                </span>

                {/* Progress bar (only during generation) */}
                {isGenerating && totalSteps > 0 && (
                  <div className="flex-1 flex items-center gap-2">
                    <div className="flex-1 h-2 bg-gray-800 rounded overflow-hidden">
                      <div className="h-full bg-amber-500 transition-all duration-300"
                        style={{ width: `${(currentStep / totalSteps) * 100}%` }} />
                    </div>
                    <span className="text-xs text-gray-400 w-12 text-right">{Math.round((currentStep / totalSteps) * 100)}%</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Prompt area */}
          <div className="border-t border-gray-800 bg-gray-900 p-3">
            {error && (
              <div className="mb-2 px-3 py-1.5 bg-red-900/50 border border-red-700 rounded text-sm text-red-300 flex items-center">
                <span className="flex-1">{error}</span>
                <button onClick={() => setError(null)}><X className="w-4 h-4" /></button>
              </div>
            )}
            <div className="flex gap-2">
              <div className="flex-1 space-y-1">
                <div className="flex items-center gap-2">
                  <span className="text-green-400">+</span>
                  <textarea ref={promptRef} value={prompt} onChange={e => setPrompt(e.target.value)}
                    placeholder="Type your prompt here... (or drag/paste an image to use Image Prompting)"
                    className="flex-1 px-2 py-1.5 text-sm bg-gray-800 border border-gray-700 rounded resize-none h-8 focus:outline-none focus:border-amber-500"
                    onKeyDown={e => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), generate())} />
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-gray-500 text-xs">Neg:</span>
                  <input type="text" value={negPrompt} onChange={e => setNegPrompt(e.target.value)}
                    placeholder="Optionally, type a negative prompt here..."
                    className="flex-1 px-2 py-1 text-sm bg-gray-800 border border-gray-700 rounded focus:outline-none focus:border-amber-500" />
                </div>
              </div>
              <div className="flex flex-col gap-1">
                {isGenerating ? (
                  <button onClick={cancel}
                    className="px-6 py-3 bg-red-600 hover:bg-red-500 text-white font-medium rounded">
                    Cancel
                  </button>
                ) : (
                  <button onClick={generate} disabled={!modelPath && !modelType}
                    className="px-6 py-3 bg-amber-500 hover:bg-amber-400 text-black font-medium rounded disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1">
                    Generate <span className="text-xs">▼</span>
                  </button>
                )}
                <button className="p-1 text-gray-500 hover:text-gray-300 text-xs">⚙️</button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom bar */}
      <div className="bg-gray-900 border-t border-gray-800">
        {/* Bottom content area */}
        <div className="h-20 overflow-x-auto">
          {bottomTab === 'history' && (
            <div className="flex gap-1 p-1 h-full">
              {gallery.map(img => (
                <img key={img.id} src={img.thumbnail} alt=""
                  className={`h-full aspect-square object-cover rounded cursor-pointer border-2 ${selectedImage?.id === img.id ? 'border-amber-500' : 'border-transparent'}`}
                  onClick={() => setSelectedImage(img)} />
              ))}
              {gallery.length === 0 && <div className="flex items-center px-4 text-gray-500 text-sm">No images yet</div>}
            </div>
          )}
          {bottomTab === 'models' && (
            <div className="p-2 text-sm text-gray-400">Model browser coming soon...</div>
          )}
          {bottomTab === 'loras' && (
            <div className="p-2 text-sm text-gray-400">LoRA browser coming soon...</div>
          )}
        </div>

        {/* Bottom tabs & model selector */}
        <div className="flex items-center border-t border-gray-800 bg-gray-950">
          <div className="flex items-center px-2 py-1 border-r border-gray-800">
            <span className="text-xs text-gray-500 mr-2">Model:</span>
            <select value={modelType} onChange={e => { setModelType(e.target.value); setModelPath(''); }}
              className="px-2 py-1 text-sm bg-gray-800 border border-gray-700 rounded text-gray-300">
              {MODEL_OPTIONS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
            </select>
            {modelLoaded && <span className="ml-2 text-xs text-green-400">●</span>}
          </div>

          <button onClick={() => setBottomTab('history')}
            className={`px-4 py-2 text-sm ${bottomTab === 'history' ? 'text-amber-400' : 'text-gray-500 hover:text-gray-300'}`}>
            History
          </button>
          <button onClick={() => setBottomTab('models')}
            className={`px-4 py-2 text-sm ${bottomTab === 'models' ? 'text-amber-400' : 'text-gray-500 hover:text-gray-300'}`}>
            Models
          </button>
          <button className="px-4 py-2 text-sm text-gray-500 hover:text-gray-300">VAEs</button>
          <button onClick={() => setBottomTab('loras')}
            className={`px-4 py-2 text-sm ${bottomTab === 'loras' ? 'text-amber-400' : 'text-gray-500 hover:text-gray-300'}`}>
            LoRAs
          </button>
          <button className="px-4 py-2 text-sm text-gray-500 hover:text-gray-300">ControlNets</button>

          <div className="flex-1" />
          <div className="px-4 py-2 text-xs text-gray-600">OneTrainer Inference v1.0</div>
        </div>
      </div>
      </>
      )}
    </div>
  );
}
