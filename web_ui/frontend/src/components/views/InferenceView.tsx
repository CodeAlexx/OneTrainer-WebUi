import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { ChevronRight, ChevronDown, X, Plus, Trash2, Paintbrush, ImageIcon, Wand2 } from 'lucide-react';
import { MaskEditor } from '../inference/MaskEditor';

const API = axios.create({ baseURL: '/api/inference' });

// Types
type ModelType = 'flux_dev' | 'flux_schnell' | 'sdxl' | 'sd_35' | 'z_image' | 'qwen_image' | 'lumina_2' | 'omnigen_2' | 'wan_t2v' | string;
type GenerationMode = 'txt2img' | 'img2img' | 'inpaint';

interface LoRA { path: string; weight: number; enabled: boolean; }
interface GeneratedImage { id: string; path: string; thumbnail: string; prompt: string; seed: number; }

// Slider Component with Swarm-like Fill
const Slider = ({ label, value, onChange, min, max, step = 1 }: {
  label: string; value: number; onChange: (v: number) => void; min: number; max: number; step?: number;
}) => {
  const percentage = Math.min(100, Math.max(0, ((value - min) / (max - min)) * 100));

  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-gray-400 w-16 shrink-0">{label}</span>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))}
        className="flex-1 h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider-thumb-style"
        style={{
          background: `linear-gradient(to right, #f59e0b 0%, #f59e0b ${percentage}%, #374151 ${percentage}%, #374151 100%)`
        }}
      />
      <input type="number" value={value} onChange={e => onChange(Number(e.target.value))}
        className="w-14 px-1 py-0.5 text-xs bg-gray-800 border border-gray-700 rounded text-right text-gray-300 focus:border-amber-500 outline-none" />
    </div>
  );
};

// Collapsible Section Component with "Card" styling
const Section = ({ title, children, defaultOpen = false, toggle, enabled, onToggle }: {
  title: string; children?: React.ReactNode; defaultOpen?: boolean;
  toggle?: boolean; enabled?: boolean; onToggle?: (v: boolean) => void;
}) => {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border border-gray-800 rounded-lg bg-gray-900/40 mb-2 overflow-hidden">
      <div className={`flex items-center px-3 py-2 cursor-pointer select-none transition-colors ${open ? 'bg-gray-800/80 text-gray-200' : 'hover:bg-gray-800/50 text-gray-400'}`}
        onClick={() => setOpen(!open)}>
        {open ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
        <span className="ml-2 text-xs font-bold uppercase tracking-wide flex-1">{title}</span>
        {toggle && (
          <div className="ml-2" onClick={e => { e.stopPropagation(); onToggle?.(!enabled); }}>
            <div className={`w-8 h-4 rounded-full relative transition-colors ${enabled ? 'bg-amber-500' : 'bg-gray-600'}`}>
              <div className={`absolute top-0.5 w-3 h-3 rounded-full bg-white transition-all ${enabled ? 'left-4' : 'left-0.5'}`} />
            </div>
          </div>
        )}
      </div>
      {open && <div className="px-3 py-3 space-y-4 bg-gray-900/20 border-t border-gray-800">{children}</div>}
    </div>
  );
};



// Select Component
const Select = ({ label, value, onChange, options }: {
  label: string; value: string; onChange: (v: string) => void; options: string[] | { value: string; label: string }[];
}) => (
  <div className="flex items-center gap-2">
    <span className="text-xs text-gray-400 w-16 shrink-0">{label}</span>
    <select value={value} onChange={e => onChange(e.target.value)}
      className="flex-1 px-2 py-1 text-xs bg-gray-800 border border-gray-700 rounded text-gray-300">
      {options.map(o => typeof o === 'string' ?
        <option key={o} value={o}>{o}</option> :
        <option key={o.value} value={o.value}>{o.label}</option>
      )}
    </select>
  </div>
);

// Model options - loaded from settings API
interface ModelOption {
  value: string;
  label: string;
  path: string;
  category?: string;
  type?: string;
}

// Fallback hardcoded options in case API fails
const DEFAULT_MODEL_OPTIONS: ModelOption[] = [
  { value: 'FLUX_DEV_1', label: 'FLUX Dev', path: '/home/alex/SwarmUI/Models/diffusion_models/flux1-dev.safetensors', category: 'Image' },
  { value: 'kandinsky_5_video', label: 'Kandinsky 5 T2V Lite', path: '/home/alex/SwarmUI/Models/diffusion_models/kandinsky5lite_t2v_sft_5s.safetensors', category: 'Video' },
];

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
  const lower = model.toLowerCase();
  return lower.includes('video') || lower.includes('t2v') || lower.includes('i2v') ||
         lower.includes('wan') || lower.includes('kandinsky_5_video') || lower.includes('hunyuan');
};

export function InferenceView() {
  // Model options from settings
  const [modelOptions, setModelOptions] = useState<ModelOption[]>(DEFAULT_MODEL_OPTIONS);

  // Model state
  const [modelType, setModelType] = useState<ModelType>('FLUX_DEV_1');
  const [modelPath, setModelPath] = useState('');
  const [precision] = useState('bf16');
  const [modelLoaded, setModelLoaded] = useState(false);

  // Load models from settings API
  useEffect(() => {
    const loadModels = async () => {
      try {
        const res = await axios.get('/api/settings/models');
        if (res.data.models?.length > 0) {
          const options = res.data.models.map((m: any) => ({
            value: m.modelType,
            label: m.name,
            path: m.path,
            category: m.category,
            type: m.type,
          }));
          setModelOptions(options);
        }
      } catch (e) {
        console.error('Failed to load models from settings', e);
      }
    };
    loadModels();
  }, []);

  // Get path for model type
  const getModelPath = (mt: string): string => {
    const model = modelOptions.find(m => m.value === mt);
    return model?.path || mt;
  };

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
  const [videoDuration, setVideoDuration] = useState(5);
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
      } catch { }
    };
    poll();
    const i = setInterval(poll, 500); // Poll faster for better UX
    return () => clearInterval(i);
  }, []);

  // Load gallery
  useEffect(() => {
    API.get('/gallery').then(r => setGallery(r.data.images || [])).catch(() => { });
  }, []);

  // Parse resolution - use videoResolution for video models
  const effectiveResolution = isVideoModel(modelType) ? videoResolution : resolution;
  const [width, height] = effectiveResolution.split('x').map(Number);

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
    const selectedModel = modelOptions.find(m => m.value === modelType);
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

      let newImages: any[] = [];
      if (r.data.images?.length) {
        newImages = r.data.images;
      } else if (r.data.image) {
        newImages = [r.data.image];
      }

      if (newImages.length > 0) {
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
      const response = await fetch(`/api/inference/gallery/${selectedImage.id}`);
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
    <div className="h-full flex flex-col bg-gray-950 text-gray-200 overflow-hidden">
      <style>{`
        .slider-thumb-style::-webkit-slider-thumb {
          -webkit-appearance: none;
          appearance: none;
          width: 12px;
          height: 12px;
          border-radius: 50%;
          background: #ffffff;
          border: 2px solid #f59e0b;
          cursor: pointer;
          margin-top: 0px; box-shadow: 0 0 2px rgba(0,0,0,0.5);
        }
        .slider-thumb-style::-moz-range-thumb {
          width: 12px;
          height: 12px;
          border-radius: 50%;
          background: #ffffff;
          border: 2px solid #f59e0b;
          cursor: pointer;
          box-shadow: 0 0 2px rgba(0,0,0,0.5);
        }
      `}</style>

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

      {/* Main Content Area - Horizontal Layout */}
      <div className="flex-1 flex overflow-hidden">

        {/* LEFT SIDEBAR - Parameters */}
        <div className="w-80 bg-gray-900 border-r border-gray-800 flex flex-col z-10">
          {/* Tab Buttons (Txt2Img etc) */}
          <div className="flex border-b border-gray-800">
            <button
              onClick={() => { setMode('txt2img'); }}
              className={`flex-1 py-3 text-xs flex justify-center items-center gap-1 ${mode === 'txt2img' ? 'bg-gray-800 text-amber-500 border-b-2 border-amber-500' : 'text-gray-400 hover:text-gray-200'}`}
              title="Text to Image"
            >
              <Wand2 className="w-4 h-4" />
            </button>
            <button
              onClick={() => { setMode('img2img'); }}
              className={`flex-1 py-3 text-xs flex justify-center items-center gap-1 ${mode === 'img2img' ? 'bg-gray-800 text-amber-500 border-b-2 border-amber-500' : 'text-gray-400 hover:text-gray-200'}`}
              title="Image to Image"
            >
              <ImageIcon className="w-4 h-4" />
            </button>
            <button
              onClick={() => { setMode('inpaint'); }}
              className={`flex-1 py-3 text-xs flex justify-center items-center gap-1 ${mode === 'inpaint' ? 'bg-gray-800 text-amber-500 border-b-2 border-amber-500' : 'text-gray-400 hover:text-gray-200'}`}
              title="Inpainting"
            >
              <Paintbrush className="w-4 h-4" />
            </button>
          </div>

          {/* Parameter Scroll Area */}
          <div className="flex-1 overflow-y-auto">
            <div className="p-2 border-b border-gray-800">
              <input type="text" placeholder="Filter parameters..."
                className="w-full px-2 py-1 text-xs bg-gray-800 border border-gray-700 rounded text-gray-300 placeholder-gray-500" />
            </div>

            <Section title="Core Parameters" defaultOpen>
              <div className="space-y-4">
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
              </div>
            </Section>

            <Section title="Variation Seed" toggle enabled={variationSeed} onToggle={setVariationSeed}>
              {variationSeed && <Slider label="Strength" value={0.5} onChange={() => { }} min={0} max={1} step={0.05} />}
            </Section>

            <Section title="Resolution" defaultOpen>
              <Select label="Preset" value={resolution} onChange={setResolution} options={RESOLUTION_OPTIONS} />
            </Section>

            <Section title="Sampling" defaultOpen>
              <Select label="Sampler" value={sampler} onChange={setSampler} options={SAMPLER_OPTIONS} />
            </Section>

            <Section title="Init Image" defaultOpen={mode !== 'txt2img'}>
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
                      Use selected
                    </button>
                  )}
                </div>
              )}
            </Section>

            <Section title="Refine / Upscale" toggle enabled={refineEnabled} onToggle={setRefineEnabled}>
              {refineEnabled && <Slider label="Scale" value={refineScale} onChange={setRefineScale} min={1} max={4} step={0.5} />}
            </Section>

            <Section title="ControlNet" toggle enabled={cnEnabled} onToggle={setCnEnabled}>
              {cnEnabled && <Select label="Model" value="" onChange={() => { }} options={['canny', 'depth', 'pose']} />}
            </Section>

            <Section title="VIDEO SETTINGS" defaultOpen={isVideoModel(modelType)}>
              <Slider label="Duration" value={videoDuration} onChange={setVideoDuration} min={1} max={10} />
              <Slider label="Frames" value={numFrames} onChange={setNumFrames} min={4} max={64} />
              <div className="flex items-center gap-2">
                <span className="text-xs text-gray-400 w-16 shrink-0">Resolution</span>
                <select value={videoResolution} onChange={e => setVideoResolution(e.target.value)}
                  className="flex-1 px-2 py-1 text-xs bg-gray-800 border border-gray-700 rounded text-gray-300">
                  {VIDEO_RESOLUTION_OPTIONS.map(o => (
                    <option key={o.value} value={o.value}>{o.label}</option>
                  ))}
                </select>
              </div>
              <Slider label="FPS" value={videoFps} onChange={setVideoFps} min={8} max={30} />
              <p className="text-xs text-gray-500">K5: ~{Math.floor(videoDuration * 6 + 1)} frames</p>
            </Section>

            <Section title="FreeU" toggle enabled={freeU} onToggle={setFreeU} />

            <Section title="LoRAs" defaultOpen={true}>
              {loras.map((l, i) => (
                <div key={i} className="flex items-center gap-1 mb-2">
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
          </div>
        </div>

        {/* CENTER - Image and Prompt */}
        <div className="flex-1 flex flex-col bg-gray-950 min-w-0">
          {/* Status Bar */}
          {(isLoadingModel || isGenerating) && (
            <div className="px-4 py-2 bg-gray-900 border-b border-gray-800 flex items-center gap-3">
              <div className="w-4 h-4 border-2 border-amber-500 border-t-transparent rounded-full animate-spin" />
              <span className="text-sm text-amber-400 font-medium flex-1">
                {isLoadingModel && !isGenerating ? loadingMessage :
                  isGenerating && currentStep > 0 ? `Step ${currentStep}/${totalSteps}` :
                    isGenerating ? 'Starting generation...' : ''}
              </span>
              {isGenerating && totalSteps > 0 && (
                <div className="w-64 flex items-center gap-2">
                  <div className="flex-1 h-2 bg-gray-800 rounded overflow-hidden">
                    <div className="h-full bg-amber-500 transition-all duration-300" style={{ width: `${(currentStep / totalSteps) * 100}%` }} />
                  </div>
                  <span className="text-xs text-gray-400 w-10 text-right">{Math.round((currentStep / totalSteps) * 100)}%</span>
                </div>
              )}
            </div>
          )}

          {/* Image Preview Container */}
          <div className="flex-1 flex items-center justify-center p-6 bg-gray-950 relative overflow-hidden">
            {selectedImage ? (
              <div className="relative group max-w-full max-h-full shadow-2xl">
                <img
                  key={selectedImage.id}
                  src={`/api/inference/gallery/${selectedImage.id}`}
                  alt=""
                  className="max-w-full max-h-full object-contain rounded border border-gray-800"
                  style={{ maxHeight: 'calc(100vh - 250px)' }} // Ensure space for prompt bar
                />
                {/* Overlay Actions */}
                <div className="absolute top-2 right-2 flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                  <button onClick={useAsInitImage} className="px-2 py-1 bg-black/60 hover:bg-black/80 text-white text-xs rounded border border-white/20">Use As Init</button>
                  <button onClick={() => { useAsInitImage(); setTimeout(() => setShowMaskEditor(true), 100); }} className="px-2 py-1 bg-black/60 hover:bg-black/80 text-white text-xs rounded border border-white/20">Inpaint</button>
                </div>
              </div>
            ) : (
              <div className="text-center text-gray-600">
                <div className="mb-4 text-6xl opacity-10">🎨</div>
                <p className="text-lg mb-2 text-gray-500">Ready to Generate</p>
                <p className="text-sm text-gray-600">Select a model from the right sidebar to begin</p>
              </div>
            )}
          </div>

          {/* Prompt Bar (Fixed at bottom of center) */}
          <div className="p-4 bg-gray-900 border-t border-gray-800 z-10">
            {error && (
              <div className="mb-2 px-3 py-1.5 bg-red-900/50 border border-red-700 rounded text-sm text-red-300 flex items-center justify-between">
                <span>{error}</span>
                <button onClick={() => setError(null)}><X className="w-4 h-4" /></button>
              </div>
            )}

            <div className="flex gap-2 items-end">
              <button className="p-2 h-10 bg-gray-800 border border-gray-700 rounded hover:bg-gray-750 text-amber-500" title="Add token">
                <Plus className="w-5 h-5" />
              </button>

              <div className="flex-1 flex flex-col gap-2">
                <textarea
                  ref={promptRef}
                  value={prompt}
                  onChange={e => setPrompt(e.target.value)}
                  placeholder="Type your prompt here..."
                  className="w-full px-3 py-2 text-sm bg-gray-800 border border-gray-700 rounded focus:outline-none focus:border-amber-500 min-h-[5rem] resize-none"
                  onKeyDown={e => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), generate())}
                />
                <input
                  type="text"
                  value={negPrompt}
                  onChange={e => setNegPrompt(e.target.value)}
                  placeholder="Negative prompt..."
                  className="w-full px-3 py-1.5 text-xs bg-gray-800 border border-gray-700 rounded focus:outline-none focus:border-amber-500 text-gray-400"
                />
              </div>

              <div className="flex flex-col gap-2 h-full">
                {isGenerating ? (
                  <button onClick={cancel} className="h-full px-6 bg-red-600 hover:bg-red-500 text-white font-bold rounded uppercase tracking-wide text-sm">
                    Cancel
                  </button>
                ) : (
                  <button onClick={generate} disabled={!modelPath && !modelType}
                    className="h-full px-8 bg-amber-500 hover:bg-amber-400 text-black font-bold rounded uppercase tracking-wide text-sm shadow-lg shadow-amber-900/20 disabled:opacity-50 disabled:shadow-none">
                    Generate
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* RIGHT SIDEBAR - Gallery and Models */}
        <div className="w-80 bg-gray-900 border-l border-gray-800 flex flex-col z-10">
          {/* Model Selector Area */}
          <div className="p-3 border-b border-gray-800 bg-gray-900">
            <label className="text-xs text-gray-500 block mb-1">Model</label>
            <select value={modelType} onChange={e => { setModelType(e.target.value); setModelPath(''); }}
              className="w-full px-2 py-1.5 text-sm bg-gray-800 border border-gray-700 rounded text-gray-200 focus:outline-none focus:border-amber-500">
              {modelOptions.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
            </select>
            <div className="mt-2 flex items-center justify-between">
              <span className={`text-xs flex items-center gap-1 ${modelLoaded ? 'text-green-400' : 'text-gray-500'}`}>
                <div className={`w-2 h-2 rounded-full ${modelLoaded ? 'bg-green-400' : 'bg-gray-600'}`} />
                {modelLoaded ? 'Loaded' : 'Not Loaded'}
              </span>
              <button className="text-xs text-gray-500 hover:text-gray-300">Rescan</button>
            </div>
          </div>

          {/* Gallery Header */}
          <div className="px-3 py-2 bg-gray-850 flex justify-between items-center border-b border-gray-800">
            <span className="text-xs font-medium text-gray-400">Session Gallery</span>
            <span className="text-xs text-gray-600">{gallery.length} images</span>
          </div>

          {/* Vertical Scrolling Gallery */}
          <div className="flex-1 overflow-y-auto p-2 space-y-2">
            {isGenerating && (
              <div className="relative aspect-square rounded overflow-hidden border-2 border-amber-500/50 bg-gray-800 flex flex-col items-center justify-center">
                <div className="relative w-16 h-16 mb-2">
                  <svg className="w-full h-full rotate-[-90deg]">
                    <circle cx="32" cy="32" r="28" stroke="currentColor" strokeWidth="6" fill="transparent" className="text-gray-700" />
                    <circle
                      cx="32" cy="32" r="28"
                      stroke="currentColor" strokeWidth="6"
                      fill="transparent"
                      className="text-amber-500 transition-all duration-300"
                      strokeDasharray={176}
                      strokeDashoffset={176 - (176 * (totalSteps > 0 ? currentStep / totalSteps : 0))}
                      strokeLinecap="round"
                    />
                  </svg>
                  <div className="absolute inset-0 flex items-center justify-center text-xs font-bold text-amber-500">
                    {Math.round((totalSteps > 0 ? currentStep / totalSteps : 0) * 100)}%
                  </div>
                </div>
                <span className="text-xs text-amber-400 animate-pulse">Generating...</span>
                <span className="text-[10px] text-gray-500 mt-1">Step {currentStep}/{totalSteps}</span>
              </div>
            )}

            {gallery.length === 0 && !isGenerating ? (
              <div className="text-center py-10 text-gray-600 text-xs">
                No images generated yet
              </div>
            ) : (
              gallery.map(img => (
                <div
                  key={img.id}
                  onClick={() => setSelectedImage(img)}
                  className={`relative aspect-square rounded overflow-hidden cursor-pointer border-2 bg-gray-800 group ${selectedImage?.id === img.id ? 'border-amber-500 ring-1 ring-amber-500/50' : 'border-gray-800 hover:border-gray-600'}`}
                >
                  <img
                    src={img.thumbnail || `/api/inference/gallery/${img.id}`}
                    onError={(e) => { (e.target as HTMLImageElement).src = `/api/inference/gallery/${img.id}`; }}
                    alt=""
                    className="w-full h-full object-cover"
                  />
                  {selectedImage?.id !== img.id && (
                    <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors" />
                  )}
                </div>
              ))
            )}
          </div>
        </div>

      </div>
    </div>
  );
}
