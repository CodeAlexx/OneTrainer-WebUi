import { useState, useEffect, useCallback } from 'react';
import { Play, Shuffle, Save, Plus, Trash2, Image as ImageIcon, XCircle, RefreshCw, Square, Upload, Video, Edit3, Layers, Paintbrush } from 'lucide-react';
import { inferenceApi, InferenceState, GeneratedImage, GenerateParams } from '../../lib/api';
import { useConfigStore } from '../../stores/configStore';

type GenerationMode = 'txt2img' | 'img2img' | 'inpainting' | 'edit' | 'video';

const GENERATION_MODES: { value: GenerationMode; label: string; icon: any; description: string }[] = [
  { value: 'txt2img', label: 'Text to Image', icon: ImageIcon, description: 'Generate from text prompt' },
  { value: 'img2img', label: 'Image to Image', icon: Layers, description: 'Transform an existing image' },
  { value: 'inpainting', label: 'Inpainting', icon: Paintbrush, description: 'Edit parts of an image with mask' },
  { value: 'edit', label: 'Edit', icon: Edit3, description: 'Instruction-based editing (Z-Edit, Qwen-Edit)' },
  { value: 'video', label: 'Video', icon: Video, description: 'Generate video (Wan, Hunyuan Video)' },
];

const MODEL_TYPES = [
  // FLUX models
  'FLUX_DEV_1', 'FLUX_FILL_DEV_1', 'FLUX_SCHNELL', 'FLUX_2_DEV',
  // Stable Diffusion
  'STABLE_DIFFUSION_XL_10_BASE', 'STABLE_DIFFUSION_15', 'STABLE_DIFFUSION_3', 'STABLE_DIFFUSION_35',
  // PixArt
  'PIXART_ALPHA', 'PIXART_SIGMA',
  // Sana
  'SANA',
  // Video models
  'HUNYUAN_VIDEO', 'WAN_T2V', 'WAN_I2V',
  // Other image models
  'HI_DREAM_FULL', 'CHROMA_1',
  // Z-Image (Alibaba/Tongyi)
  'Z_IMAGE', 'Z_IMAGE_TURBO', 'Z_IMAGE_EDIT',
  // Qwen Image
  'QWEN_IMAGE', 'QWEN_IMAGE_EDIT', 'QWEN_IMAGE_LAYERED',
  // Kandinsky
  'KANDINSKY_5',
];

const SAMPLERS = ['Euler', 'Euler a', 'DPM++ 2M', 'DPM++ 2M Karras', 'DPM++ SDE', 'DPM++ SDE Karras', 'DDIM', 'UniPC', 'LCM', 'Flow Match'];
const SCHEDULERS = ['Simple', 'Normal', 'Karras', 'Exponential', 'SGM Uniform', 'AYS'];
const PRECISIONS = ['fp8_e4m3fn', 'fp16', 'bf16', 'fp32'];

interface LoraEntry {
  id: string;
  path: string;
  weight: number;
  enabled: boolean;
}

export function InferenceView() {
  // Get training config for pre-filling fields
  const { config } = useConfigStore();

  // Backend state
  const [inferenceState, setInferenceState] = useState<InferenceState | null>(null);
  const [gallery, setGallery] = useState<GeneratedImage[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  // Model settings
  const [modelPath, setModelPath] = useState('');
  const [modelType, setModelType] = useState('FLUX_DEV_1');
  const [precision, setPrecision] = useState('bf16');
  const [vaePath, setVaePath] = useState('');
  const [sampler, setSampler] = useState('Euler');
  const [scheduler, setScheduler] = useState('Simple');
  const [steps, setSteps] = useState(30);
  const [cfgScale, setCfgScale] = useState(5.0);
  const [width, setWidth] = useState(1024);
  const [height, setHeight] = useState(1024);
  const [seed, setSeed] = useState(-1);
  const [batchSize, setBatchSize] = useState(1);
  const [batchCount, setBatchCount] = useState(1);

  // LoRA
  const [loras, setLoras] = useState<LoraEntry[]>([]);

  // Prompts
  const [prompt, setPrompt] = useState('');
  const [negativePrompt, setNegativePrompt] = useState('');

  // Generation mode
  const [generationMode, setGenerationMode] = useState<GenerationMode>('txt2img');
  const [initImagePath, setInitImagePath] = useState('');
  const [maskImagePath, setMaskImagePath] = useState('');
  const [strength, setStrength] = useState(0.75);
  const [editInstruction, setEditInstruction] = useState('');
  const [numFrames, setNumFrames] = useState(16);
  const [fps, setFps] = useState(8);

  // Selected image
  const [selectedImage, setSelectedImage] = useState<GeneratedImage | null>(null);

  // Track if we've already attempted auto-load
  const [autoLoadTriggered, setAutoLoadTriggered] = useState(false);

  // Pre-fill from training config
  useEffect(() => {
    const c = config as any;
    if (c) {
      // Pre-fill base model path
      if (c.base_model_name && !modelPath) {
        setModelPath(c.base_model_name);
      }
      // Pre-fill model type
      if (c.model_type && MODEL_TYPES.includes(c.model_type)) {
        setModelType(c.model_type);
      }
      // Pre-fill trained LoRA if output exists and training_method is LORA
      if (c.output_model_destination && c.training_method === 'LORA') {
        const loraPath = c.output_model_destination;
        // Check if already added
        if (!loras.some(l => l.path === loraPath)) {
          setLoras(prev => [...prev, {
            id: 'trained-lora',
            path: loraPath,
            weight: 1.0,
            enabled: true,
          }]);
        }
      }
    }
  }, [config]);

  // Auto-load model when config is available and model isn't loaded
  useEffect(() => {
    const c = config as any;
    // Only auto-load if:
    // - Config has a base model path
    // - Model is not currently loaded
    // - We haven't already tried auto-loading
    // - Not currently loading
    if (
      c?.base_model_name &&
      inferenceState !== null &&
      !inferenceState.model_loaded &&
      !autoLoadTriggered &&
      !loading
    ) {
      setAutoLoadTriggered(true);
      // Trigger load after a short delay to ensure state is settled
      const modelPathToLoad = c.base_model_name;
      const modelTypeToLoad = c.model_type && MODEL_TYPES.includes(c.model_type) ? c.model_type : 'FLUX_DEV_1';
      const loraPathsToLoad: string[] = [];

      if (c.output_model_destination && c.training_method === 'LORA') {
        loraPathsToLoad.push(c.output_model_destination);
      }

      console.log('Auto-loading model:', modelPathToLoad, 'with LoRAs:', loraPathsToLoad);

      inferenceApi.loadModel(modelPathToLoad, modelTypeToLoad, loraPathsToLoad.length > 0 ? loraPathsToLoad : undefined)
        .then(() => fetchData())
        .catch((err: any) => {
          console.error('Auto-load failed:', err);
          setError('Auto-load failed: ' + (err.response?.data?.detail || err.message));
        });
    }
  }, [config, inferenceState, autoLoadTriggered, loading]);

  // Suppress unused warnings - will be used when full inference is connected
  void precision; void vaePath; void sampler; void scheduler; void batchSize; void batchCount;


  // Fetch status and gallery
  const fetchData = useCallback(async () => {
    try {
      const [statusRes, galleryRes] = await Promise.all([
        inferenceApi.getStatus(),
        inferenceApi.getGallery(50),
      ]);
      setInferenceState(statusRes.data);
      setGallery(galleryRes.data.images);
      setError(null);
    } catch (err: any) {
      console.error('Failed to fetch inference status:', err);
      setError(err.response?.data?.detail || err.message || 'Failed to fetch status');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Poll while generating
  useEffect(() => {
    if (!inferenceState?.is_generating) return;
    const interval = setInterval(fetchData, 1000);
    return () => clearInterval(interval);
  }, [inferenceState?.is_generating, fetchData]);

  const addLora = () => {
    setLoras([...loras, { id: Date.now().toString(), path: '', weight: 1.0, enabled: true }]);
  };

  const updateLora = (id: string, field: keyof LoraEntry, value: any) => {
    setLoras(loras.map(l => l.id === id ? { ...l, [field]: value } : l));
  };

  const removeLora = (id: string) => {
    setLoras(loras.filter(l => l.id !== id));
  };

  const randomizeSeed = () => {
    setSeed(Math.floor(Math.random() * 2147483647));
  };

  // Load model
  const handleLoadModel = async () => {
    if (!modelPath.trim()) {
      setError('Please enter a model path');
      return;
    }
    try {
      setError(null);
      const loraPaths = loras.filter(l => l.enabled && l.path.trim()).map(l => l.path.trim());
      await inferenceApi.loadModel(modelPath, modelType, loraPaths.length > 0 ? loraPaths : undefined);
      await fetchData();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load model');
    }
  };

  // Unload model
  const handleUnloadModel = async () => {
    try {
      await inferenceApi.unloadModel();
      await fetchData();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to unload model');
    }
  };

  // Generate image
  const handleGenerate = async () => {
    if (!prompt.trim() && generationMode !== 'edit') {
      setError('Please enter a prompt');
      return;
    }
    if ((generationMode === 'img2img' || generationMode === 'inpainting' || generationMode === 'edit') && !initImagePath.trim()) {
      setError('Please provide an init image path');
      return;
    }
    if (generationMode === 'inpainting' && !maskImagePath.trim()) {
      setError('Please provide a mask image path');
      return;
    }
    try {
      setError(null);
      const params: GenerateParams = {
        prompt: prompt.trim(),
        negative_prompt: negativePrompt.trim(),
        width,
        height,
        steps,
        guidance_scale: cfgScale,
        seed,
        batch_size: batchSize,
        mode: generationMode,
        init_image_path: initImagePath.trim(),
        mask_image_path: maskImagePath.trim(),
        strength,
        num_frames: numFrames,
        fps,
        edit_instruction: editInstruction.trim(),
      };
      const result = await inferenceApi.generate(params);
      if (result.data.image) {
        setSelectedImage(result.data.image);
      }
      await fetchData();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Generation failed');
    }
  };

  // Cancel generation
  const handleCancel = async () => {
    try {
      await inferenceApi.cancelGeneration();
      await fetchData();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to cancel');
    }
  };

  // Delete image
  const handleDeleteImage = async (imageId: string) => {
    try {
      await inferenceApi.deleteImage(imageId);
      if (selectedImage?.id === imageId) {
        setSelectedImage(null);
      }
      await fetchData();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to delete');
    }
  };

  // Clear gallery
  const handleClearGallery = async () => {
    try {
      await inferenceApi.clearGallery();
      setSelectedImage(null);
      setGallery([]);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to clear gallery');
    }
  };

  // Use image settings
  const handleUseImageSettings = (image: GeneratedImage) => {
    setPrompt(image.prompt);
    setNegativePrompt(image.negative_prompt);
    setWidth(image.width);
    setHeight(image.height);
    setSteps(image.steps);
    setCfgScale(image.guidance_scale);
    setSeed(image.seed);
  };

  const isModelLoaded = inferenceState?.model_loaded ?? false;
  const isGenerating = inferenceState?.is_generating ?? false;

  return (
    <div className="h-full flex">
      {/* Left Panel - Settings */}
      <div className="w-72 bg-dark-surface border-r border-dark-border flex flex-col overflow-y-auto">
        <div className="p-3 border-b border-dark-border bg-dark-surface flex items-center justify-between">
          <h2 className="text-sm font-medium text-white">Model Config</h2>
          <button onClick={fetchData} disabled={loading} className="text-muted hover:text-white">
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>

        {/* Error banner */}
        {error && (
          <div className="mx-3 mt-3 px-3 py-2 bg-red-900/20 border border-red-800/50 rounded flex items-center justify-between">
            <span className="text-red-400 text-xs">{error}</span>
            <button onClick={() => setError(null)} className="text-red-400 hover:text-red-300">
              <XCircle className="w-3 h-3" />
            </button>
          </div>
        )}

        <div className="p-3 space-y-3">
          {/* Model Status */}
          {isModelLoaded ? (
            <div className="p-3 bg-cyan-600/10 border border-cyan-600/30 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <div className="w-2 h-2 bg-cyan-500 rounded-full" />
                <span className="text-cyan-400 text-sm font-medium">Model Loaded</span>
              </div>
              <div className="text-xs text-muted truncate mb-2">
                {inferenceState?.model_path}
              </div>
              <button
                onClick={handleUnloadModel}
                className="w-full px-3 py-1.5 text-sm text-muted border border-dark-border rounded hover:bg-dark-hover"
              >
                Unload Model
              </button>
            </div>
          ) : (
            <>
              {/* Model */}
              <div>
                <label className="text-xs text-muted block mb-1">Model</label>
                <input type="text" value={modelPath} onChange={(e) => setModelPath(e.target.value)}
                  className="input w-full text-sm" placeholder="Path to model..." />
              </div>

              <div>
                <label className="text-xs text-muted block mb-1">Model Type</label>
                <select value={modelType} onChange={(e) => setModelType(e.target.value)} className="input w-full text-sm">
                  {MODEL_TYPES.map(m => <option key={m} value={m}>{m}</option>)}
                </select>
              </div>

              <div>
                <label className="text-xs text-muted block mb-1">Precision</label>
                <select value={precision} onChange={(e) => setPrecision(e.target.value)} className="input w-full text-sm">
                  {PRECISIONS.map(p => <option key={p} value={p}>{p}</option>)}
                </select>
              </div>

              <div>
                <label className="text-xs text-muted block mb-1">VAE</label>
                <input type="text" value={vaePath} onChange={(e) => setVaePath(e.target.value)}
                  className="input w-full text-sm" placeholder="Optional VAE override..." />
              </div>

              <button
                onClick={handleLoadModel}
                className="w-full px-3 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg text-sm font-medium flex items-center justify-center gap-2"
              >
                <Upload className="w-4 h-4" />
                Load Model
              </button>
            </>
          )}

          <div className="border-t border-dark-border pt-3">
            <label className="text-xs text-muted block mb-1">Sampler</label>
            <select value={sampler} onChange={(e) => setSampler(e.target.value)} className="input w-full text-sm">
              {SAMPLERS.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>

          <div>
            <label className="text-xs text-muted block mb-1">Scheduler</label>
            <select value={scheduler} onChange={(e) => setScheduler(e.target.value)} className="input w-full text-sm">
              {SCHEDULERS.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>

          <div>
            <label className="text-xs text-muted block mb-1">Steps</label>
            <input type="text" value={steps} onChange={(e) => setSteps(parseInt(e.target.value) || 1)}
              className="input w-full text-sm" min="1" max="150" />
          </div>

          <div>
            <label className="text-xs text-muted block mb-1">CFG Scale</label>
            <input type="text" value={cfgScale} onChange={(e) => setCfgScale(parseFloat(e.target.value) || 1)}
              className="input w-full text-sm" min="1" max="30" step="0.5" />
          </div>

          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="text-xs text-muted block mb-1">Width</label>
              <input type="text" value={width} onChange={(e) => setWidth(parseInt(e.target.value) || 512)}
                className="input w-full text-sm" min="256" max="2048" step="64" />
            </div>
            <div>
              <label className="text-xs text-muted block mb-1">Height</label>
              <input type="text" value={height} onChange={(e) => setHeight(parseInt(e.target.value) || 512)}
                className="input w-full text-sm" min="256" max="2048" step="64" />
            </div>
          </div>

          {/* LoRA Section */}
          <div className="border-t border-dark-border pt-3">
            <div className="flex items-center justify-between mb-2">
              <label className="text-xs text-muted uppercase tracking-wider">LoRA / Addons</label>
              <button onClick={addLora} className="text-muted hover:text-white">
                <Plus className="w-4 h-4" />
              </button>
            </div>

            {loras.length === 0 ? (
              <p className="text-xs text-muted italic">No LoRAs added</p>
            ) : (
              <div className="space-y-2">
                {loras.map((lora) => (
                  <div key={lora.id} className="bg-dark-bg rounded border border-dark-border p-2">
                    <div className="flex items-center gap-2 mb-2">
                      <input
                        type="checkbox"
                        checked={lora.enabled}
                        onChange={(e) => updateLora(lora.id, 'enabled', e.target.checked)}
                        className="rounded"
                      />
                      <input
                        type="text"
                        value={lora.path}
                        onChange={(e) => updateLora(lora.id, 'path', e.target.value)}
                        className="input flex-1 text-xs"
                        placeholder="LoRA path..."
                      />
                      <button onClick={() => removeLora(lora.id)} className="text-muted hover:text-danger">
                        <Trash2 className="w-3 h-3" />
                      </button>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-muted">Weight:</span>
                      <input
                        type="text"
                        value={lora.weight}
                        onChange={(e) => updateLora(lora.id, 'weight', parseFloat(e.target.value) || 0)}
                        className="input w-20 text-xs"
                        min="-2" max="2" step="0.1"
                      />
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Seed & Batch */}
          <div className="border-t border-dark-border pt-3">
            <div className="flex items-center gap-2">
              <div className="flex-1">
                <label className="text-xs text-muted block mb-1">Seed</label>
                <input type="text" value={seed} onChange={(e) => setSeed(parseInt(e.target.value))}
                  className="input w-full text-sm" />
              </div>
              <button onClick={randomizeSeed} className="mt-5 p-2 text-muted hover:text-white">
                <Shuffle className="w-4 h-4" />
              </button>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="text-xs text-muted block mb-1">Batch Size</label>
              <input type="text" value={batchSize} onChange={(e) => setBatchSize(parseInt(e.target.value) || 1)}
                className="input w-full text-sm" min="1" max="8" />
            </div>
            <div>
              <label className="text-xs text-muted block mb-1">Batches</label>
              <input type="text" value={batchCount} onChange={(e) => setBatchCount(parseInt(e.target.value) || 1)}
                className="input w-full text-sm" min="1" max="100" />
            </div>
          </div>
        </div>
      </div>

      {/* Center Panel - Prompt & Output */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Mode Selector */}
        <div className="p-3 border-b border-dark-border">
          <div className="flex items-center gap-1">
            {GENERATION_MODES.map((mode) => {
              const Icon = mode.icon;
              return (
                <button
                  key={mode.value}
                  onClick={() => setGenerationMode(mode.value)}
                  title={mode.description}
                  className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors ${
                    generationMode === mode.value
                      ? 'bg-cyan-600 text-white'
                      : 'bg-dark-bg text-muted hover:bg-dark-hover hover:text-white'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  {mode.label}
                </button>
              );
            })}
          </div>
        </div>

        {/* Mode-specific inputs */}
        {(generationMode === 'img2img' || generationMode === 'inpainting' || generationMode === 'edit') && (
          <div className="p-3 border-b border-dark-border bg-dark-bg/50 space-y-3">
            <div>
              <label className="text-xs text-muted block mb-1">Init Image Path</label>
              <input
                type="text"
                value={initImagePath}
                onChange={(e) => setInitImagePath(e.target.value)}
                className="input w-full text-sm"
                placeholder="/path/to/image.png"
              />
            </div>
            {generationMode === 'inpainting' && (
              <div>
                <label className="text-xs text-muted block mb-1">Mask Image Path</label>
                <input
                  type="text"
                  value={maskImagePath}
                  onChange={(e) => setMaskImagePath(e.target.value)}
                  className="input w-full text-sm"
                  placeholder="/path/to/mask.png (white = inpaint area)"
                />
              </div>
            )}
            {generationMode === 'edit' && (
              <div>
                <label className="text-xs text-muted block mb-1">Edit Instruction</label>
                <input
                  type="text"
                  value={editInstruction}
                  onChange={(e) => setEditInstruction(e.target.value)}
                  className="input w-full text-sm"
                  placeholder="e.g., Change the shirt color to blue"
                />
              </div>
            )}
            {(generationMode === 'img2img' || generationMode === 'inpainting') && (
              <div>
                <label className="text-xs text-muted block mb-1">Denoising Strength: {strength.toFixed(2)}</label>
                <input
                  type="range"
                  value={strength}
                  onChange={(e) => setStrength(parseFloat(e.target.value))}
                  className="w-full"
                  min="0" max="1" step="0.05"
                />
              </div>
            )}
          </div>
        )}

        {generationMode === 'video' && (
          <div className="p-3 border-b border-dark-border bg-dark-bg/50 grid grid-cols-2 gap-3">
            <div>
              <label className="text-xs text-muted block mb-1">Frames</label>
              <input
                type="text"
                value={numFrames}
                onChange={(e) => setNumFrames(parseInt(e.target.value) || 16)}
                className="input w-full text-sm"
                min="1" max="128"
              />
            </div>
            <div>
              <label className="text-xs text-muted block mb-1">FPS</label>
              <input
                type="text"
                value={fps}
                onChange={(e) => setFps(parseInt(e.target.value) || 8)}
                className="input w-full text-sm"
                min="1" max="60"
              />
            </div>
          </div>
        )}

        {/* Prompt Area */}
        <div className="p-4 border-b border-dark-border space-y-3">
          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="text-xs text-muted uppercase tracking-wider">Prompt</label>
              <span className="text-xs text-muted">{prompt.length} chars</span>
            </div>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="input w-full h-24 text-sm resize-none"
              placeholder="Enter your prompt here..."
            />
          </div>

          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="text-xs text-muted uppercase tracking-wider">Negative Prompt</label>
            </div>
            <textarea
              value={negativePrompt}
              onChange={(e) => setNegativePrompt(e.target.value)}
              className="input w-full h-16 text-sm resize-none"
              placeholder="Enter negative prompt (optional)..."
            />
          </div>
        </div>

        {/* Image Output */}
        <div className="flex-1 p-4 flex items-center justify-center bg-dark-bg overflow-hidden">
          {selectedImage ? (
            <div className="max-w-full max-h-full flex flex-col items-center">
              <img
                src={inferenceApi.getImage(selectedImage.id)}
                alt={selectedImage.prompt}
                className="max-w-full max-h-[calc(100vh-350px)] object-contain rounded-lg"
              />
              <div className="mt-4 text-center max-w-xl">
                <p className="text-white text-sm truncate">{selectedImage.prompt}</p>
                <p className="text-muted text-xs mt-1">
                  {selectedImage.width}×{selectedImage.height} | {selectedImage.steps} steps | CFG {selectedImage.guidance_scale} | Seed {selectedImage.seed}
                </p>
                <div className="flex items-center justify-center gap-2 mt-2">
                  <button
                    onClick={() => handleUseImageSettings(selectedImage)}
                    className="px-3 py-1 text-xs text-muted border border-dark-border rounded hover:text-white"
                  >
                    Use Settings
                  </button>
                  <button
                    onClick={() => handleDeleteImage(selectedImage.id)}
                    className="px-3 py-1 text-xs text-danger border border-dark-border rounded hover:bg-danger/20"
                  >
                    Delete
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center text-muted">
              <ImageIcon className="w-24 h-24 mb-4 opacity-20" />
              <p className="text-sm">Generated image will appear here</p>
            </div>
          )}
        </div>

        {/* Generate Button */}
        <div className="p-4 border-t border-dark-border flex flex-col items-center justify-center gap-3">
          {isGenerating && (
            <div className="w-full max-w-md">
              <div className="flex items-center justify-between text-sm mb-1">
                <span className="text-muted">Generating...</span>
                <span className="text-white">{inferenceState?.generation_progress ?? 0}%</span>
              </div>
              <div className="h-2 bg-dark-border rounded-full overflow-hidden">
                <div
                  className="h-full bg-cyan-600 transition-all duration-300"
                  style={{ width: `${inferenceState?.generation_progress ?? 0}%` }}
                />
              </div>
            </div>
          )}
          <div className="flex items-center gap-3">
            {isGenerating ? (
              <button
                onClick={handleCancel}
                className="flex items-center gap-2 px-8 py-3 bg-red-600 hover:bg-red-500 text-white rounded-lg font-medium transition-colors"
              >
                <Square className="w-5 h-5" />
                Cancel
              </button>
            ) : (
              <button
                onClick={handleGenerate}
                disabled={!prompt.trim()}
                title={!isModelLoaded ? 'No model loaded - will use placeholder mode' : 'Generate image'}
                className={`flex items-center gap-2 px-8 py-3 ${isModelLoaded ? 'bg-cyan-600 hover:bg-cyan-500' : 'bg-amber-600 hover:bg-amber-500'} disabled:bg-dark-border disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors`}
              >
                <Play className="w-5 h-5" />
                {isModelLoaded ? 'Generate' : 'Generate (Demo)'}
              </button>
            )}
            {selectedImage && (
              <button className="p-3 bg-dark-surface hover:bg-dark-hover border border-dark-border rounded-lg text-muted hover:text-white">
                <Save className="w-5 h-5" />
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Right Panel - Gallery */}
      <div className="w-64 bg-dark-surface border-l border-dark-border flex flex-col">
        <div className="p-3 border-b border-dark-border bg-dark-surface flex items-center justify-between">
          <h2 className="text-sm font-medium text-white">Gallery</h2>
          {gallery.length > 0 && (
            <button onClick={handleClearGallery} className="text-xs text-muted hover:text-danger">
              Clear
            </button>
          )}
        </div>

        <div className="flex-1 overflow-y-auto p-2">
          {gallery.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-muted">
              <ImageIcon className="w-12 h-12 mb-2 opacity-20" />
              <p className="text-xs text-center">Generated images<br />will appear here</p>
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-2">
              {gallery.map((img) => (
                <button
                  key={img.id}
                  onClick={() => setSelectedImage(img)}
                  className={`aspect-square rounded border-2 overflow-hidden hover:border-cyan-500 transition-colors ${selectedImage?.id === img.id ? 'border-cyan-500' : 'border-dark-border'
                    }`}
                >
                  <img src={inferenceApi.getImage(img.id)} alt="" className="w-full h-full object-cover" />
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
