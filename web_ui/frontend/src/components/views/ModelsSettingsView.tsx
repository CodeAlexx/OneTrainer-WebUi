import { useState, useEffect } from 'react';
import { Plus, Trash2, Save, FolderOpen } from 'lucide-react';
import axios from 'axios';

interface ModelEntry {
    id: string;
    name: string;
    path: string;
    type: 'safetensors' | 'diffusers';
    category: 'Image' | 'Video';
    modelType: string;
}

const DEFAULT_MODELS: ModelEntry[] = [
    // Image models
    { id: '1', name: 'FLUX Dev', path: '/home/alex/SwarmUI/Models/diffusion_models/flux1-dev.safetensors', type: 'safetensors', category: 'Image', modelType: 'FLUX_DEV_1' },
    { id: '2', name: 'FLUX Schnell', path: '/home/alex/SwarmUI/Models/diffusion_models/uncensoredFemalesFLUX4step_nf4Schnell4step.safetensors', type: 'safetensors', category: 'Image', modelType: 'FLUX_SCHNELL' },
    { id: '3', name: 'SDXL', path: '/home/alex/SwarmUI/Models/diffusion_models/lustifySDXLNSFW_ggwpV7.safetensors', type: 'safetensors', category: 'Image', modelType: 'STABLE_DIFFUSION_XL_10_BASE' },
    { id: '4', name: 'SD 3.5', path: '/home/alex/SwarmUI/Models/diffusion_models/sd3.5_large.safetensors', type: 'safetensors', category: 'Image', modelType: 'STABLE_DIFFUSION_35' },
    { id: '5', name: 'Z-Image', path: '/home/alex/SwarmUI/Models/diffusion_models/z_image_de_turbo_v1_bf16.safetensors', type: 'safetensors', category: 'Image', modelType: 'Z_IMAGE' },
    { id: '6', name: 'Z-Image Turbo', path: '/home/alex/SwarmUI/Models/diffusion_models/z_image_turbo_bf16.safetensors', type: 'safetensors', category: 'Image', modelType: 'Z_IMAGE_TURBO' },
    { id: '7', name: 'Qwen Image', path: '/home/alex/SwarmUI/Models/diffusion_models/qwen_image_fp8_e4m3fn.safetensors', type: 'safetensors', category: 'Image', modelType: 'QWEN_IMAGE' },
    { id: '8', name: 'Qwen Edit', path: 'alibaba-pai/OmniGen2-Edit', type: 'diffusers', category: 'Image', modelType: 'QWEN_IMAGE_EDIT' },
    { id: '9', name: 'Lumina 2', path: 'Alpha-VLLM/Lumina-Image-2.0', type: 'diffusers', category: 'Image', modelType: 'lumina_2' },
    { id: '10', name: 'OmniGen 2', path: 'BAAI/OmniGen2', type: 'diffusers', category: 'Image', modelType: 'omnigen_2' },
    { id: '11', name: 'Kandinsky 5 T2I', path: 'kandinskylab/Kandinsky-5.0-T2I-Lite', type: 'diffusers', category: 'Image', modelType: 'KANDINSKY_5' },
    { id: '12', name: 'Chroma HD', path: 'lodestones/Chroma1-HD', type: 'diffusers', category: 'Image', modelType: 'CHROMA_1' },
    // Video models
    { id: '13', name: 'Kandinsky 5 T2V Lite', path: '/home/alex/SwarmUI/Models/diffusion_models/kandinsky5lite_t2v_sft_5s.safetensors', type: 'safetensors', category: 'Video', modelType: 'kandinsky_5_video' },
    { id: '14', name: 'Kandinsky 5 T2V Pro', path: '/home/alex/OneTrainer/models/kandinsky-5-video-pro/model/kandinsky5pro_t2v_sft_5s.safetensors', type: 'safetensors', category: 'Video', modelType: 'kandinsky_5_video_pro' },
    { id: '15', name: 'Wan 2.2 T2V (High)', path: '/home/alex/SwarmUI/Models/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors', type: 'safetensors', category: 'Video', modelType: 'wan_t2v_high' },
    { id: '16', name: 'Wan 2.2 T2V (Low)', path: '/home/alex/SwarmUI/Models/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors', type: 'safetensors', category: 'Video', modelType: 'wan_t2v_low' },
];

const MODEL_TYPES = [
    'FLUX_DEV_1', 'FLUX_SCHNELL', 'STABLE_DIFFUSION_XL_10_BASE', 'STABLE_DIFFUSION_35',
    'Z_IMAGE', 'Z_IMAGE_TURBO', 'QWEN_IMAGE', 'QWEN_IMAGE_EDIT',
    'lumina_2', 'omnigen_2', 'KANDINSKY_5', 'kandinsky_5_video', 'kandinsky_5_video_pro',
    'CHROMA_1', 'wan_t2v_high', 'wan_t2v_low', 'wan_i2v_high', 'wan_i2v_low', 'wan_vace'
];

export function ModelsSettingsView() {
    const [models, setModels] = useState<ModelEntry[]>(DEFAULT_MODELS);
    const [saving, setSaving] = useState(false);
    const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

    useEffect(() => {
        loadModels();
    }, []);

    const loadModels = async () => {
        try {
            const res = await axios.get('/api/settings/models');
            if (res.data.models?.length > 0) {
                setModels(res.data.models);
            }
        } catch {
            // Use defaults if not saved yet
        }
    };

    const saveModels = async () => {
        setSaving(true);
        try {
            await axios.post('/api/settings/models', { models });
            setMessage({ type: 'success', text: 'Models saved successfully!' });
            setTimeout(() => setMessage(null), 3000);
        } catch (e) {
            setMessage({ type: 'error', text: 'Failed to save models' });
        } finally {
            setSaving(false);
        }
    };

    const addModel = () => {
        const newId = String(Date.now());
        setModels([...models, {
            id: newId,
            name: 'New Model',
            path: '',
            type: 'safetensors',
            category: 'Image',
            modelType: 'FLUX_DEV_1'
        }]);
    };

    const removeModel = (id: string) => {
        setModels(models.filter(m => m.id !== id));
    };

    const updateModel = (id: string, field: keyof ModelEntry, value: string) => {
        setModels(models.map(m => m.id === id ? { ...m, [field]: value } : m));
    };

    const imageModels = models.filter(m => m.category === 'Image');
    const videoModels = models.filter(m => m.category === 'Video');

    return (
        <div className="p-6 max-w-6xl mx-auto">
            <div className="flex items-center justify-between mb-6">
                <h1 className="text-2xl font-bold text-white">Model Paths</h1>
                <div className="flex gap-2">
                    <button
                        onClick={addModel}
                        className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg"
                    >
                        <Plus className="w-4 h-4" /> Add Model
                    </button>
                    <button
                        onClick={saveModels}
                        disabled={saving}
                        className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-500 text-white rounded-lg disabled:opacity-50"
                    >
                        <Save className="w-4 h-4" /> {saving ? 'Saving...' : 'Save'}
                    </button>
                </div>
            </div>

            {message && (
                <div className={`mb-4 p-3 rounded-lg ${message.type === 'success' ? 'bg-green-900/50 text-green-300' : 'bg-red-900/50 text-red-300'}`}>
                    {message.text}
                </div>
            )}

            {/* Image Models */}
            <div className="mb-8">
                <h2 className="text-lg font-semibold text-gray-300 mb-4 flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full bg-blue-500"></span>
                    Image Models ({imageModels.length})
                </h2>
                <div className="space-y-2">
                    {imageModels.map(model => (
                        <ModelRow key={model.id} model={model} onUpdate={updateModel} onRemove={removeModel} />
                    ))}
                </div>
            </div>

            {/* Video Models */}
            <div>
                <h2 className="text-lg font-semibold text-gray-300 mb-4 flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full bg-purple-500"></span>
                    Video Models ({videoModels.length})
                </h2>
                <div className="space-y-2">
                    {videoModels.map(model => (
                        <ModelRow key={model.id} model={model} onUpdate={updateModel} onRemove={removeModel} />
                    ))}
                </div>
            </div>
        </div>
    );
}

function ModelRow({ model, onUpdate, onRemove }: {
    model: ModelEntry;
    onUpdate: (id: string, field: keyof ModelEntry, value: string) => void;
    onRemove: (id: string) => void;
}) {
    return (
        <div className="flex items-center gap-3 p-3 bg-dark-surface rounded-lg border border-dark-border">
            {/* Name */}
            <input
                type="text"
                value={model.name}
                onChange={e => onUpdate(model.id, 'name', e.target.value)}
                className="w-40 px-3 py-2 bg-dark-bg border border-dark-border rounded text-sm text-white focus:border-blue-500 focus:outline-none"
                placeholder="Name"
            />

            {/* Path */}
            <div className="flex-1 flex items-center gap-2">
                <input
                    type="text"
                    value={model.path}
                    onChange={e => onUpdate(model.id, 'path', e.target.value)}
                    className="flex-1 px-3 py-2 bg-dark-bg border border-dark-border rounded text-sm text-white font-mono focus:border-blue-500 focus:outline-none"
                    placeholder="/path/to/model.safetensors or huggingface/repo-id"
                />
                <button className="p-2 text-gray-400 hover:text-white hover:bg-dark-hover rounded" title="Browse">
                    <FolderOpen className="w-4 h-4" />
                </button>
            </div>

            {/* Type */}
            <select
                value={model.type}
                onChange={e => onUpdate(model.id, 'type', e.target.value)}
                className="w-28 px-3 py-2 bg-dark-bg border border-dark-border rounded text-sm text-white focus:border-blue-500 focus:outline-none"
            >
                <option value="safetensors">Safetensors</option>
                <option value="diffusers">Diffusers</option>
            </select>

            {/* Category */}
            <select
                value={model.category}
                onChange={e => onUpdate(model.id, 'category', e.target.value)}
                className="w-24 px-3 py-2 bg-dark-bg border border-dark-border rounded text-sm text-white focus:border-blue-500 focus:outline-none"
            >
                <option value="Image">Image</option>
                <option value="Video">Video</option>
            </select>

            {/* Model Type */}
            <select
                value={model.modelType}
                onChange={e => onUpdate(model.id, 'modelType', e.target.value)}
                className="w-48 px-3 py-2 bg-dark-bg border border-dark-border rounded text-sm text-white focus:border-blue-500 focus:outline-none"
            >
                {MODEL_TYPES.map(t => <option key={t} value={t}>{t}</option>)}
            </select>

            {/* Delete */}
            <button
                onClick={() => onRemove(model.id)}
                className="p-2 text-red-400 hover:text-red-300 hover:bg-red-900/30 rounded"
                title="Remove"
            >
                <Trash2 className="w-4 h-4" />
            </button>
        </div>
    );
}
