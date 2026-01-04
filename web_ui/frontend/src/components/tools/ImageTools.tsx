/**
 * Image Tools Component - Face Restore, Face Swap, Upscale, Background Removal
 */
import { useState, useRef } from 'react';
import axios from 'axios';

interface ToolResult {
    success: boolean;
    image: string;
    message?: string;
}

const API = axios.create({ baseURL: '/api/tools' });

// Utility to convert file to base64
const fileToBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result as string);
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
};

// Reusable image upload component
function ImageUpload({
    label,
    image,
    onUpload,
    onClear
}: {
    label: string;
    image: string | null;
    onUpload: (b64: string) => void;
    onClear: () => void;
}) {
    const inputRef = useRef<HTMLInputElement>(null);

    const handleFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            const b64 = await fileToBase64(file);
            onUpload(b64);
        }
    };

    return (
        <div className="space-y-2">
            <label className="text-sm text-muted block">{label}</label>
            <div
                className="border-2 border-dashed border-dark-border rounded-lg p-4 cursor-pointer hover:border-primary transition-colors min-h-[160px] flex items-center justify-center"
                onClick={() => inputRef.current?.click()}
            >
                {image ? (
                    <div className="relative w-full">
                        <img src={image} alt={label} className="max-h-40 mx-auto rounded" />
                        <button
                            onClick={(e) => { e.stopPropagation(); onClear(); }}
                            className="absolute top-0 right-0 p-1.5 bg-error/80 hover:bg-error rounded text-white text-xs"
                        >
                            ✕
                        </button>
                    </div>
                ) : (
                    <div className="text-center text-muted">
                        <div className="text-3xl mb-2">📷</div>
                        <p className="text-sm">Click to upload</p>
                    </div>
                )}
            </div>
            <input ref={inputRef} type="file" accept="image/*" onChange={handleFile} className="hidden" />
        </div>
    );
}

// Result display component
function ResultDisplay({
    result,
    loading,
    transparent
}: {
    result: string | null;
    loading: boolean;
    transparent?: boolean;
}) {
    const bgStyle = transparent
        ? { background: 'repeating-conic-gradient(#222 0% 25%, #333 0% 50%) 0 0/20px 20px' }
        : {};

    return (
        <div className="space-y-2">
            <label className="text-sm text-muted block">Result</label>
            <div
                className="border border-dark-border rounded-lg p-4 min-h-[160px] flex items-center justify-center bg-dark-bg"
                style={bgStyle}
            >
                {loading ? (
                    <div className="text-primary animate-pulse">Processing...</div>
                ) : result ? (
                    <img src={result} alt="Result" className="max-h-40 rounded" />
                ) : (
                    <p className="text-muted text-sm">Result will appear here</p>
                )}
            </div>
        </div>
    );
}

export function ImageTools() {
    const [activeTool, setActiveTool] = useState<'face-restore' | 'face-swap' | 'upscale' | 'background'>('face-restore');

    // Face Restore state
    const [restoreImage, setRestoreImage] = useState<string | null>(null);
    const [restoreResult, setRestoreResult] = useState<string | null>(null);
    const [restoreLoading, setRestoreLoading] = useState(false);
    const [restoreUpscale, setRestoreUpscale] = useState(2);

    // Face Swap state
    const [swapSource, setSwapSource] = useState<string | null>(null);
    const [swapTarget, setSwapTarget] = useState<string | null>(null);
    const [swapResult, setSwapResult] = useState<string | null>(null);
    const [swapLoading, setSwapLoading] = useState(false);

    // Upscale state
    const [upscaleImage, setUpscaleImage] = useState<string | null>(null);
    const [upscaleResult, setUpscaleResult] = useState<string | null>(null);
    const [upscaleLoading, setUpscaleLoading] = useState(false);
    const [upscaleScale, setUpscaleScale] = useState(4);
    const [upscaleMethod, setUpscaleMethod] = useState('esrgan');

    // Background state
    const [bgImage, setBgImage] = useState<string | null>(null);
    const [bgResult, setBgResult] = useState<string | null>(null);
    const [bgLoading, setBgLoading] = useState(false);
    const [bgModel, setBgModel] = useState('u2net');

    // API calls
    const doFaceRestore = async () => {
        if (!restoreImage) return;
        setRestoreLoading(true);
        try {
            const res = await API.post<ToolResult>('/face/restore', {
                image: restoreImage.split(',')[1],
                upscale: restoreUpscale,
                method: 'gfpgan'
            });
            if (res.data.success) {
                setRestoreResult(`data:image/png;base64,${res.data.image}`);
            }
        } catch (e) { console.error(e); }
        setRestoreLoading(false);
    };

    const doFaceSwap = async () => {
        if (!swapSource || !swapTarget) return;
        setSwapLoading(true);
        try {
            const res = await API.post<ToolResult>('/face/swap', {
                source_image: swapSource.split(',')[1],
                target_image: swapTarget.split(',')[1],
                swap_all: false
            });
            if (res.data.success) {
                setSwapResult(`data:image/png;base64,${res.data.image}`);
            }
        } catch (e) { console.error(e); }
        setSwapLoading(false);
    };

    const doUpscale = async () => {
        if (!upscaleImage) return;
        setUpscaleLoading(true);
        try {
            const res = await API.post<ToolResult>('/upscale', {
                image: upscaleImage.split(',')[1],
                scale: upscaleScale,
                method: upscaleMethod
            });
            if (res.data.success) {
                setUpscaleResult(`data:image/png;base64,${res.data.image}`);
            }
        } catch (e) { console.error(e); }
        setUpscaleLoading(false);
    };

    const doBackgroundRemove = async () => {
        if (!bgImage) return;
        setBgLoading(true);
        try {
            const res = await API.post<ToolResult>('/background/remove', {
                image: bgImage.split(',')[1],
                model: bgModel,
                alpha_matting: false
            });
            if (res.data.success) {
                setBgResult(`data:image/png;base64,${res.data.image}`);
            }
        } catch (e) { console.error(e); }
        setBgLoading(false);
    };

    return (
        <div className="space-y-6">
            {/* Tool tabs */}
            <div className="flex gap-2 border-b border-dark-border pb-2">
                {[
                    { id: 'face-restore', label: '👤 Face Restore' },
                    { id: 'face-swap', label: '🔄 Face Swap' },
                    { id: 'upscale', label: '⬆️ Upscale' },
                    { id: 'background', label: '✂️ Background' },
                ].map(tool => (
                    <button
                        key={tool.id}
                        onClick={() => setActiveTool(tool.id as typeof activeTool)}
                        className={`px-4 py-2 text-sm rounded-t transition-colors ${activeTool === tool.id
                                ? 'bg-primary text-white'
                                : 'text-muted hover:text-white hover:bg-dark-hover'
                            }`}
                    >
                        {tool.label}
                    </button>
                ))}
            </div>

            {/* Face Restore */}
            {activeTool === 'face-restore' && (
                <div className="bg-dark-surface rounded-lg border border-dark-border p-4 space-y-4">
                    <h3 className="text-white font-medium">Face Restoration (GFPGAN)</h3>
                    <p className="text-sm text-muted">Enhance and restore faces in images using AI.</p>
                    <div className="grid grid-cols-2 gap-4">
                        <ImageUpload label="Input Image" image={restoreImage} onUpload={setRestoreImage} onClear={() => setRestoreImage(null)} />
                        <ResultDisplay result={restoreResult} loading={restoreLoading} />
                    </div>
                    <div className="flex items-center gap-4">
                        <label className="text-sm text-muted">Upscale:</label>
                        <select value={restoreUpscale} onChange={e => setRestoreUpscale(Number(e.target.value))} className="input">
                            <option value={1}>1x</option>
                            <option value={2}>2x</option>
                            <option value={4}>4x</option>
                        </select>
                        <button
                            onClick={doFaceRestore}
                            disabled={!restoreImage || restoreLoading}
                            className="px-4 py-2 bg-primary hover:bg-primary-light disabled:bg-dark-border text-white rounded transition-colors"
                        >
                            Restore Face
                        </button>
                    </div>
                </div>
            )}

            {/* Face Swap */}
            {activeTool === 'face-swap' && (
                <div className="bg-dark-surface rounded-lg border border-dark-border p-4 space-y-4">
                    <h3 className="text-white font-medium">Face Swap (InsightFace)</h3>
                    <p className="text-sm text-muted">Swap a face from one image onto another.</p>
                    <div className="grid grid-cols-3 gap-4">
                        <ImageUpload label="Source Face" image={swapSource} onUpload={setSwapSource} onClear={() => setSwapSource(null)} />
                        <ImageUpload label="Target Image" image={swapTarget} onUpload={setSwapTarget} onClear={() => setSwapTarget(null)} />
                        <ResultDisplay result={swapResult} loading={swapLoading} />
                    </div>
                    <button
                        onClick={doFaceSwap}
                        disabled={!swapSource || !swapTarget || swapLoading}
                        className="px-4 py-2 bg-primary hover:bg-primary-light disabled:bg-dark-border text-white rounded transition-colors"
                    >
                        Swap Face
                    </button>
                </div>
            )}

            {/* Upscale */}
            {activeTool === 'upscale' && (
                <div className="bg-dark-surface rounded-lg border border-dark-border p-4 space-y-4">
                    <h3 className="text-white font-medium">Image Upscaling (Real-ESRGAN)</h3>
                    <p className="text-sm text-muted">Upscale images using AI super-resolution.</p>
                    <div className="grid grid-cols-2 gap-4">
                        <ImageUpload label="Input Image" image={upscaleImage} onUpload={setUpscaleImage} onClear={() => setUpscaleImage(null)} />
                        <ResultDisplay result={upscaleResult} loading={upscaleLoading} />
                    </div>
                    <div className="flex items-center gap-4">
                        <label className="text-sm text-muted">Scale:</label>
                        <select value={upscaleScale} onChange={e => setUpscaleScale(Number(e.target.value))} className="input">
                            <option value={2}>2x</option>
                            <option value={4}>4x</option>
                        </select>
                        <label className="text-sm text-muted">Method:</label>
                        <select value={upscaleMethod} onChange={e => setUpscaleMethod(e.target.value)} className="input">
                            <option value="esrgan">Real-ESRGAN</option>
                            <option value="lanczos">Lanczos (CPU)</option>
                        </select>
                        <button
                            onClick={doUpscale}
                            disabled={!upscaleImage || upscaleLoading}
                            className="px-4 py-2 bg-primary hover:bg-primary-light disabled:bg-dark-border text-white rounded transition-colors"
                        >
                            Upscale
                        </button>
                    </div>
                </div>
            )}

            {/* Background Removal */}
            {activeTool === 'background' && (
                <div className="bg-dark-surface rounded-lg border border-dark-border p-4 space-y-4">
                    <h3 className="text-white font-medium">Background Removal (rembg)</h3>
                    <p className="text-sm text-muted">Remove backgrounds from images.</p>
                    <div className="grid grid-cols-2 gap-4">
                        <ImageUpload label="Input Image" image={bgImage} onUpload={setBgImage} onClear={() => setBgImage(null)} />
                        <ResultDisplay result={bgResult} loading={bgLoading} transparent />
                    </div>
                    <div className="flex items-center gap-4">
                        <label className="text-sm text-muted">Model:</label>
                        <select value={bgModel} onChange={e => setBgModel(e.target.value)} className="input">
                            <option value="u2net">U2Net (General)</option>
                            <option value="u2netp">U2Net-P (Fast)</option>
                            <option value="u2net_human_seg">Human Segmentation</option>
                            <option value="isnet-general-use">ISNet General</option>
                            <option value="isnet-anime">ISNet Anime</option>
                        </select>
                        <button
                            onClick={doBackgroundRemove}
                            disabled={!bgImage || bgLoading}
                            className="px-4 py-2 bg-primary hover:bg-primary-light disabled:bg-dark-border text-white rounded transition-colors"
                        >
                            Remove Background
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
}
