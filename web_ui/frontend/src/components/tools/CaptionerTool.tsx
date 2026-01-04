import { useState, useEffect, useRef } from 'react';
import { captionApi, CaptionState, CaptionBatchStatus } from '../../lib/api';

export function CaptionerTool() {
    // State
    const [modelState, setModelState] = useState<CaptionState>({
        loaded: false,
        model_id: null,
        device: 'N/A',
        vram_used: 'N/A',
        dtype: 'N/A'
    });

    const [settings, setSettings] = useState({
        model_id: "Qwen/Qwen2-VL-7B-Instruct",
        custom_model_id: "",
        quantization: "None",
        attn_impl: "eager",
        folder_path: "",
        base_prompt: "Give one detailed paragraph (max 250 words) describing everything clearly visible in the image—subjects, objects, environment, style, lighting, and mood. Do not use openings like 'This is' or 'The image shows'; start directly with the main subject. Avoid guessing anything not clearly visible.",
        max_tokens: 256,
        skip_existing: false,
        resolution_mode: "auto",
        summary_mode: false,
        one_sentence_mode: false,
        retain_preview: true
    });

    const [batchStatus, setBatchStatus] = useState<CaptionBatchStatus>({
        active: false,
        stats: { processed: 0, skipped: 0, failed: 0 },
        current_file: null,
        last_caption: null,
        progress: 0
    });

    const [isLoading, setIsLoading] = useState(false);

    const [elapsedTime, setElapsedTime] = useState(0);
    const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

    // Computed
    const finalPrompt = `${settings.base_prompt}${settings.summary_mode ? " Give a short summary." : ""}${settings.one_sentence_mode ? " Describe in one sentence." : ""}`;

    // Polling for status
    useEffect(() => {
        fetchState();
        const interval = setInterval(() => {
            if (batchStatus.active) {
                fetchBatchStatus();
            }
        }, 500);
        return () => clearInterval(interval);
    }, [batchStatus.active]);

    // Timer Logic
    useEffect(() => {
        if (batchStatus.active) {
            if (!timerRef.current) {
                const startTime = Date.now() - (elapsedTime * 1000);
                timerRef.current = setInterval(() => {
                    setElapsedTime(Math.floor((Date.now() - startTime) / 1000));
                }, 1000);
            }
        } else {
            if (timerRef.current) {
                clearInterval(timerRef.current);
                timerRef.current = null;
            }
        }
    }, [batchStatus.active]);

    const fetchState = async () => {
        try {
            const res = await captionApi.getState();
            setModelState(res.data);
        } catch (err) {
            console.error("Failed to fetch state", err);
        }
    };

    const fetchBatchStatus = async () => {
        try {
            const res = await captionApi.getBatchStatus();
            setBatchStatus(res.data);
        } catch (err) {
            console.error(err);
        }
    };

    const handleLoadModel = async () => {
        setIsLoading(true);
        try {
            const id = settings.model_id === "Custom..." ? settings.custom_model_id : settings.model_id;
            const res = await captionApi.loadModel(id, settings.quantization, settings.attn_impl);
            setModelState(res.data);
        } catch (err) {
            alert(`Error loading model: ${err}`);
        } finally {
            setIsLoading(false);
        }
    };

    const handleStartBatch = async () => {
        if (!settings.folder_path) return alert("Folder path required");

        setElapsedTime(0);
        try {
            await captionApi.startBatch({
                folder_path: settings.folder_path,
                prompt: finalPrompt,
                skip_existing: settings.skip_existing,
                max_tokens: settings.max_tokens,
                resolution_mode: settings.resolution_mode
            });
            setBatchStatus(prev => ({ ...prev, active: true }));
        } catch (err) {
            alert(`Error starting batch: ${err}`);
        }
    };

    const handleStopBatch = async () => {
        await captionApi.stopBatch();
    };

    const formatTime = (seconds: number) => {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    };

    return (
        <div className="h-full overflow-y-auto p-6 max-w-6xl mx-auto space-y-6 text-gray-200">
            {/* Model Information Accordion/Panel */}
            <div className="bg-dark-surface border border-dark-border rounded-lg p-4 shadow-sm">
                <div className="flex justify-between items-center mb-4">
                    <h3 className="font-medium text-sm text-gray-300 flex items-center gap-2">
                        <span>Model Information</span>
                        <span className={`text-[10px] px-1.5 py-0.5 rounded ${modelState.loaded ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}`}>
                            {modelState.loaded ? 'LOADED' : 'UNLOADED'}
                        </span>
                    </h3>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                        <label className="text-xs text-gray-400 block mb-1">Model ID</label>
                        <div className="flex gap-2">
                            <select
                                value={settings.model_id}
                                onChange={e => setSettings({ ...settings, model_id: e.target.value })}
                                className="input flex-1 text-sm bg-dark-bg border-dark-border text-gray-200"
                            >
                                <option value="Qwen/Qwen2.5-VL-7B-Instruct">Qwen2.5-VL-7B-Instruct</option>
                                <option value="Qwen/Qwen2-VL-7B-Instruct">Qwen2-VL-7B-Instruct</option>
                            </select>
                            <button
                                onClick={handleLoadModel}
                                disabled={modelState.loaded || isLoading}
                                className="btn-primary text-xs px-3 min-w-[60px]"
                            >
                                {isLoading ? '⏳ Loading...' : 'Load'}
                            </button>
                        </div>
                    </div>
                    {modelState.loaded && (
                        <div className="text-xs flex flex-col justify-end text-gray-400">
                            <div>Using: {modelState.vram_used} VRAM</div>
                            <div>Quant: {settings.quantization} | Attn: {settings.attn_impl}</div>
                        </div>
                    )}
                </div>
            </div>

            {/* Main Inputs */}
            <div className="bg-dark-surface border border-dark-border rounded-lg p-6 space-y-6 shadow-sm">
                {/* Folder Path */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                        <label className="text-xs font-bold text-gray-400 uppercase tracking-wide mb-1 block">Folder Path</label>
                        <input
                            type="text"
                            value={settings.folder_path}
                            onChange={e => setSettings({ ...settings, folder_path: e.target.value })}
                            className="input w-full bg-dark-bg border-dark-border text-gray-200 placeholder-gray-600"
                            placeholder="/path/to/images"
                        />
                    </div>
                    <div>
                        <label className="text-xs font-bold text-gray-400 uppercase tracking-wide mb-1 block">Custom Prompt</label>
                        <input
                            type="text"
                            value={settings.base_prompt}
                            onChange={e => setSettings({ ...settings, base_prompt: e.target.value })}
                            className="input w-full bg-dark-bg border-dark-border text-gray-200"
                        />
                    </div>
                </div>

                {/* Skip Existing */}
                <div className="flex items-center gap-2">
                    <input
                        type="checkbox"
                        checked={settings.skip_existing}
                        onChange={e => setSettings({ ...settings, skip_existing: e.target.checked })}
                        className="rounded border-gray-600 bg-transparent text-blue-500 focus:ring-offset-0"
                    />
                    <label className="text-sm text-gray-300">Skip already captioned media (.txt exists)</label>
                </div>

                {/* Prompt Controls */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-start">
                    <div className="col-span-2 flex gap-8">
                        <label className="flex items-center gap-2 text-sm text-gray-300 cursor-pointer">
                            <input
                                type="checkbox"
                                checked={settings.summary_mode}
                                onChange={e => setSettings({ ...settings, summary_mode: e.target.checked })}
                                className="rounded border-gray-600 bg-transparent text-blue-500"
                            />
                            Summary Mode
                        </label>
                        <label className="flex items-center gap-2 text-sm text-gray-300 cursor-pointer">
                            <input
                                type="checkbox"
                                checked={settings.one_sentence_mode}
                                onChange={e => setSettings({ ...settings, one_sentence_mode: e.target.checked })}
                                className="rounded border-gray-600 bg-transparent text-blue-500"
                            />
                            One-Sentence Mode
                        </label>
                    </div>
                </div>

                {/* Final Prompt Preview */}
                <div>
                    <label className="text-xs text-gray-500 mb-1 block">Final Prompt Preview</label>
                    <div className="p-3 bg-dark-bg rounded text-gray-300 text-sm border border-dark-border">
                        {finalPrompt}
                    </div>
                </div>

                {/* Tokens & Resolution */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    <div>
                        <div className="flex justify-between items-center mb-1">
                            <label className="text-xs font-bold text-gray-400 uppercase tracking-wide">Max Tokens</label>
                            <span className="text-xs bg-dark-bg text-gray-300 px-1 rounded border border-dark-border">{settings.max_tokens}</span>
                        </div>
                        <input
                            type="range"
                            min="32" max="512" step="16"
                            value={settings.max_tokens}
                            onChange={e => setSettings({ ...settings, max_tokens: parseInt(e.target.value) })}
                            className="w-full h-1.5 bg-gray-600 rounded-lg appearance-none cursor-pointer accent-blue-600"
                        />
                        <div className="flex justify-between text-[10px] text-gray-500 mt-1">
                            <span>32</span>
                            <span>512</span>
                        </div>
                    </div>
                    <div>
                        <label className="text-xs font-bold text-gray-400 uppercase tracking-wide block mb-1">
                            Image Resolution <span className="font-normal text-gray-500 text-[10px] lowercase ml-1">choose the resolution mode</span>
                        </label>
                        <div className="relative">
                            <select
                                value={settings.resolution_mode}
                                onChange={e => setSettings({ ...settings, resolution_mode: e.target.value })}
                                className="input w-full bg-dark-bg border-dark-border text-gray-200 text-sm appearance-none"
                            >
                                <option value="auto">auto</option>
                                <option value="auto_high">auto_high</option>
                                <option value="high">high</option>
                                <option value="fast">fast</option>
                            </select>
                            {/* Arrow icon would go here usually */}
                        </div>
                    </div>
                </div>
            </div>

            {/* Action Bar */}
            <div className="flex justify-between gap-4">
                <button
                    onClick={() => setSettings({ ...settings, base_prompt: "Give one detailed paragraph (max 250 words) describing everything clearly visible in the image—subjects, objects, environment, style, lighting, and mood. Do not use openings like 'This is' or 'The image shows'; start directly with the main subject. Avoid guessing anything not clearly visible." })}
                    className="px-6 py-2 bg-dark-surface border border-dark-border hover:bg-dark-hover text-gray-300 font-medium rounded text-sm flex items-center gap-2 transition-colors"
                >
                    Reset to Default Prompt
                </button>
                <div className="flex gap-4 flex-1 justify-end">
                    <button
                        onClick={handleStartBatch}
                        disabled={batchStatus.active}
                        className="px-8 py-2 bg-primary hover:bg-primary-light disabled:opacity-50 text-white font-medium rounded text-sm flex items-center gap-2 min-w-[150px] justify-center shadow-lg transition-all"
                    >
                        🚀 Start Processing
                    </button>
                    <button
                        onClick={handleStopBatch}
                        disabled={!batchStatus.active}
                        className="px-8 py-2 bg-dark-surface border border-dark-border hover:bg-red-900/30 hover:border-red-500/50 text-gray-300 hover:text-red-400 font-medium rounded text-sm flex items-center gap-2 min-w-[150px] justify-center transition-all"
                    >
                        ⛔ Abort
                    </button>
                </div>
            </div>

            {/* Status Panel */}
            <div className="bg-dark-surface border border-dark-border rounded-lg p-4 shadow-sm space-y-4">
                <div className="text-sm font-medium text-gray-300">
                    Status
                </div>
                <div className="p-3 bg-dark-bg rounded text-sm text-gray-300 flex items-center gap-2 border border-dark-border">
                    {batchStatus.active ? (
                        <>Processing {batchStatus.stats.processed + 1}: {batchStatus.current_file ? batchStatus.current_file.split('/').pop() : 'Initializing...'}</>
                    ) : (
                        `Ready`
                    )}
                </div>

                <div>
                    <div className="text-xs text-gray-500 mb-1">Progress</div>
                    <div className="w-full bg-dark-bg rounded-full h-1.5 overflow-hidden border border-dark-border">
                        <div
                            className="bg-blue-500 h-full transition-all duration-300"
                            style={{ width: `${(batchStatus.progress || 0) * 100}%` }}
                        ></div>
                    </div>
                </div>

                <div>
                    <div className="text-xs text-gray-500 mb-1 flex items-center gap-1">
                        Time Taken (s)
                    </div>
                    <div className="p-2 bg-dark-bg rounded text-gray-300 font-mono text-sm inline-block min-w-[80px] border border-dark-border">
                        {formatTime(elapsedTime)}
                    </div>
                </div>
            </div>

            {/* Results Area (Split) */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 h-[400px]">
                {/* Current Image */}
                <div className="bg-black rounded-lg overflow-hidden border border-dark-border flex flex-col shadow-sm">
                    <div className="p-2 bg-white/5 text-xs text-gray-400 flex justify-between border-b border-white/5">
                        <span>Current Image</span>
                        <button className="text-gray-400 hover:text-white transition-colors">⛶</button>
                    </div>
                    <div className="flex-1 relative flex items-center justify-center bg-gray-900/50">
                        {batchStatus.current_file ? (
                            <img
                                src={`http://localhost:8000/api/caption/preview?path=${encodeURIComponent(settings.folder_path + '/' + batchStatus.current_file)}`}
                                className="max-w-full max-h-full object-contain"
                                key={batchStatus.current_file} // Force reload on change
                            />
                        ) : (
                            <div className="text-gray-600 text-sm">Waiting for media...</div>
                        )}
                    </div>
                    {batchStatus.current_file && (
                        <div className="p-1 bg-black text-[10px] text-gray-500 truncate px-2 border-t border-white/10">
                            File: {batchStatus.current_file.split('/').pop()}
                        </div>
                    )}
                </div>

                {/* Generated Caption */}
                <div className="bg-dark-surface rounded-lg border border-dark-border flex flex-col overflow-hidden shadow-sm">
                    <div className="p-2 border-b border-dark-border text-xs text-gray-400 font-medium bg-white/5">Generated Caption</div>
                    <div className="flex-1 p-4 bg-dark-bg overflow-y-auto text-sm text-gray-300 leading-relaxed font-sans">
                        {batchStatus.last_caption || "Caption will appear here..."}
                    </div>
                </div>
            </div>

            <div className="flex items-center gap-2">
                <input
                    type="checkbox"
                    checked={settings.retain_preview}
                    onChange={e => setSettings({ ...settings, retain_preview: e.target.checked })}
                    className="rounded border-gray-600 bg-transparent text-blue-500 focus:ring-offset-0"
                />
                <label className="text-sm text-gray-400">Retain preview on skip</label>
            </div>
        </div>
    );
}
