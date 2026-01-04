import { useState, useMemo } from 'react';
import { X, Search, Star, StarOff, Database, FolderOpen, Clock, Trash2 } from 'lucide-react';

interface PresetInfo {
    name: string;
    path: string;
    last_modified: string;
    id?: number;
    isDbPreset?: boolean;
}

interface PresetCardSelectorProps {
    presets: PresetInfo[];
    currentPreset: string | null;
    onSelect: (preset: PresetInfo) => void;
    onDelete: (preset: PresetInfo) => void;
    onClose: () => void;
}

// Extract model type from preset name
function getModelType(name: string): string {
    const lower = name.toLowerCase();
    // Check specific variants before general ones
    if (lower.includes('qwen') && (lower.includes('edit') || lower.includes('qedit'))) return 'Qwen-Edit';
    if (lower.includes('qwen')) return 'Qwen';
    if (lower.includes('kandinsky') || lower.includes('k5')) return 'Kandinsky';
    if (lower.includes('flux')) return 'Flux';
    if (lower.includes('sdxl') || lower.includes('xl')) return 'SDXL';
    if (lower.includes('sd 3') || lower.includes('sd3')) return 'SD3';
    if (lower.includes('sd 1') || lower.includes('sd 2') || lower.includes('sd1') || lower.includes('sd2')) return 'SD';
    if (lower.includes('chroma')) return 'Chroma';
    if (lower.includes('z-image') || lower.includes('zimage')) return 'Z-Image';
    if (lower.includes('pixart')) return 'PixArt';
    if (lower.includes('hunyuan')) return 'Hunyuan';
    if (lower.includes('hidream')) return 'HiDream';
    if (lower.includes('sana')) return 'Sana';
    if (lower.includes('cascade') || lower.includes('wuerstchen')) return 'Cascade';
    if (lower.includes('wan')) return 'Wan';
    return 'Other';
}

// Extract method type from preset name
function getMethodType(name: string): string {
    const lower = name.toLowerCase();
    if (lower.includes('lora') || lower.includes('lokr') || lower.includes('loha')) return 'LoRA';
    if (lower.includes('finetune') || lower.includes('fine_tune')) return 'Finetune';
    if (lower.includes('embedding')) return 'Embedding';
    return 'Other';
}

// Extract VRAM requirement from preset name
function getVramTier(name: string): string | null {
    const match = name.match(/(\d+)\s*GB/i);
    return match ? `${match[1]}GB` : null;
}

// Get color for model type badge
function getModelColor(modelType: string): string {
    const colors: Record<string, string> = {
        'Qwen': 'bg-purple-600',
        'Qwen-Edit': 'bg-violet-600',
        'Kandinsky': 'bg-rose-600',
        'Flux': 'bg-blue-600',
        'SDXL': 'bg-green-600',
        'SD3': 'bg-teal-600',
        'SD': 'bg-gray-600',
        'Chroma': 'bg-pink-600',
        'Z-Image': 'bg-orange-600',
        'PixArt': 'bg-cyan-600',
        'Hunyuan': 'bg-red-600',
        'HiDream': 'bg-indigo-600',
        'Sana': 'bg-yellow-600',
        'Cascade': 'bg-amber-600',
        'Wan': 'bg-emerald-600',
        'Other': 'bg-gray-500',
    };
    return colors[modelType] || 'bg-gray-500';
}

const ALL_FILTERS = ['All', 'Kandinsky', 'Qwen', 'Qwen-Edit', 'Flux', 'SDXL', 'SD3', 'SD', 'Chroma', 'Z-Image', 'LoRA', 'Finetune'];

export function PresetCardSelector({ presets, currentPreset, onSelect, onDelete, onClose }: PresetCardSelectorProps) {
    const [searchQuery, setSearchQuery] = useState('');
    const [activeFilter, setActiveFilter] = useState('All');
    const [favorites, setFavorites] = useState<Set<string>>(() => {
        const saved = localStorage.getItem('preset-favorites');
        return saved ? new Set(JSON.parse(saved)) : new Set();
    });

    // Sort and filter presets
    const filteredPresets = useMemo(() => {
        let result = [...presets];

        // Apply search filter
        if (searchQuery) {
            const query = searchQuery.toLowerCase();
            result = result.filter(p => p.name.toLowerCase().includes(query));
        }

        // Apply category filter
        if (activeFilter !== 'All') {
            if (['LoRA', 'Finetune', 'Embedding'].includes(activeFilter)) {
                result = result.filter(p => getMethodType(p.name) === activeFilter);
            } else {
                result = result.filter(p => getModelType(p.name) === activeFilter);
            }
        }

        // Sort: favorites first, then alphabetically
        result.sort((a, b) => {
            const aFav = favorites.has(a.name);
            const bFav = favorites.has(b.name);
            if (aFav && !bFav) return -1;
            if (!aFav && bFav) return 1;
            return a.name.localeCompare(b.name);
        });

        return result;
    }, [presets, searchQuery, activeFilter, favorites]);

    const toggleFavorite = (name: string, e: React.MouseEvent) => {
        e.stopPropagation();
        const newFavorites = new Set(favorites);
        if (newFavorites.has(name)) {
            newFavorites.delete(name);
        } else {
            newFavorites.add(name);
        }
        setFavorites(newFavorites);
        localStorage.setItem('preset-favorites', JSON.stringify([...newFavorites]));
    };

    // Count presets by model type for filter badges
    const filterCounts = useMemo(() => {
        const counts: Record<string, number> = { All: presets.length };
        presets.forEach(p => {
            const model = getModelType(p.name);
            const method = getMethodType(p.name);
            counts[model] = (counts[model] || 0) + 1;
            counts[method] = (counts[method] || 0) + 1;
        });
        return counts;
    }, [presets]);

    return (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-8" onClick={onClose}>
            <div
                className="bg-dark-surface border border-dark-border rounded-xl shadow-2xl w-full max-w-5xl max-h-[85vh] flex flex-col"
                onClick={e => e.stopPropagation()}
            >
                {/* Header */}
                <div className="flex items-center justify-between p-4 border-b border-dark-border">
                    <h2 className="text-lg font-semibold text-white">Select Preset</h2>
                    <button onClick={onClose} className="p-1 hover:bg-dark-hover rounded">
                        <X className="w-5 h-5 text-muted" />
                    </button>
                </div>

                {/* Search & Filters */}
                <div className="p-4 space-y-3 border-b border-dark-border">
                    {/* Search */}
                    <div className="relative">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted" />
                        <input
                            type="text"
                            placeholder="Search presets..."
                            value={searchQuery}
                            onChange={e => setSearchQuery(e.target.value)}
                            className="w-full pl-10 pr-4 py-2 bg-dark-bg border border-dark-border rounded-lg text-white placeholder:text-muted focus:border-primary focus:outline-none"
                            autoFocus
                        />
                    </div>

                    {/* Filter Buttons */}
                    <div className="flex flex-wrap gap-2">
                        {ALL_FILTERS.filter(f => (filterCounts[f] || 0) > 0).map(filter => (
                            <button
                                key={filter}
                                onClick={() => setActiveFilter(filter)}
                                className={`px-3 py-1 text-sm rounded-full transition-colors ${activeFilter === filter
                                    ? 'bg-primary text-white'
                                    : 'bg-dark-bg text-muted hover:text-white hover:bg-dark-hover'
                                    }`}
                            >
                                {filter}
                                <span className="ml-1 opacity-60">({filterCounts[filter] || 0})</span>
                            </button>
                        ))}
                    </div>
                </div>

                {/* Preset Grid */}
                <div className="flex-1 overflow-auto p-4">
                    {filteredPresets.length === 0 ? (
                        <div className="flex flex-col items-center justify-center h-48 text-muted">
                            <FolderOpen className="w-12 h-12 mb-2 opacity-50" />
                            <p>No presets found</p>
                        </div>
                    ) : (
                        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                            {filteredPresets.map(preset => {
                                const modelType = getModelType(preset.name);
                                const methodType = getMethodType(preset.name);
                                const vram = getVramTier(preset.name);
                                const isFavorite = favorites.has(preset.name);
                                const isSelected = currentPreset === preset.name;

                                return (
                                    <button
                                        key={preset.id || preset.name}
                                        onClick={() => onSelect(preset)}
                                        className={`relative p-3 rounded-lg border text-left transition-all hover:scale-[1.02] ${isSelected
                                            ? 'border-primary bg-primary/10'
                                            : 'border-dark-border bg-dark-bg hover:border-primary/50'
                                            }`}
                                    >
                                        {/* Favorite Star - top right */}
                                        <button
                                            onClick={(e) => toggleFavorite(preset.name, e)}
                                            className="absolute top-2 right-2 p-1 hover:bg-dark-hover rounded"
                                        >
                                            {isFavorite ? (
                                                <Star className="w-4 h-4 text-yellow-400 fill-yellow-400" />
                                            ) : (
                                                <StarOff className="w-4 h-4 text-muted hover:text-yellow-400" />
                                            )}
                                        </button>

                                        {/* Preset Name */}
                                        <div className="font-medium text-white text-sm truncate pr-6 mb-2">
                                            {preset.name.replace(/^#/, '')}
                                        </div>

                                        {/* Badges */}
                                        <div className="flex flex-wrap gap-1 mb-2">
                                            <span className={`px-1.5 py-0.5 text-[10px] font-medium rounded ${getModelColor(modelType)} text-white`}>
                                                {modelType}
                                            </span>
                                            {methodType !== 'Other' && (
                                                <span className="px-1.5 py-0.5 text-[10px] font-medium rounded bg-gray-600 text-white">
                                                    {methodType}
                                                </span>
                                            )}
                                            {vram && (
                                                <span className="px-1.5 py-0.5 text-[10px] font-medium rounded bg-dark-border text-muted">
                                                    {vram}
                                                </span>
                                            )}
                                        </div>

                                        {/* Bottom row: Source indicator + Delete button */}
                                        <div className="flex items-center justify-between">
                                            <div className="flex items-center gap-1 text-[10px] text-muted">
                                                {preset.isDbPreset ? (
                                                    <>
                                                        <Database className="w-3 h-3" />
                                                        <span>Database</span>
                                                    </>
                                                ) : (
                                                    <>
                                                        <Clock className="w-3 h-3" />
                                                        <span>JSON</span>
                                                    </>
                                                )}
                                            </div>
                                            {/* Delete button */}
                                            <button
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    e.preventDefault();
                                                    if (window.confirm(`Delete preset "${preset.name}"?`)) {
                                                        onDelete(preset);
                                                    }
                                                }}
                                                className="p-1 hover:bg-red-600/30 rounded opacity-60 hover:opacity-100 transition-opacity"
                                                title="Delete preset"
                                            >
                                                <Trash2 className="w-3.5 h-3.5 text-red-400" />
                                            </button>
                                        </div>
                                    </button>
                                );
                            })}
                        </div>
                    )}
                </div>

                {/* Footer */}
                <div className="flex items-center justify-between p-4 border-t border-dark-border text-sm text-muted">
                    <span>{filteredPresets.length} presets</span>
                    <span>Click to load • Star to favorite</span>
                </div>
            </div>
        </div>
    );
}
