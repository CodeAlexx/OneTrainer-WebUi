import { useState, useEffect } from 'react';
import { Plus, Trash2, GripVertical, ChevronDown, ChevronUp, Layers } from 'lucide-react';
import { useConfigStore } from '../../stores/configStore';

interface AdapterConfig {
  id: string;
  name: string;
  enabled: boolean;
  peft_type: string;
  rank: number;
  alpha: number;
  dropout: number;
  target_layers: string;
  use_tucker: boolean;
  use_scalar: boolean;
  train_norm: boolean;
  // LyCORIS-specific
  factor: number;
  decompose_both: boolean;
  full_matrix: boolean;
  block_size: number;
  // DoRA
  weight_decompose: boolean;
}

const DEFAULT_ADAPTER: Omit<AdapterConfig, 'id'> = {
  name: 'Adapter 1',
  enabled: true,
  peft_type: 'LORA',
  rank: 16,
  alpha: 16,
  dropout: 0.0,
  target_layers: '',
  use_tucker: false,
  use_scalar: false,
  train_norm: false,
  factor: -1,
  decompose_both: false,
  full_matrix: false,
  block_size: 4,
  weight_decompose: false,
};

const PEFT_TYPES = [
  { value: 'LORA', label: 'LoRA', description: 'Low-Rank Adaptation' },
  { value: 'LOHA', label: 'LoHa', description: 'Low-Rank Hadamard Product' },
  { value: 'LOKR', label: 'LoKr', description: 'Low-Rank Kronecker Product (LyCORIS)' },
  { value: 'LOCON', label: 'LoCon', description: 'LoRA for Convolution (LyCORIS)' },
  { value: 'IA3', label: 'IA3', description: 'Infused Adapter (LyCORIS)' },
  { value: 'DYLORA', label: 'DyLoRA', description: 'Dynamic LoRA (LyCORIS)' },
  { value: 'GLORA', label: 'GLoRA', description: 'Generalized LoRA (LyCORIS)' },
  { value: 'OFT_2', label: 'OFT', description: 'Orthogonal Fine-Tuning' },
  { value: 'BOFT', label: 'BOFT', description: 'Butterfly OFT (LyCORIS)' },
];

const isLycorisType = (type: string) => ['LOKR', 'LOCON', 'IA3', 'DYLORA', 'GLORA', 'BOFT'].includes(type);

const LAYER_PRESETS = [
  { value: '', label: 'Full (All Layers)', description: 'Train all trainable layers' },
  { value: '.*attn.*', label: 'Attention Only', description: 'Only attention layers (q, k, v, out)' },
  { value: '.*\\.(q|k|v|out)_proj.*', label: 'QKV + Out Proj', description: 'Query, Key, Value and output projections' },
  { value: '.*\\.mlp\\..*', label: 'MLP/FFN Only', description: 'Only feed-forward/MLP layers' },
  { value: '.*\\.(to_q|to_k|to_v|to_out).*', label: 'Diffusers Attention', description: 'Diffusers-style attention layers' },
  { value: '.*transformer_blocks.*', label: 'Transformer Blocks', description: 'All transformer block layers' },
  { value: '^(?=.*attention)(?!.*refiner).*', label: 'Attention (no refiner)', description: 'Attention layers excluding refiner' },
  { value: '.*single_transformer_blocks.*', label: 'Single Blocks (Flux)', description: 'Flux single transformer blocks' },
  { value: '.*double_transformer_blocks.*', label: 'Double Blocks (Flux)', description: 'Flux double transformer blocks' },
  { value: 'custom', label: 'Custom Pattern...', description: 'Enter custom regex pattern' },
];

const createAdapter = (index: number): AdapterConfig => ({
  ...DEFAULT_ADAPTER,
  id: `adapter-${Date.now()}-${Math.random()}`,
  name: `Adapter ${index + 1}`,
});

export function LoRAView() {
  const { config: storeConfig, updateConfig } = useConfigStore();
  const [adapters, setAdapters] = useState<AdapterConfig[]>([createAdapter(0)]);
  const [expandedAdapter, setExpandedAdapter] = useState<string | null>(null);
  const [multiAdapterMode, setMultiAdapterMode] = useState(false);

  // Sync from store when it changes
  useEffect(() => {
    if (storeConfig && Object.keys(storeConfig).length > 0) {
      const c = storeConfig as any;

      // Check if we have multi-adapter config
      if (c.adapters && Array.isArray(c.adapters)) {
        setAdapters(c.adapters);
        setMultiAdapterMode(c.adapters.length > 1);
      } else {
        // Legacy single adapter config
        const singleAdapter: AdapterConfig = {
          id: 'adapter-primary',
          name: 'Primary Adapter',
          enabled: true,
          peft_type: c.peft_type || 'LORA',
          rank: c.lora_rank ?? 16,
          alpha: c.lora_alpha ?? 16,
          dropout: c.dropout_probability ?? c.lora_dropout ?? 0.0,
          target_layers: c.layer_filter || '',
          use_tucker: c.lycoris_use_tucker ?? false,
          use_scalar: c.lycoris_use_scalar ?? false,
          train_norm: c.lycoris_train_norm ?? false,
          factor: c.lycoris_factor ?? -1,
          decompose_both: c.lycoris_decompose_both ?? false,
          full_matrix: c.lycoris_full_matrix ?? false,
          block_size: c.lycoris_block_size ?? 4,
          weight_decompose: c.lora_decompose ?? false,
        };
        setAdapters([singleAdapter]);
        setExpandedAdapter(singleAdapter.id);
      }
    }
  }, [storeConfig]);

  // Sync to store
  const syncToStore = (newAdapters: AdapterConfig[]) => {
    setAdapters(newAdapters);

    if (newAdapters.length === 1 && !multiAdapterMode) {
      // Single adapter - use legacy format for compatibility
      const a = newAdapters[0];
      updateConfig({
        peft_type: a.peft_type,
        lora_rank: a.rank,
        lora_alpha: a.alpha,
        dropout_probability: a.dropout,
        layer_filter: a.target_layers,
        lycoris_use_tucker: a.use_tucker,
        lycoris_use_scalar: a.use_scalar,
        lycoris_train_norm: a.train_norm,
        lycoris_factor: a.factor,
        lycoris_decompose_both: a.decompose_both,
        lycoris_full_matrix: a.full_matrix,
        lycoris_block_size: a.block_size,
        lora_decompose: a.weight_decompose,
      });
    } else {
      // Multi-adapter mode
      updateConfig({ adapters: newAdapters } as any);
    }
  };

  const addAdapter = () => {
    const newAdapter = createAdapter(adapters.length);
    const newAdapters = [...adapters, newAdapter];
    syncToStore(newAdapters);
    setExpandedAdapter(newAdapter.id);
    setMultiAdapterMode(true);
  };

  const removeAdapter = (id: string) => {
    if (adapters.length <= 1) return;
    const newAdapters = adapters.filter(a => a.id !== id);
    syncToStore(newAdapters);
    if (expandedAdapter === id) {
      setExpandedAdapter(newAdapters[0]?.id || null);
    }
    if (newAdapters.length === 1) {
      setMultiAdapterMode(false);
    }
  };

  const updateAdapter = (id: string, updates: Partial<AdapterConfig>) => {
    const newAdapters = adapters.map(a => a.id === id ? { ...a, ...updates } : a);
    syncToStore(newAdapters);
  };

  const moveAdapter = (id: string, direction: 'up' | 'down') => {
    const idx = adapters.findIndex(a => a.id === id);
    if (idx === -1) return;
    const newIdx = direction === 'up' ? idx - 1 : idx + 1;
    if (newIdx < 0 || newIdx >= adapters.length) return;

    const newAdapters = [...adapters];
    [newAdapters[idx], newAdapters[newIdx]] = [newAdapters[newIdx], newAdapters[idx]];
    syncToStore(newAdapters);
  };

  const toggleExpand = (id: string) => {
    setExpandedAdapter(expandedAdapter === id ? null : id);
  };

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="h-14 flex items-center justify-between px-6 border-b border-dark-border bg-dark-surface">
        <div className="flex items-center gap-3">
          <Layers className="w-5 h-5 text-primary" />
          <h1 className="text-lg font-medium text-white">LoRA / PEFT Adapters</h1>
          {multiAdapterMode && (
            <span className="text-xs bg-primary/20 text-primary px-2 py-0.5 rounded">
              Multi-Adapter Mode
            </span>
          )}
        </div>
        <button
          onClick={addAdapter}
          className="flex items-center gap-2 bg-primary hover:bg-primary-hover text-white px-3 py-1.5 rounded text-sm"
        >
          <Plus className="w-4 h-4" />
          Add Adapter
        </button>
      </div>

      {/* Multi-adapter info */}
      {multiAdapterMode && (
        <div className="px-6 py-3 bg-blue-500/10 border-b border-blue-500/20">
          <p className="text-sm text-blue-400">
            <strong>Multi-Adapter Stacking:</strong> Adapters are applied in order from top to bottom.
            Each adapter can target different layers with different algorithms.
          </p>
        </div>
      )}

      {/* Content */}
      <div className="flex-1 overflow-auto p-6">
        <div className="max-w-3xl space-y-4">
          {adapters.map((adapter, index) => (
            <div
              key={adapter.id}
              className={`bg-dark-surface rounded-lg border ${adapter.enabled ? 'border-dark-border' : 'border-dark-border/50 opacity-60'}`}
            >
              {/* Adapter Header */}
              <div
                className="px-4 py-3 flex items-center justify-between cursor-pointer hover:bg-dark-hover/50"
                onClick={() => toggleExpand(adapter.id)}
              >
                <div className="flex items-center gap-3">
                  <GripVertical className="w-4 h-4 text-muted" />
                  <span className="text-xs text-muted w-6">#{index + 1}</span>
                  <input
                    type="checkbox"
                    checked={adapter.enabled}
                    onChange={(e) => {
                      e.stopPropagation();
                      updateAdapter(adapter.id, { enabled: e.target.checked });
                    }}
                    className="rounded"
                  />
                  <span className="font-medium text-white">{adapter.name}</span>
                  <span className="text-xs bg-dark-bg px-2 py-0.5 rounded text-muted">
                    {PEFT_TYPES.find(t => t.value === adapter.peft_type)?.label || adapter.peft_type}
                  </span>
                  <span className="text-xs text-muted">
                    r={adapter.rank} α={adapter.alpha}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  {multiAdapterMode && (
                    <>
                      <button
                        onClick={(e) => { e.stopPropagation(); moveAdapter(adapter.id, 'up'); }}
                        disabled={index === 0}
                        className="p-1 text-muted hover:text-white disabled:opacity-30"
                      >
                        <ChevronUp className="w-4 h-4" />
                      </button>
                      <button
                        onClick={(e) => { e.stopPropagation(); moveAdapter(adapter.id, 'down'); }}
                        disabled={index === adapters.length - 1}
                        className="p-1 text-muted hover:text-white disabled:opacity-30"
                      >
                        <ChevronDown className="w-4 h-4" />
                      </button>
                    </>
                  )}
                  {adapters.length > 1 && (
                    <button
                      onClick={(e) => { e.stopPropagation(); removeAdapter(adapter.id); }}
                      className="p-1 text-red-400 hover:text-red-300"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  )}
                  {expandedAdapter === adapter.id ? (
                    <ChevronUp className="w-4 h-4 text-muted" />
                  ) : (
                    <ChevronDown className="w-4 h-4 text-muted" />
                  )}
                </div>
              </div>

              {/* Adapter Details (expanded) */}
              {expandedAdapter === adapter.id && (
                <div className="px-4 pb-4 border-t border-dark-border pt-4 space-y-4">
                  {/* Name */}
                  <div>
                    <label className="text-xs text-muted block mb-1">Adapter Name</label>
                    <input
                      type="text"
                      value={adapter.name}
                      onChange={(e) => updateAdapter(adapter.id, { name: e.target.value })}
                      className="input w-full text-sm"
                    />
                  </div>

                  {/* PEFT Type */}
                  <div>
                    <label className="text-xs text-muted block mb-1">PEFT Type</label>
                    <select
                      value={adapter.peft_type}
                      onChange={(e) => updateAdapter(adapter.id, { peft_type: e.target.value })}
                      className="input w-full text-sm"
                    >
                      {PEFT_TYPES.map((type) => (
                        <option key={type.value} value={type.value}>
                          {type.label} - {type.description}
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* Rank & Alpha */}
                  <div className="grid grid-cols-3 gap-4">
                    <div>
                      <label className="text-xs text-muted block mb-1">Rank</label>
                      <input
                        type="text"
                        value={adapter.rank}
                        onChange={(e) => updateAdapter(adapter.id, { rank: parseInt(e.target.value) || 1 })}
                        className="input w-full text-sm"
                        min="1"
                        max="512"
                      />
                    </div>
                    <div>
                      <label className="text-xs text-muted block mb-1">Alpha</label>
                      <input
                        type="text"
                        value={adapter.alpha}
                        onChange={(e) => updateAdapter(adapter.id, { alpha: parseInt(e.target.value) || 1 })}
                        className="input w-full text-sm"
                        min="1"
                        max="512"
                      />
                    </div>
                    <div>
                      <label className="text-xs text-muted block mb-1">Dropout</label>
                      <input
                        type="text"
                        value={adapter.dropout}
                        onChange={(e) => updateAdapter(adapter.id, { dropout: parseFloat(e.target.value) || 0 })}
                        className="input w-full text-sm"
                        min="0"
                        max="1"
                        step="0.01"
                      />
                    </div>
                  </div>

                  {/* Target Layers */}
                  <div>
                    <label className="text-xs text-muted block mb-1">Target Layers</label>
                    <select
                      value={LAYER_PRESETS.find(p => p.value === adapter.target_layers)?.value ?? 'custom'}
                      onChange={(e) => {
                        if (e.target.value !== 'custom') {
                          updateAdapter(adapter.id, { target_layers: e.target.value });
                        }
                      }}
                      className="input w-full text-sm mb-2"
                    >
                      {LAYER_PRESETS.map((preset) => (
                        <option key={preset.value} value={preset.value}>
                          {preset.label} {preset.value && preset.value !== 'custom' ? `(${preset.value})` : ''}
                        </option>
                      ))}
                    </select>
                    <label className="text-xs text-muted block mb-1">Custom Pattern (regex)</label>
                    <input
                      type="text"
                      value={adapter.target_layers}
                      onChange={(e) => updateAdapter(adapter.id, { target_layers: e.target.value })}
                      className="input w-full text-sm font-mono"
                      placeholder="Empty = all layers"
                    />
                  </div>

                  {/* DoRA (for LoRA type) */}
                  {adapter.peft_type === 'LORA' && (
                    <div className="flex items-center gap-2 py-1">
                      <input
                        type="checkbox"
                        checked={adapter.weight_decompose}
                        onChange={(e) => updateAdapter(adapter.id, { weight_decompose: e.target.checked })}
                        className="rounded"
                      />
                      <span className="text-sm text-white">Enable DoRA (Weight Decomposition)</span>
                    </div>
                  )}

                  {/* LyCORIS-specific options */}
                  {isLycorisType(adapter.peft_type) && (
                    <div className="border-t border-dark-border pt-4 mt-4">
                      <h3 className="text-xs font-medium text-muted uppercase mb-3">LyCORIS Options</h3>
                      <div className="grid grid-cols-2 gap-4">
                        <div className="flex items-center gap-2">
                          <input
                            type="checkbox"
                            checked={adapter.use_tucker}
                            onChange={(e) => updateAdapter(adapter.id, { use_tucker: e.target.checked })}
                            className="rounded"
                          />
                          <span className="text-sm text-white">Use Tucker</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <input
                            type="checkbox"
                            checked={adapter.use_scalar}
                            onChange={(e) => updateAdapter(adapter.id, { use_scalar: e.target.checked })}
                            className="rounded"
                          />
                          <span className="text-sm text-white">Use Scalar</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <input
                            type="checkbox"
                            checked={adapter.train_norm}
                            onChange={(e) => updateAdapter(adapter.id, { train_norm: e.target.checked })}
                            className="rounded"
                          />
                          <span className="text-sm text-white">Train Norm</span>
                        </div>
                        {adapter.peft_type === 'LOKR' && (
                          <>
                            <div className="flex items-center gap-2">
                              <input
                                type="checkbox"
                                checked={adapter.decompose_both}
                                onChange={(e) => updateAdapter(adapter.id, { decompose_both: e.target.checked })}
                                className="rounded"
                              />
                              <span className="text-sm text-white">Decompose Both</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <input
                                type="checkbox"
                                checked={adapter.full_matrix}
                                onChange={(e) => updateAdapter(adapter.id, { full_matrix: e.target.checked })}
                                className="rounded"
                              />
                              <span className="text-sm text-white">Full Matrix</span>
                            </div>
                            <div>
                              <label className="text-xs text-muted block mb-1">Factor</label>
                              <input
                                type="text"
                                value={adapter.factor}
                                onChange={(e) => updateAdapter(adapter.id, { factor: parseInt(e.target.value) })}
                                className="input w-full text-sm"
                              />
                            </div>
                          </>
                        )}
                        {adapter.peft_type === 'BOFT' && (
                          <div>
                            <label className="text-xs text-muted block mb-1">Block Size</label>
                            <input
                              type="text"
                              value={adapter.block_size}
                              onChange={(e) => updateAdapter(adapter.id, { block_size: parseInt(e.target.value) || 4 })}
                              className="input w-full text-sm"
                              min="2"
                            />
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* OFT-specific options */}
                  {adapter.peft_type === 'OFT_2' && (
                    <div className="border-t border-dark-border pt-4 mt-4">
                      <h3 className="text-xs font-medium text-muted uppercase mb-3">OFT Options</h3>
                      <div>
                        <label className="text-xs text-muted block mb-1">Block Size</label>
                        <input
                          type="text"
                          value={adapter.block_size}
                          onChange={(e) => updateAdapter(adapter.id, { block_size: parseInt(e.target.value) || 4 })}
                          className="input w-full text-sm"
                          min="2"
                        />
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}

          {/* Stacking Summary */}
          {multiAdapterMode && adapters.length > 1 && (
            <div className="bg-dark-surface rounded-lg border border-dark-border p-4">
              <h3 className="text-sm font-medium text-muted uppercase mb-3">Adapter Stack Order</h3>
              <div className="space-y-2">
                {adapters.filter(a => a.enabled).map((adapter, idx) => (
                  <div key={adapter.id} className="flex items-center gap-2 text-sm">
                    <span className="text-muted w-8">#{idx + 1}</span>
                    <span className="text-white">{adapter.name}</span>
                    <span className="text-muted">→</span>
                    <span className="text-primary">{adapter.peft_type}</span>
                    {adapter.target_layers && (
                      <span className="text-xs text-muted font-mono">({adapter.target_layers})</span>
                    )}
                  </div>
                ))}
              </div>
              <p className="text-xs text-muted mt-3">
                Adapters are stacked in this order. Layer targeting allows different adapters to modify different parts of the model.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
