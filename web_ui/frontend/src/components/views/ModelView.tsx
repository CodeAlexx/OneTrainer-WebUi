import { useState, useEffect } from 'react';
import { useConfigStore } from '../../stores/configStore';

// All model types from OneTrainer
const MODEL_TYPES = [
  'STABLE_DIFFUSION_15', 'STABLE_DIFFUSION_15_INPAINTING',
  'STABLE_DIFFUSION_20', 'STABLE_DIFFUSION_20_BASE', 'STABLE_DIFFUSION_20_INPAINTING', 'STABLE_DIFFUSION_20_DEPTH',
  'STABLE_DIFFUSION_21', 'STABLE_DIFFUSION_21_BASE',
  'STABLE_DIFFUSION_3', 'STABLE_DIFFUSION_35',
  'STABLE_DIFFUSION_XL_10_BASE', 'STABLE_DIFFUSION_XL_10_BASE_INPAINTING',
  'WUERSTCHEN_2', 'STABLE_CASCADE_1',
  'PIXART_ALPHA', 'PIXART_SIGMA',
  'FLUX_DEV_1', 'FLUX_FILL_DEV_1',
  'SANA', 'HUNYUAN_VIDEO', 'HI_DREAM_FULL', 'CHROMA_1', 'QWEN', 'Z_IMAGE'
];

const TRAINING_METHODS = ['FINE_TUNE', 'LORA', 'EMBEDDING', 'FINE_TUNE_VAE'];
const OUTPUT_FORMATS = ['SAFETENSORS', 'DIFFUSERS'];

// Base data types (human-readable label -> enum value)
const BASE_DATA_TYPES = [
  { label: 'float32', value: 'FLOAT_32' },
  { label: 'bfloat16', value: 'BFLOAT_16' },
  { label: 'float16', value: 'FLOAT_16' },
  { label: 'float8 (W8)', value: 'FLOAT_8' },
  { label: 'nfloat4', value: 'NFLOAT_4' },
];

// Transformer/UNet data types (includes A8 quantization types)
const TRANSFORMER_DATA_TYPES = [
  ...BASE_DATA_TYPES,
  { label: 'float W8A8', value: 'FLOAT_W8A8' },
  { label: 'int W8A8', value: 'INT_W8A8' },
];

// Transformer data types with GGUF support
const TRANSFORMER_DATA_TYPES_WITH_GGUF = [
  ...TRANSFORMER_DATA_TYPES,
  { label: 'GGUF', value: 'GGUF' },
  { label: 'GGUF A8 float', value: 'GGUF_A8_FLOAT' },
  { label: 'GGUF A8 int', value: 'GGUF_A8_INT' },
];

// Output data types
const OUTPUT_DATA_TYPES = [
  { label: 'float16', value: 'FLOAT_16' },
  { label: 'float32', value: 'FLOAT_32' },
  { label: 'bfloat16', value: 'BFLOAT_16' },
  { label: 'float8', value: 'FLOAT_8' },
  { label: 'nfloat4', value: 'NFLOAT_4' },
];

const CONFIG_INCLUDE = ['NONE', 'SETTINGS', 'ALL'];

interface ModelConfig {
  model_type: string;
  training_method: string;
  base_model_name: string;
  huggingface_token: string;
  compile: boolean;
  transformer_model_name: string;
  vae_model_name: string;
  transformer_weight_dtype: string;
  text_encoder_weight_dtype: string;
  text_encoder_2_weight_dtype: string;
  text_encoder_3_weight_dtype: string;
  vae_weight_dtype: string;
  quantization_layer_filter: string;
  svd_dtype: string;
  svd_rank: number;
  output_model_destination: string;
  output_dtype: string;
  output_model_format: string;
  include_train_config: string;
}

const DEFAULT_CONFIG: ModelConfig = {
  model_type: 'FLUX_DEV_1',
  training_method: 'LORA',
  base_model_name: '',
  huggingface_token: '',
  compile: false,  // Default off per original
  transformer_model_name: '',
  vae_model_name: '',
  transformer_weight_dtype: 'FLOAT_8',  // Match original presets
  text_encoder_weight_dtype: 'FLOAT_8',  // Match original presets
  text_encoder_2_weight_dtype: 'BFLOAT_16',
  text_encoder_3_weight_dtype: 'BFLOAT_16',
  vae_weight_dtype: 'FLOAT_32',  // Match original presets
  quantization_layer_filter: '',
  svd_dtype: 'NONE',
  svd_rank: 64,
  output_model_destination: '',
  output_dtype: 'BFLOAT_16',  // Match original presets
  output_model_format: 'SAFETENSORS',
  include_train_config: 'SETTINGS',
};

export function ModelView() {
  const { config: storeConfig, updateConfig } = useConfigStore();
  const [config, setConfig] = useState<ModelConfig>(DEFAULT_CONFIG);

  // Helper to get nested value
  const getNestedValue = (obj: any, path: string, defaultVal: any) => {
    const parts = path.split('.');
    let value = obj;
    for (const part of parts) {
      value = value?.[part];
      if (value === undefined) return defaultVal;
    }
    return value ?? defaultVal;
  };

  // Sync from store when it changes
  useEffect(() => {
    if (storeConfig && Object.keys(storeConfig).length > 0) {
      const c = storeConfig as any;
      setConfig({
        model_type: c.model_type || DEFAULT_CONFIG.model_type,
        training_method: c.training_method || DEFAULT_CONFIG.training_method,
        base_model_name: c.base_model_name || '',
        huggingface_token: c.huggingface_token || '',
        compile: c.compile ?? DEFAULT_CONFIG.compile,
        transformer_model_name: getNestedValue(c, 'transformer.model_name', ''),
        vae_model_name: getNestedValue(c, 'vae.model_name', ''),
        transformer_weight_dtype: getNestedValue(c, 'transformer.weight_dtype', DEFAULT_CONFIG.transformer_weight_dtype),
        text_encoder_weight_dtype: getNestedValue(c, 'text_encoder.weight_dtype', DEFAULT_CONFIG.text_encoder_weight_dtype),
        text_encoder_2_weight_dtype: getNestedValue(c, 'text_encoder_2.weight_dtype', DEFAULT_CONFIG.text_encoder_2_weight_dtype),
        text_encoder_3_weight_dtype: getNestedValue(c, 'text_encoder_3.weight_dtype', DEFAULT_CONFIG.text_encoder_3_weight_dtype),
        vae_weight_dtype: getNestedValue(c, 'vae.weight_dtype', DEFAULT_CONFIG.vae_weight_dtype),
        quantization_layer_filter: getNestedValue(c, 'quantization.layer_filter', ''),
        svd_dtype: getNestedValue(c, 'quantization.svd_dtype', DEFAULT_CONFIG.svd_dtype),
        svd_rank: getNestedValue(c, 'quantization.svd_rank', DEFAULT_CONFIG.svd_rank),
        output_model_destination: c.output_model_destination || '',
        output_dtype: c.output_dtype || DEFAULT_CONFIG.output_dtype,
        output_model_format: c.output_model_format || DEFAULT_CONFIG.output_model_format,
        include_train_config: c.include_train_config || DEFAULT_CONFIG.include_train_config,
      });
    }
  }, [storeConfig]);

  // Mapping of local field names to config paths
  const CONFIG_PATHS: Record<string, string> = {
    transformer_model_name: 'transformer.model_name',
    vae_model_name: 'vae.model_name',
    transformer_weight_dtype: 'transformer.weight_dtype',
    text_encoder_weight_dtype: 'text_encoder.weight_dtype',
    text_encoder_2_weight_dtype: 'text_encoder_2.weight_dtype',
    text_encoder_3_weight_dtype: 'text_encoder_3.weight_dtype',
    vae_weight_dtype: 'vae.weight_dtype',
    quantization_layer_filter: 'quantization.layer_filter',
    svd_dtype: 'quantization.svd_dtype',
    svd_rank: 'quantization.svd_rank',
  };

  const update = (field: keyof ModelConfig, value: any) => {
    setConfig({ ...config, [field]: value });
    // Use nested path if defined, otherwise use flat field name
    const configPath = CONFIG_PATHS[field as string];
    if (configPath) {
      // Create nested update
      const parts = configPath.split('.');
      let updateObj: any = value;
      for (let i = parts.length - 1; i >= 0; i--) {
        updateObj = { [parts[i]]: updateObj };
      }
      updateConfig(updateObj);
    } else {
      updateConfig({ [field]: value });
    }
  };


  const hasTransformer = ['STABLE_DIFFUSION_3', 'STABLE_DIFFUSION_35', 'PIXART_ALPHA', 'PIXART_SIGMA',
    'FLUX_DEV_1', 'FLUX_FILL_DEV_1', 'SANA', 'HUNYUAN_VIDEO', 'HI_DREAM_FULL', 'CHROMA_1', 'QWEN', 'Z_IMAGE'].includes(config.model_type);
  const hasUnet = ['STABLE_DIFFUSION_15', 'STABLE_DIFFUSION_15_INPAINTING', 'STABLE_DIFFUSION_20',
    'STABLE_DIFFUSION_20_BASE', 'STABLE_DIFFUSION_20_INPAINTING', 'STABLE_DIFFUSION_20_DEPTH',
    'STABLE_DIFFUSION_21', 'STABLE_DIFFUSION_21_BASE', 'STABLE_DIFFUSION_XL_10_BASE',
    'STABLE_DIFFUSION_XL_10_BASE_INPAINTING'].includes(config.model_type);
  const hasMultipleTextEncoders = ['STABLE_DIFFUSION_3', 'STABLE_DIFFUSION_35', 'STABLE_DIFFUSION_XL_10_BASE',
    'STABLE_DIFFUSION_XL_10_BASE_INPAINTING', 'FLUX_DEV_1', 'FLUX_FILL_DEV_1', 'HUNYUAN_VIDEO', 'HI_DREAM_FULL'].includes(config.model_type);
  const hasThreeTextEncoders = ['STABLE_DIFFUSION_3', 'STABLE_DIFFUSION_35', 'HI_DREAM_FULL'].includes(config.model_type);

  return (
    <div className="h-full flex flex-col">
      <div className="h-14 flex items-center px-6 border-b border-dark-border bg-dark-surface">
        <h1 className="text-lg font-medium text-white">Model</h1>
      </div>

      <div className="flex-1 overflow-auto p-4">
        <div className="grid grid-cols-2 gap-4">
          {/* Left Column */}
          <div className="space-y-4">
            <div className="bg-dark-surface rounded-lg border border-dark-border p-4 space-y-3">
              <h2 className="text-xs font-medium text-muted uppercase tracking-wider mb-3">Base Model</h2>

              <div>
                <label className="text-xs text-muted block mb-1">Model Type</label>
                <select value={config.model_type} onChange={(e) => update('model_type', e.target.value)} className="input w-full text-sm">
                  {MODEL_TYPES.map(m => <option key={m} value={m}>{m}</option>)}
                </select>
              </div>

              <div>
                <label className="text-xs text-muted block mb-1">Training Method</label>
                <select value={config.training_method} onChange={(e) => update('training_method', e.target.value)} className="input w-full text-sm">
                  {TRAINING_METHODS.map(m => <option key={m} value={m}>{m}</option>)}
                </select>
              </div>

              <div>
                <label className="text-xs text-muted block mb-1">Hugging Face Token</label>
                <input type="password" value={config.huggingface_token} onChange={(e) => update('huggingface_token', e.target.value)}
                  className="input w-full text-sm" placeholder="Optional: for protected repos" />
              </div>

              <div>
                <label className="text-xs text-muted block mb-1">Base Model</label>
                <input type="text" value={config.base_model_name} onChange={(e) => update('base_model_name', e.target.value)}
                  className="input w-full text-sm" placeholder="Path or HuggingFace repo" />
              </div>

              <div className="flex items-center justify-between">
                <span className="text-sm text-white">Compile Transformer Blocks</span>
                <button onClick={() => update('compile', !config.compile)}
                  className={`relative w-9 h-5 rounded-full transition-colors ${config.compile ? 'bg-green-600' : 'bg-gray-600'}`}>
                  <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform ${config.compile ? 'translate-x-4' : 'translate-x-0'}`} />
                </button>
              </div>
            </div>

            <div className="bg-dark-surface rounded-lg border border-dark-border p-4 space-y-3">
              <h2 className="text-xs font-medium text-muted uppercase tracking-wider mb-3">Model Overrides</h2>

              {hasTransformer && (
                <div>
                  <label className="text-xs text-muted block mb-1">Override Transformer / GGUF</label>
                  <input type="text" value={config.transformer_model_name} onChange={(e) => update('transformer_model_name', e.target.value)}
                    className="input w-full text-sm" placeholder="Optional" />
                </div>
              )}

              <div>
                <label className="text-xs text-muted block mb-1">VAE Override</label>
                <input type="text" value={config.vae_model_name} onChange={(e) => update('vae_model_name', e.target.value)}
                  className="input w-full text-sm" placeholder="Optional" />
              </div>
            </div>

            <div className="bg-dark-surface rounded-lg border border-dark-border p-4 space-y-3">
              <h2 className="text-xs font-medium text-muted uppercase tracking-wider mb-3">Quantization</h2>

              <div>
                <label className="text-xs text-muted block mb-1">Quantization Layer Filter</label>
                <input type="text" value={config.quantization_layer_filter} onChange={(e) => update('quantization_layer_filter', e.target.value)}
                  className="input w-full text-sm" placeholder="Comma-separated layers" />
              </div>

              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="text-xs text-muted block mb-1">SVDQuant</label>
                  <select value={config.svd_dtype} onChange={(e) => update('svd_dtype', e.target.value)} className="input w-full text-sm">
                    <option value="NONE">disabled</option>
                    <option value="FLOAT_32">float32</option>
                    <option value="BFLOAT_16">bfloat16</option>
                  </select>
                </div>
                <div>
                  <label className="text-xs text-muted block mb-1">SVDQuant Rank</label>
                  <input type="text" value={config.svd_rank} onChange={(e) => update('svd_rank', parseInt(e.target.value) || 64)}
                    className="input w-full text-sm" />
                </div>
              </div>
            </div>
          </div>

          {/* Right Column */}
          <div className="space-y-4">
            <div className="bg-dark-surface rounded-lg border border-dark-border p-4 space-y-3">
              <h2 className="text-xs font-medium text-muted uppercase tracking-wider mb-3">Component Data Types</h2>

              {hasTransformer && (
                <div>
                  <label className="text-xs text-muted block mb-1">Transformer Data Type</label>
                  <select value={config.transformer_weight_dtype} onChange={(e) => update('transformer_weight_dtype', e.target.value)} className="input w-full text-sm">
                    {TRANSFORMER_DATA_TYPES_WITH_GGUF.map(d => <option key={d.value} value={d.value}>{d.label}</option>)}
                  </select>
                </div>
              )}

              {hasUnet && (
                <div>
                  <label className="text-xs text-muted block mb-1">UNet Data Type</label>
                  <select value={config.transformer_weight_dtype} onChange={(e) => update('transformer_weight_dtype', e.target.value)} className="input w-full text-sm">
                    {TRANSFORMER_DATA_TYPES.map(d => <option key={d.value} value={d.value}>{d.label}</option>)}
                  </select>
                </div>
              )}

              <div>
                <label className="text-xs text-muted block mb-1">{hasMultipleTextEncoders ? 'Text Encoder 1' : 'Text Encoder'} Data Type</label>
                <select value={config.text_encoder_weight_dtype} onChange={(e) => update('text_encoder_weight_dtype', e.target.value)} className="input w-full text-sm">
                  {BASE_DATA_TYPES.map(d => <option key={d.value} value={d.value}>{d.label}</option>)}
                </select>
              </div>

              {hasMultipleTextEncoders && (
                <div>
                  <label className="text-xs text-muted block mb-1">Text Encoder 2 Data Type</label>
                  <select value={config.text_encoder_2_weight_dtype} onChange={(e) => update('text_encoder_2_weight_dtype', e.target.value)} className="input w-full text-sm">
                    {BASE_DATA_TYPES.map(d => <option key={d.value} value={d.value}>{d.label}</option>)}
                  </select>
                </div>
              )}

              {hasThreeTextEncoders && (
                <div>
                  <label className="text-xs text-muted block mb-1">Text Encoder 3 Data Type</label>
                  <select value={config.text_encoder_3_weight_dtype} onChange={(e) => update('text_encoder_3_weight_dtype', e.target.value)} className="input w-full text-sm">
                    {BASE_DATA_TYPES.map(d => <option key={d.value} value={d.value}>{d.label}</option>)}
                  </select>
                </div>
              )}

              <div>
                <label className="text-xs text-muted block mb-1">VAE Data Type</label>
                <select value={config.vae_weight_dtype} onChange={(e) => update('vae_weight_dtype', e.target.value)} className="input w-full text-sm">
                  {BASE_DATA_TYPES.map(d => <option key={d.value} value={d.value}>{d.label}</option>)}
                </select>
              </div>
            </div>

            <div className="bg-dark-surface rounded-lg border border-dark-border p-4 space-y-3">
              <h2 className="text-xs font-medium text-muted uppercase tracking-wider mb-3">Output</h2>

              <div>
                <label className="text-xs text-muted block mb-1">Model Output Destination</label>
                <input type="text" value={config.output_model_destination} onChange={(e) => update('output_model_destination', e.target.value)}
                  className="input w-full text-sm" placeholder="Output path" />
              </div>

              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="text-xs text-muted block mb-1">Output Data Type</label>
                  <select value={config.output_dtype} onChange={(e) => update('output_dtype', e.target.value)} className="input w-full text-sm">
                    {OUTPUT_DATA_TYPES.map(d => <option key={d.value} value={d.value}>{d.label}</option>)}
                  </select>
                </div>
                <div>
                  <label className="text-xs text-muted block mb-1">Output Format</label>
                  <select value={config.output_model_format} onChange={(e) => update('output_model_format', e.target.value)} className="input w-full text-sm">
                    {OUTPUT_FORMATS.map(f => <option key={f} value={f}>{f}</option>)}
                  </select>
                </div>
              </div>

              <div>
                <label className="text-xs text-muted block mb-1">Include Config</label>
                <select value={config.include_train_config} onChange={(e) => update('include_train_config', e.target.value)} className="input w-full text-sm">
                  {CONFIG_INCLUDE.map(c => <option key={c} value={c}>{c}</option>)}
                </select>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
