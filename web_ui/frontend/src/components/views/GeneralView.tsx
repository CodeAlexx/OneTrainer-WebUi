import { useState, useEffect } from 'react';
import { useConfigStore } from '../../stores/configStore';
import { Settings, Box, Database, Save, FolderPlus } from 'lucide-react';

// ============= CONSTANTS =============
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

const TABS = [
  { id: 'general', label: 'General', icon: Settings },
  { id: 'model', label: 'Model', icon: Box },
  { id: 'data', label: 'Data', icon: Database },
  { id: 'backup', label: 'Backup', icon: Save },
];

// ============= COMPONENT =============
export function GeneralView() {
  const { config: storeConfig, updateConfig } = useConfigStore();
  const [activeTab, setActiveTab] = useState('general');

  // ============= GENERAL STATE =============
  const [workspace_dir, setWorkspaceDir] = useState('');
  const [cache_dir, setCacheDir] = useState('');
  const [samples_dir, setSamplesDir] = useState('');
  const [tensorboard, setTensorboard] = useState(true);
  const [debug, setDebug] = useState(false);
  const [validation_steps, setValidationSteps] = useState(500);
  const [continue_last_backup, setContinueLastBackup] = useState(false);
  const [only_cache, setOnlyCache] = useState(false);
  const [debug_dir, setDebugDir] = useState('');
  const [tensorboard_expose, setTensorboardExpose] = useState(false);
  const [tensorboard_always_on, setTensorboardAlwaysOn] = useState(false);
  const [tensorboard_port, setTensorboardPort] = useState(6006);
  const [wandb, setWandb] = useState(false);
  const [wandb_project, setWandbProject] = useState('');
  const [wandb_entity, setWandbEntity] = useState('');
  const [wandb_run_name, setWandbRunName] = useState('');
  const [wandb_tags, setWandbTags] = useState('');
  const [wandb_base_url, setWandbBaseUrl] = useState('');
  const [validate_after_unit, setValidateAfterUnit] = useState('EPOCH');
  const [train_device, setTrainDevice] = useState('cuda');
  const [multi_gpu, setMultiGpu] = useState(false);
  const [device_indexes, setDeviceIndexes] = useState('');
  const [gradient_reduce_prevision, setGradientReducePrevision] = useState('FLOAT_32_STOCHASTIC');
  const [fused_gradient_reduce, setFusedGradientReduce] = useState(false);
  const [async_gradient_reduce, setAsyncGradientReduce] = useState(true);
  const [async_gradient_reduce_buffer, setAsyncGradientReduceBuffer] = useState(100);
  const [temp_device, setTempDevice] = useState('cpu');

  // ============= MODEL STATE =============
  const [model_type, setModelType] = useState('FLUX_DEV_1');
  const [training_method, setTrainingMethod] = useState('LORA');
  const [base_model_name, setBaseModelName] = useState('');
  const [huggingface_token, setHuggingfaceToken] = useState('');
  const [compile, setCompile] = useState(false);  // Default off per original
  const [transformer_model_name, setTransformerModelName] = useState('');
  const [vae_model_name, setVaeModelName] = useState('');
  const [transformer_weight_dtype, setTransformerWeightDtype] = useState('BFLOAT_16');
  const [text_encoder_weight_dtype, setTextEncoderWeightDtype] = useState('BFLOAT_16');
  const [text_encoder_2_weight_dtype, setTextEncoder2WeightDtype] = useState('BFLOAT_16');
  const [text_encoder_3_weight_dtype, setTextEncoder3WeightDtype] = useState('BFLOAT_16');
  const [vae_weight_dtype, setVaeWeightDtype] = useState('FLOAT_32');  // Match original presets
  const [quantization_layer_filter, setQuantizationLayerFilter] = useState('');
  const [svd_dtype, setSvdDtype] = useState('NONE');
  const [svd_rank, setSvdRank] = useState(64);
  const [output_model_destination, setOutputModelDestination] = useState('');
  const [output_dtype, setOutputDtype] = useState('FLOAT_16');
  const [output_model_format, setOutputModelFormat] = useState('SAFETENSORS');
  const [include_train_config, setIncludeTrainConfig] = useState('SETTINGS');

  // ============= DATA STATE =============
  const [resolution, setResolution] = useState(512);
  const [batch_size, setBatchSize] = useState(2);
  const [gradient_accumulation_steps, setGradientAccumulationSteps] = useState(1);
  const [dataloader_threads, setDataloaderThreads] = useState(1);
  const [latent_caching, setLatentCaching] = useState(true);

  // ============= BACKUP STATE =============
  const [backup_dir, setBackupDir] = useState('');
  const [backup_every_n_steps, setBackupEveryNSteps] = useState(500);
  const [keep_n_backups, setKeepNBackups] = useState(3);
  const [save_checkpoints, setSaveCheckpoints] = useState(true);

  // ============= SYNC FROM STORE =============
  useEffect(() => {
    if (storeConfig && Object.keys(storeConfig).length > 0) {
      const c = storeConfig as any;
      // General
      setWorkspaceDir(c.workspace_dir || c.output_base_dir || '');
      setCacheDir(c.cache_dir || c.latent_cache_dir || '');
      setSamplesDir(c.samples_dir || '');
      setTensorboard(c.tensorboard ?? true);
      setDebug(c.debug ?? false);
      setValidationSteps(c.validation_steps ?? 500);
      setContinueLastBackup(c.continue_last_backup ?? false);
      setOnlyCache(c.only_cache ?? false);
      setDebugDir(c.debug_dir || '');
      setTensorboardExpose(c.tensorboard_expose ?? false);
      setTensorboardAlwaysOn(c.tensorboard_always_on ?? false);
      setTensorboardPort(c.tensorboard_port ?? 6006);
      setWandb(c.wandb ?? false);
      setWandbProject(c.wandb_project || '');
      setWandbEntity(c.wandb_entity || '');
      setWandbRunName(c.wandb_run_name || '');
      setWandbTags(c.wandb_tags || '');
      setWandbBaseUrl(c.wandb_base_url || '');
      setValidateAfterUnit(c.validate_after_unit || 'EPOCH');
      setTrainDevice(c.train_device || 'cuda');
      setMultiGpu(c.multi_gpu ?? false);
      setDeviceIndexes(c.device_indexes || '');
      setGradientReducePrevision(c.gradient_reduce_prevision || 'FLOAT_32_STOCHASTIC');
      setFusedGradientReduce(c.fused_gradient_reduce ?? false);
      setAsyncGradientReduce(c.async_gradient_reduce ?? true);
      setAsyncGradientReduceBuffer(c.async_gradient_reduce_buffer ?? 100);
      setTempDevice(c.temp_device || 'cpu');
      // Model
      setModelType(c.model_type || 'FLUX_DEV_1');
      setTrainingMethod(c.training_method || 'LORA');
      setBaseModelName(c.base_model_name || '');
      setHuggingfaceToken(c.huggingface_token || '');
      setCompile(c.compile ?? true);
      setTransformerModelName(c.transformer_model_name || '');
      setVaeModelName(c.vae_model_name || '');
      setTransformerWeightDtype(c.transformer_weight_dtype || 'BFLOAT_16');
      setTextEncoderWeightDtype(c.text_encoder_weight_dtype || 'BFLOAT_16');
      setTextEncoder2WeightDtype(c.text_encoder_2_weight_dtype || 'BFLOAT_16');
      setTextEncoder3WeightDtype(c.text_encoder_3_weight_dtype || 'BFLOAT_16');
      setVaeWeightDtype(c.vae_weight_dtype || 'BFLOAT_16');
      setQuantizationLayerFilter(c.quantization_layer_filter || '');
      setSvdDtype(c.svd_dtype || 'NONE');
      setSvdRank(c.svd_rank ?? 64);
      setOutputModelDestination(c.output_model_destination || '');
      setOutputDtype(c.output_dtype || 'FLOAT_16');
      setOutputModelFormat(c.output_model_format || 'SAFETENSORS');
      setIncludeTrainConfig(c.include_train_config || 'SETTINGS');
      // Data
      setResolution(c.resolution ?? 512);
      setBatchSize(c.batch_size ?? 2);
      setGradientAccumulationSteps(c.gradient_accumulation_steps ?? 1);
      setDataloaderThreads(c.dataloader_threads ?? 1);
      setLatentCaching(c.latent_caching ?? true);
      // Backup
      setBackupDir(c.backup_dir || c.backup_output_dir || '');
      setBackupEveryNSteps(c.backup_every_n_steps ?? c.backup_after_n_steps ?? 500);
      setKeepNBackups(c.keep_n_backups ?? c.rolling_backup_count ?? 3);
      setSaveCheckpoints(c.save_checkpoints ?? c.rolling_backup ?? true);
    }
  }, [storeConfig]);

  // ============= UPDATE HELPER =============
  const update = (field: string, value: any) => {
    updateConfig({ [field]: value } as any);
  };

  // Create workspace subdirectories and populate related paths
  const setupWorkspace = async () => {
    if (!workspace_dir.trim()) return;

    const subdirs = ['cache', 'debug', 'model', 'backup', 'samples'];

    for (const subdir of subdirs) {
      try {
        await fetch(`/api/filesystem/mkdir?path=${encodeURIComponent(workspace_dir + '/' + subdir)}`, {
          method: 'POST'
        });
      } catch (err) {
        console.error(`Failed to create ${subdir} dir:`, err);
      }
    }

    // Auto-populate cache directory
    const cachePath = workspace_dir + '/cache';
    setCacheDir(cachePath);
    update('cache_dir', cachePath);

    // Auto-populate debug directory
    const debugPath = workspace_dir + '/debug';
    setDebugDir(debugPath);
    update('debug_dir', debugPath);

    // Auto-populate model output directory
    const modelPath = workspace_dir + '/model';
    setOutputModelDestination(modelPath);
    update('output_model_destination', modelPath);

    // Auto-populate backup directory
    const backupPath = workspace_dir + '/backup';
    setBackupDir(backupPath);
    update('backup_dir', backupPath);

    // Auto-populate samples directory
    const samplesPath = workspace_dir + '/samples';
    setSamplesDir(samplesPath);
    update('samples_dir', samplesPath);
  };

  // ============= MODEL HELPERS =============
  const hasTransformer = ['STABLE_DIFFUSION_3', 'STABLE_DIFFUSION_35', 'PIXART_ALPHA', 'PIXART_SIGMA',
    'FLUX_DEV_1', 'FLUX_FILL_DEV_1', 'SANA', 'HUNYUAN_VIDEO', 'HI_DREAM_FULL', 'CHROMA_1', 'QWEN', 'Z_IMAGE'].includes(model_type);
  const hasUnet = ['STABLE_DIFFUSION_15', 'STABLE_DIFFUSION_15_INPAINTING', 'STABLE_DIFFUSION_20',
    'STABLE_DIFFUSION_20_BASE', 'STABLE_DIFFUSION_20_INPAINTING', 'STABLE_DIFFUSION_20_DEPTH',
    'STABLE_DIFFUSION_21', 'STABLE_DIFFUSION_21_BASE', 'STABLE_DIFFUSION_XL_10_BASE',
    'STABLE_DIFFUSION_XL_10_BASE_INPAINTING'].includes(model_type);
  const hasMultipleTextEncoders = ['STABLE_DIFFUSION_3', 'STABLE_DIFFUSION_35', 'STABLE_DIFFUSION_XL_10_BASE',
    'STABLE_DIFFUSION_XL_10_BASE_INPAINTING', 'FLUX_DEV_1', 'FLUX_FILL_DEV_1', 'HUNYUAN_VIDEO', 'HI_DREAM_FULL'].includes(model_type);
  const hasThreeTextEncoders = ['STABLE_DIFFUSION_3', 'STABLE_DIFFUSION_35', 'HI_DREAM_FULL'].includes(model_type);

  // ============= TOGGLE HELPER =============
  const Toggle = ({ value, onChange, label, description }: { value: boolean; onChange: (v: boolean) => void; label: string; description?: string }) => (
    <div className="flex items-center justify-between">
      <div>
        <div className="text-sm text-white">{label}</div>
        {description && <div className="text-xs text-muted">{description}</div>}
      </div>
      <button onClick={() => onChange(!value)}
        className={`relative w-9 h-5 rounded-full transition-colors ${value ? 'bg-green-600' : 'bg-gray-600'}`}>
        <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform ${value ? 'translate-x-4' : 'translate-x-0'}`} />
      </button>
    </div>
  );

  // ============= RENDER TABS =============
  const renderGeneralTab = () => (
    <div className="grid grid-cols-2 gap-4">
      <div className="space-y-4">
        <div className="bg-dark-surface rounded-lg border border-dark-border p-4 space-y-3">
          <h2 className="text-xs font-medium text-muted uppercase tracking-wider mb-3">General Settings</h2>

          <div>
            <label className="text-xs text-muted block mb-1">Workspace Directory</label>
            <div className="flex gap-2">
              <input type="text" value={workspace_dir} onChange={(e) => { setWorkspaceDir(e.target.value); update('workspace_dir', e.target.value); }}
                className="input w-full text-sm" placeholder="/path/to/workspace" />
              <button
                onClick={setupWorkspace}
                className="px-3 bg-primary hover:bg-primary-hover border border-primary rounded text-white flex items-center gap-1"
                title="Create cache, debug, model, backup subdirectories"
              >
                <FolderPlus className="w-4 h-4" />
              </button>
            </div>
          </div>

          <Toggle value={continue_last_backup} onChange={(v) => { setContinueLastBackup(v); update('continue_last_backup', v); }} label="Continue from last backup" />
          <Toggle value={debug} onChange={(v) => { setDebug(v); update('debug', v); }} label="Debug mode" />
          <Toggle value={tensorboard} onChange={(v) => { setTensorboard(v); update('tensorboard', v); }} label="Tensorboard" />
          <Toggle value={tensorboard_expose} onChange={(v) => { setTensorboardExpose(v); update('tensorboard_expose', v); }} label="Expose Tensorboard" />

          <div className="pt-2">
            <Toggle value={validation_steps > 0} onChange={(v) => { const steps = v ? 500 : 0; setValidationSteps(steps); update('validation_steps', steps); }} label="Validation" description="(Enable by setting steps)" />
          </div>

          <div>
            <label className="text-xs text-muted block mb-1">Dataloader Threads</label>
            <input type="text" value={dataloader_threads} onChange={(e) => { setDataloaderThreads(parseInt(e.target.value) || 1); update('dataloader_threads', parseInt(e.target.value) || 1); }} className="input w-full text-sm" />
          </div>

          <div>
            <label className="text-xs text-muted block mb-1">Train Device</label>
            <input type="text" value={train_device} onChange={(e) => { setTrainDevice(e.target.value); update('train_device', e.target.value); }} className="input w-full text-sm" />
          </div>

          <Toggle value={multi_gpu} onChange={(v) => { setMultiGpu(v); update('multi_gpu', v); }} label="Multi-GPU" />

          <div>
            <label className="text-xs text-muted block mb-1">Gradient Reduce Precision</label>
            <select value={gradient_reduce_prevision} onChange={(e) => { setGradientReducePrevision(e.target.value); update('gradient_reduce_prevision', e.target.value); }} className="input w-full text-sm">
              <option value="FLOAT_32">FLOAT_32</option>
              <option value="FLOAT_16">FLOAT_16</option>
              <option value="BFLOAT_16">BFLOAT_16</option>
              <option value="FLOAT_32_STOCHASTIC">FLOAT_32_STOCHASTIC</option>
            </select>
          </div>

          <Toggle value={async_gradient_reduce} onChange={(v) => { setAsyncGradientReduce(v); update('async_gradient_reduce', v); }} label="Async Gradient Reduce" />

          <div>
            <label className="text-xs text-muted block mb-1">Temp Device</label>
            <input type="text" value={temp_device} onChange={(e) => { setTempDevice(e.target.value); update('temp_device', e.target.value); }} className="input w-full text-sm" />
          </div>
        </div>
      </div>

      <div className="space-y-4">
        <div className="bg-dark-surface rounded-lg border border-dark-border p-4 space-y-3">
          <div className="h-[24px]"></div>

          <div>
            <label className="text-xs text-muted block mb-1">Cache Directory</label>
            <div className="flex gap-2">
              <input type="text" value={cache_dir} onChange={(e) => { setCacheDir(e.target.value); update('cache_dir', e.target.value); }} className="input w-full text-sm" />
              <button className="px-3 bg-dark-bg border border-dark-border rounded hover:bg-dark-hover text-muted">...</button>
            </div>
          </div>

          <div>
            <label className="text-xs text-muted block mb-1">Samples Directory</label>
            <div className="flex gap-2">
              <input type="text" value={samples_dir} onChange={(e) => { setSamplesDir(e.target.value); update('samples_dir', e.target.value); }} className="input w-full text-sm" />
              <button className="px-3 bg-dark-bg border border-dark-border rounded hover:bg-dark-hover text-muted">...</button>
            </div>
          </div>

          <Toggle value={only_cache} onChange={(v) => { setOnlyCache(v); update('only_cache', v); }} label="Only Cache" />

          <div>
            <label className="text-xs text-muted block mb-1">Debug Directory</label>
            <div className="flex gap-2">
              <input type="text" value={debug_dir} onChange={(e) => { setDebugDir(e.target.value); update('debug_dir', e.target.value); }} className="input w-full text-sm" />
              <button className="px-3 bg-dark-bg border border-dark-border rounded hover:bg-dark-hover text-muted">...</button>
            </div>
          </div>

          <Toggle value={tensorboard_always_on} onChange={(v) => { setTensorboardAlwaysOn(v); update('tensorboard_always_on', v); }} label="Always-On Tensorboard" />

          <div>
            <label className="text-xs text-muted block mb-1">Tensorboard Port</label>
            <input type="text" value={tensorboard_port} onChange={(e) => { setTensorboardPort(parseInt(e.target.value) || 6006); update('tensorboard_port', parseInt(e.target.value) || 6006); }} className="input w-full text-sm" />
          </div>

        </div>
      </div>

      {/* WandB Settings */}
      <div className="bg-dark-surface rounded-lg border border-dark-border p-4 space-y-3">
        <h2 className="text-xs font-medium text-muted uppercase tracking-wider mb-3">Weights & Biases</h2>
        <Toggle value={wandb} onChange={(v) => { setWandb(v); update('wandb', v); }} label="Enable WandB" />
        {wandb && (
          <>
            <div>
              <label className="text-xs text-muted block mb-1">Project Name</label>
              <input type="text" value={wandb_project} onChange={(e) => { setWandbProject(e.target.value); update('wandb_project', e.target.value); }}
                className="input w-full text-sm" placeholder="onetrainer" />
            </div>
            <div>
              <label className="text-xs text-muted block mb-1">Entity (optional)</label>
              <input type="text" value={wandb_entity} onChange={(e) => { setWandbEntity(e.target.value); update('wandb_entity', e.target.value); }}
                className="input w-full text-sm" placeholder="Your username or team" />
            </div>
            <div>
              <label className="text-xs text-muted block mb-1">Run Name (optional)</label>
              <input type="text" value={wandb_run_name} onChange={(e) => { setWandbRunName(e.target.value); update('wandb_run_name', e.target.value); }}
                className="input w-full text-sm" placeholder="Auto-generated if empty" />
            </div>
            <div>
              <label className="text-xs text-muted block mb-1">Tags (comma-separated)</label>
              <input type="text" value={wandb_tags} onChange={(e) => { setWandbTags(e.target.value); update('wandb_tags', e.target.value); }}
                className="input w-full text-sm" placeholder="flux,lora,experiment" />
            </div>
            <div>
              <label className="text-xs text-muted block mb-1">Server URL (self-hosted)</label>
              <input type="text" value={wandb_base_url} onChange={(e) => { setWandbBaseUrl(e.target.value); update('wandb_base_url', e.target.value); }}
                className="input w-full text-sm" placeholder="http://localhost:8080 (leave empty for wandb.ai)" />
            </div>
            <p className="text-xs text-muted">Run <code className="bg-dark-bg px-1 rounded">./start-wandb-server.sh</code> for local server, or <code className="bg-dark-bg px-1 rounded">wandb login</code> for cloud.</p>
          </>
        )}
      </div>

      {/* Validation Settings */}
      <div className="bg-dark-surface rounded-lg border border-dark-border p-4 space-y-3">
        <h2 className="text-xs font-medium text-muted uppercase tracking-wider mb-3">Validation & GPU</h2>

        <div>
          <label className="text-xs text-muted block mb-1">Validate after</label>
          <div className="flex gap-2">
            <input type="text" value={validation_steps} onChange={(e) => { setValidationSteps(parseInt(e.target.value) || 0); update('validation_steps', parseInt(e.target.value) || 0); }} className="input flex-1 text-sm" />
            <select value={validate_after_unit} onChange={(e) => { setValidateAfterUnit(e.target.value); update('validate_after_unit', e.target.value); }} className="input w-32 text-sm">
              <option value="EPOCH">EPOCH</option>
              <option value="STEP">STEP</option>
              <option value="SECOND">SECOND</option>
              <option value="MINUTE">MINUTE</option>
            </select>
          </div>
        </div>

        <div>
          <label className="text-xs text-muted block mb-1">Device Indexes</label>
          <input type="text" value={device_indexes} onChange={(e) => { setDeviceIndexes(e.target.value); update('device_indexes', e.target.value); }} className="input w-full text-sm" placeholder="0,1..." />
        </div>

        <Toggle value={fused_gradient_reduce} onChange={(v) => { setFusedGradientReduce(v); update('fused_gradient_reduce', v); }} label="Fused Gradient Reduce" />

        <div>
          <label className="text-xs text-muted block mb-1">Buffer size (MB)</label>
          <input type="text" value={async_gradient_reduce_buffer} onChange={(e) => { setAsyncGradientReduceBuffer(parseInt(e.target.value) || 100); update('async_gradient_reduce_buffer', parseInt(e.target.value) || 100); }} className="input w-full text-sm" />
        </div>

      </div>
    </div>
  );

  const renderModelTab = () => (
    <div className="grid grid-cols-2 gap-4">
      {/* Left Column */}
      <div className="space-y-4">
        <div className="bg-dark-surface rounded-lg border border-dark-border p-4 space-y-3">
          <h2 className="text-xs font-medium text-muted uppercase tracking-wider mb-3">Base Model</h2>
          <div>
            <label className="text-xs text-muted block mb-1">Model Type</label>
            <select value={model_type} onChange={(e) => { setModelType(e.target.value); update('model_type', e.target.value); }} className="input w-full text-sm">
              {MODEL_TYPES.map(m => <option key={m} value={m}>{m}</option>)}
            </select>
          </div>
          <div>
            <label className="text-xs text-muted block mb-1">Training Method</label>
            <select value={training_method} onChange={(e) => { setTrainingMethod(e.target.value); update('training_method', e.target.value); }} className="input w-full text-sm">
              {TRAINING_METHODS.map(m => <option key={m} value={m}>{m}</option>)}
            </select>
          </div>
          <div>
            <label className="text-xs text-muted block mb-1">Hugging Face Token</label>
            <input type="password" value={huggingface_token} onChange={(e) => { setHuggingfaceToken(e.target.value); update('huggingface_token', e.target.value); }}
              className="input w-full text-sm" placeholder="Optional: for protected repos" />
          </div>
          <div>
            <label className="text-xs text-muted block mb-1">Base Model</label>
            <input type="text" value={base_model_name} onChange={(e) => { setBaseModelName(e.target.value); update('base_model_name', e.target.value); }}
              className="input w-full text-sm" placeholder="Path or HuggingFace repo" />
          </div>
          <Toggle value={compile} onChange={(v) => { setCompile(v); update('compile', v); }} label="Compile Transformer Blocks" />
        </div>

        <div className="bg-dark-surface rounded-lg border border-dark-border p-4 space-y-3">
          <h2 className="text-xs font-medium text-muted uppercase tracking-wider mb-3">Model Overrides</h2>
          {hasTransformer && (
            <div>
              <label className="text-xs text-muted block mb-1">Override Transformer / GGUF</label>
              <input type="text" value={transformer_model_name} onChange={(e) => { setTransformerModelName(e.target.value); update('transformer_model_name', e.target.value); }}
                className="input w-full text-sm" placeholder="Optional" />
            </div>
          )}
          <div>
            <label className="text-xs text-muted block mb-1">VAE Override</label>
            <input type="text" value={vae_model_name} onChange={(e) => { setVaeModelName(e.target.value); update('vae_model_name', e.target.value); }}
              className="input w-full text-sm" placeholder="Optional" />
          </div>
        </div>

        <div className="bg-dark-surface rounded-lg border border-dark-border p-4 space-y-3">
          <h2 className="text-xs font-medium text-muted uppercase tracking-wider mb-3">Quantization</h2>
          <div>
            <label className="text-xs text-muted block mb-1">Quantization Layer Filter</label>
            <input type="text" value={quantization_layer_filter} onChange={(e) => { setQuantizationLayerFilter(e.target.value); update('quantization_layer_filter', e.target.value); }}
              className="input w-full text-sm" placeholder="Comma-separated layers" />
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="text-xs text-muted block mb-1">SVDQuant</label>
              <select value={svd_dtype} onChange={(e) => { setSvdDtype(e.target.value); update('svd_dtype', e.target.value); }} className="input w-full text-sm">
                <option value="NONE">disabled</option>
                <option value="FLOAT_32">float32</option>
                <option value="BFLOAT_16">bfloat16</option>
              </select>
            </div>
            <div>
              <label className="text-xs text-muted block mb-1">SVDQuant Rank</label>
              <input type="text" value={svd_rank} onChange={(e) => { setSvdRank(parseInt(e.target.value) || 64); update('svd_rank', parseInt(e.target.value) || 64); }}
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
              <select value={transformer_weight_dtype} onChange={(e) => { setTransformerWeightDtype(e.target.value); update('transformer_weight_dtype', e.target.value); }} className="input w-full text-sm">
                {TRANSFORMER_DATA_TYPES_WITH_GGUF.map(d => <option key={d.value} value={d.value}>{d.label}</option>)}
              </select>
            </div>
          )}
          {hasUnet && (
            <div>
              <label className="text-xs text-muted block mb-1">UNet Data Type</label>
              <select value={transformer_weight_dtype} onChange={(e) => { setTransformerWeightDtype(e.target.value); update('transformer_weight_dtype', e.target.value); }} className="input w-full text-sm">
                {TRANSFORMER_DATA_TYPES.map(d => <option key={d.value} value={d.value}>{d.label}</option>)}
              </select>
            </div>
          )}
          <div>
            <label className="text-xs text-muted block mb-1">{hasMultipleTextEncoders ? 'Text Encoder 1' : 'Text Encoder'} Data Type</label>
            <select value={text_encoder_weight_dtype} onChange={(e) => { setTextEncoderWeightDtype(e.target.value); update('text_encoder_weight_dtype', e.target.value); }} className="input w-full text-sm">
              {BASE_DATA_TYPES.map(d => <option key={d.value} value={d.value}>{d.label}</option>)}
            </select>
          </div>
          {hasMultipleTextEncoders && (
            <div>
              <label className="text-xs text-muted block mb-1">Text Encoder 2 Data Type</label>
              <select value={text_encoder_2_weight_dtype} onChange={(e) => { setTextEncoder2WeightDtype(e.target.value); update('text_encoder_2_weight_dtype', e.target.value); }} className="input w-full text-sm">
                {BASE_DATA_TYPES.map(d => <option key={d.value} value={d.value}>{d.label}</option>)}
              </select>
            </div>
          )}
          {hasThreeTextEncoders && (
            <div>
              <label className="text-xs text-muted block mb-1">Text Encoder 3 Data Type</label>
              <select value={text_encoder_3_weight_dtype} onChange={(e) => { setTextEncoder3WeightDtype(e.target.value); update('text_encoder_3_weight_dtype', e.target.value); }} className="input w-full text-sm">
                {BASE_DATA_TYPES.map(d => <option key={d.value} value={d.value}>{d.label}</option>)}
              </select>
            </div>
          )}
          <div>
            <label className="text-xs text-muted block mb-1">VAE Data Type</label>
            <select value={vae_weight_dtype} onChange={(e) => { setVaeWeightDtype(e.target.value); update('vae_weight_dtype', e.target.value); }} className="input w-full text-sm">
              {BASE_DATA_TYPES.map(d => <option key={d.value} value={d.value}>{d.label}</option>)}
            </select>
          </div>
        </div>

        <div className="bg-dark-surface rounded-lg border border-dark-border p-4 space-y-3">
          <h2 className="text-xs font-medium text-muted uppercase tracking-wider mb-3">Output</h2>
          <div>
            <label className="text-xs text-muted block mb-1">Model Output Destination</label>
            <input type="text" value={output_model_destination} onChange={(e) => { setOutputModelDestination(e.target.value); update('output_model_destination', e.target.value); }}
              className="input w-full text-sm" placeholder="Output path" />
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="text-xs text-muted block mb-1">Output Data Type</label>
              <select value={output_dtype} onChange={(e) => { setOutputDtype(e.target.value); update('output_dtype', e.target.value); }} className="input w-full text-sm">
                {OUTPUT_DATA_TYPES.map(d => <option key={d.value} value={d.value}>{d.label}</option>)}
              </select>
            </div>
            <div>
              <label className="text-xs text-muted block mb-1">Output Format</label>
              <select value={output_model_format} onChange={(e) => { setOutputModelFormat(e.target.value); update('output_model_format', e.target.value); }} className="input w-full text-sm">
                {OUTPUT_FORMATS.map(f => <option key={f} value={f}>{f}</option>)}
              </select>
            </div>
          </div>
          <div>
            <label className="text-xs text-muted block mb-1">Include Config</label>
            <select value={include_train_config} onChange={(e) => { setIncludeTrainConfig(e.target.value); update('include_train_config', e.target.value); }} className="input w-full text-sm">
              {CONFIG_INCLUDE.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>
        </div>
      </div>
    </div>
  );

  const renderDataTab = () => (
    <div className="max-w-2xl space-y-4">
      <div className="bg-dark-surface rounded-lg border border-dark-border p-4 space-y-3">
        <h2 className="text-xs font-medium text-muted uppercase tracking-wider mb-3">Dataset Settings</h2>
        <div>
          <label className="text-xs text-muted block mb-1">Resolution</label>
          <input type="text" value={resolution} onChange={(e) => { setResolution(parseInt(e.target.value) || 512); update('resolution', parseInt(e.target.value) || 512); }}
            className="input w-full text-sm" min="64" step="64" />
          <p className="text-xs text-muted mt-1">Image resolution for training (typically 512, 768, or 1024)</p>
        </div>
        <div>
          <label className="text-xs text-muted block mb-1">Batch Size</label>
          <input type="text" value={batch_size} onChange={(e) => { setBatchSize(parseInt(e.target.value) || 1); update('batch_size', parseInt(e.target.value) || 1); }}
            className="input w-full text-sm" min="1" />
          <p className="text-xs text-muted mt-1">Number of samples per training batch</p>
        </div>
        <div>
          <label className="text-xs text-muted block mb-1">Gradient Accumulation Steps</label>
          <input type="text" value={gradient_accumulation_steps} onChange={(e) => { setGradientAccumulationSteps(parseInt(e.target.value) || 1); update('gradient_accumulation_steps', parseInt(e.target.value) || 1); }}
            className="input w-full text-sm" min="1" />
          <p className="text-xs text-muted mt-1">Number of steps to accumulate gradients before updating weights</p>
        </div>
        <div>
          <label className="text-xs text-muted block mb-1">Dataloader Threads</label>
          <input type="text" value={dataloader_threads} onChange={(e) => { setDataloaderThreads(parseInt(e.target.value) || 1); update('dataloader_threads', parseInt(e.target.value) || 1); }}
            className="input w-full text-sm" min="1" />
          <p className="text-xs text-muted mt-1">Number of worker threads for data loading</p>
        </div>
      </div>

      <div className="bg-dark-surface rounded-lg border border-dark-border p-4">
        <h2 className="text-xs font-medium text-muted uppercase tracking-wider mb-3">Performance Options</h2>
        <Toggle value={latent_caching} onChange={(v) => { setLatentCaching(v); update('latent_caching', v); }}
          label="Latent Caching" description="Cache VAE-encoded latents to speed up training" />
      </div>
    </div>
  );

  const renderBackupTab = () => (
    <div className="max-w-2xl space-y-4">
      <div className="bg-dark-surface rounded-lg border border-dark-border p-4 space-y-3">
        <h2 className="text-xs font-medium text-muted uppercase tracking-wider mb-3">Backup Settings</h2>
        <div>
          <label className="text-xs text-muted block mb-1">Backup Directory</label>
          <input type="text" value={backup_dir} onChange={(e) => { setBackupDir(e.target.value); update('backup_dir', e.target.value); }}
            className="input w-full text-sm" placeholder="/path/to/backups" />
          <p className="text-xs text-muted mt-1">Directory where backups will be stored</p>
        </div>
        <div>
          <label className="text-xs text-muted block mb-1">Backup Every N Steps</label>
          <input type="text" value={backup_every_n_steps} onChange={(e) => { setBackupEveryNSteps(parseInt(e.target.value) || 500); update('backup_every_n_steps', parseInt(e.target.value) || 500); }}
            className="input w-full text-sm" min="1" />
          <p className="text-xs text-muted mt-1">Create a backup every N training steps</p>
        </div>
        <div>
          <label className="text-xs text-muted block mb-1">Keep N Backups</label>
          <input type="text" value={keep_n_backups} onChange={(e) => { setKeepNBackups(parseInt(e.target.value) || 3); update('keep_n_backups', parseInt(e.target.value) || 3); }}
            className="input w-full text-sm" min="1" />
          <p className="text-xs text-muted mt-1">Maximum number of backups to retain (older backups will be deleted)</p>
        </div>
        <Toggle value={save_checkpoints} onChange={(v) => { setSaveCheckpoints(v); update('save_checkpoints', v); }}
          label="Save Checkpoints" description="Include model checkpoints in backups" />
      </div>
    </div>
  );

  return (
    <div className="h-full flex flex-col">
      {/* Header with Tabs */}
      <div className="h-14 flex items-center px-6 border-b border-dark-border bg-dark-surface">
        <h1 className="text-lg font-medium text-white mr-8">Configuration</h1>
        <div className="flex gap-1">
          {TABS.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-2 text-sm rounded-t transition-colors ${activeTab === tab.id
                  ? 'text-white bg-dark-bg border-b-2 border-primary'
                  : 'text-muted hover:text-white'
                  }`}
              >
                <Icon className="w-4 h-4" />
                {tab.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-4">
        {activeTab === 'general' && renderGeneralTab()}
        {activeTab === 'model' && renderModelTab()}
        {activeTab === 'data' && renderDataTab()}
        {activeTab === 'backup' && renderBackupTab()}
      </div>
    </div>
  );
}
