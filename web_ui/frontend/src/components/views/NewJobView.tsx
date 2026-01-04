import { useState, useEffect } from 'react';
import { HelpCircle, X, Save, ExternalLink, MoreHorizontal } from 'lucide-react';
import { configApi, trainingApi, type PresetInfo } from '../../lib/api';
import { useConfigStore } from '../../stores/configStore';
import { MODEL_TYPES, getTrainingMethods } from '../../model_constants';

interface NewJobViewProps {
  onViewChange?: (view: string) => void;
}

export function NewJobView({ onViewChange }: NewJobViewProps) {
  const [presets, setPresets] = useState<PresetInfo[]>([]);
  const [selectedPreset, setSelectedPreset] = useState<string>('');
  const [config, setConfigLocal] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [modelType, setModelType] = useState('FLUX_DEV_1');
  const [trainingMethod, setTrainingMethod] = useState('LORA');
  // Save config dialog state
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [savePresetName, setSavePresetName] = useState('');
  const [saving, setSaving] = useState(false);
  // Expanded settings panels
  const [expandedSettings, setExpandedSettings] = useState<Set<string>>(new Set());

  const toggleSettings = (settingKey: string) => {
    const newExpanded = new Set(expandedSettings);
    if (newExpanded.has(settingKey)) {
      newExpanded.delete(settingKey);
    } else {
      newExpanded.add(settingKey);
    }
    setExpandedSettings(newExpanded);
  };

  // Zustand store for sharing config across views (now persisted)
  const {
    config: storeConfig,
    currentPreset: storePreset,
    setConfig: setStoreConfig,
    setCurrentPreset,
    setWorkspacePath
  } = useConfigStore();

  // Sync local state from persisted store on mount
  useEffect(() => {
    if (storeConfig && Object.keys(storeConfig).length > 0) {
      setConfigLocal(storeConfig);
      const c = storeConfig as any;
      if (c.model_type) setModelType(c.model_type);
      if (c.training_method) setTrainingMethod(c.training_method);
    }
    if (storePreset) {
      setSelectedPreset(storePreset);
    }
  }, []); // Only on mount

  // Wrapper to update both local state and store
  const setConfig = (newConfig: any) => {
    setConfigLocal(newConfig);
    if (newConfig) {
      setStoreConfig(newConfig);
      // Update workspace path from loaded config
      if (newConfig.workspace_dir) {
        setWorkspacePath(newConfig.workspace_dir);
      }
    }
  };

  // Fetch available presets from training_presets folder (same as Dashboard)
  const fetchPresets = async () => {
    try {
      const response = await configApi.getPresets();
      const presetsList = response.data.presets || [];
      setPresets(presetsList);
      setError(null);

      if (presetsList.length === 0) {
        setError('No presets found. Check training_presets directory.');
      }
    } catch (err) {
      console.error('Failed to load presets:', err);
      setError('Failed to load presets');
    }
  };

  // Fetch presets on mount
  useEffect(() => {
    fetchPresets();
  }, []);

  // Load preset when selected
  const handlePresetChange = async (presetName: string) => {
    setSelectedPreset(presetName);
    setCurrentPreset(presetName || null);
    if (!presetName) {
      setConfig(null);
      return;
    }

    setLoading(true);
    setError(null);
    try {
      // Find preset in our list to get its path/config dir
      const preset = presets.find(p => p.name === presetName);
      let configDir: string | undefined;
      if (preset?.path) {
        // Extract the directory from the preset path
        const pathParts = preset.path.split('/');
        pathParts.pop(); // Remove filename
        configDir = pathParts.join('/');
      }

      const response = await configApi.loadPreset(presetName, configDir);
      // API returns { config: {...} } so extract the config
      const configData = response.data.config || response.data;
      setConfig(configData);
      // Also set in store for other views
      setStoreConfig(configData);
      // Update model type and training method from loaded config
      if (configData.model_type) {
        setModelType(configData.model_type);
      }
      if (configData.training_method) {
        setTrainingMethod(configData.training_method);
      }
    } catch (err) {
      console.error('Failed to load preset:', err);
      setError('Failed to load preset configuration');
    } finally {
      setLoading(false);
    }
  };

  // Create job handler
  const handleCreateJob = async () => {
    if (!config) {
      setError('Please select a preset first');
      return;
    }

    setStarting(true);
    setError(null);

    try {
      // 1. Validate configuration
      const validation = await configApi.validate(config);
      if (!validation.data.valid) {
        setError(validation.data.errors.join(', '));
        setStarting(false);
        return;
      }

      // Show warnings if any
      if (validation.data.warnings?.length > 0) {
        console.warn('Config warnings:', validation.data.warnings);
      }

      // 2. Save config to temp file
      const saveResponse = await configApi.saveTemp(config);
      const configPath = saveResponse.data.path;

      // 3. Start training
      await trainingApi.start(configPath);

      // 4. Navigate to dashboard to watch progress
      onViewChange?.('dashboard');

    } catch (err: any) {
      console.error('Failed to start training:', err);
      setError(err.response?.data?.detail || err.response?.data?.error || 'Failed to start training');
    } finally {
      setStarting(false);
    }
  };

  // Update training method when model type changes
  const handleModelTypeChange = (newModelType: string) => {
    setModelType(newModelType);
    const methods = getTrainingMethods(newModelType);
    // If current method isn't valid for new model, reset to first available
    if (!methods.find(m => m.value === trainingMethod)) {
      setTrainingMethod(methods[0].value);
    }
    let newConfig = { ...config, model_type: newModelType };

    // Auto-update base model name for specific types if it's empty or set to default SD 1.5
    if (newModelType === 'QWEN') {
      if (!config?.base_model_name || config.base_model_name.includes('stable-diffusion-v1-5')) {
        newConfig.base_model_name = 'Qwen/Qwen2-VL-7B-Instruct';
      }
    } else if (newModelType === 'QWEN_IMAGE_EDIT') {
      if (!config?.base_model_name || config.base_model_name.includes('stable-diffusion-v1-5')) {
        newConfig.base_model_name = 'Qwen/Qwen-Image-Edit';
      }
    } else if (newModelType === 'KANDINSKY_5' || newModelType === 'KANDINSKY_5_VIDEO') {
      if (!config?.base_model_name || config.base_model_name.includes('stable-diffusion-v1-5')) {
        newConfig.base_model_name = 'kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s';
      }
    }

    setConfig(newConfig);
  };

  const handleSaveConfig = () => {
    if (!config) {
      setError('No configuration to save');
      return;
    }
    // Generate default name based on model type and training method
    const defaultName = `${modelType}_${trainingMethod}_${new Date().toISOString().slice(0, 10)}`;
    setSavePresetName(defaultName);
    setShowSaveDialog(true);
  };

  const handleConfirmSave = async () => {
    if (!savePresetName.trim()) {
      setError('Please enter a preset name');
      return;
    }

    setSaving(true);
    setError(null);

    try {
      await configApi.savePreset(savePresetName.trim(), config);
      setShowSaveDialog(false);
      setSavePresetName('');
      // Refresh presets list and select the new one
      await fetchPresets();
      setSelectedPreset(savePresetName.trim());
    } catch (err: any) {
      console.error('Failed to save preset:', err);
      setError(err.response?.data?.detail || 'Failed to save preset');
    } finally {
      setSaving(false);
    }
  };

  const openWiki = () => {
    window.open('https://github.com/Nerogar/OneTrainer/wiki', '_blank');
  };

  const trainingMethods = getTrainingMethods(modelType);

  return (
    <div className="h-full flex flex-col">
      {/* Header - matches OneTrainer TopBar layout */}
      <div className="h-14 flex items-center justify-between px-4 border-b border-dark-border bg-dark-surface">
        <div className="flex items-center gap-3">
          <h1 className="text-lg font-medium text-white">New Training Job</h1>
        </div>

        {/* Center section: Preset, Save, Wiki */}
        <div className="flex items-center gap-2">
          {/* Preset Selector - with scrollbar support */}
          <select
            className="bg-dark-bg border border-dark-border rounded px-3 py-1.5 text-sm text-white min-w-[200px] max-h-[300px] overflow-y-auto"
            value={selectedPreset}
            onChange={(e) => handlePresetChange(e.target.value)}
            disabled={loading}
            style={{ maxHeight: '300px' }}
          >
            <option value="">Select a preset...</option>
            {presets.map((preset) => (
              <option key={preset.name} value={preset.name}>
                {preset.name}
              </option>
            ))}
          </select>

          {/* Save Config Button */}
          <button
            onClick={handleSaveConfig}
            className="bg-dark-bg border border-dark-border hover:bg-dark-hover text-white px-3 py-1.5 rounded text-sm flex items-center gap-1.5"
          >
            <Save className="w-4 h-4" />
            Save config
          </button>

          {/* Wiki Button */}
          <button
            onClick={openWiki}
            className="bg-dark-bg border border-dark-border hover:bg-dark-hover text-white px-3 py-1.5 rounded text-sm flex items-center gap-1.5"
          >
            <ExternalLink className="w-4 h-4" />
            Wiki
          </button>
        </div>

        {/* Right section: Model Type, Training Method, Create */}
        <div className="flex items-center gap-2">
          {/* Model Type Dropdown */}
          <select
            className="bg-dark-bg border border-dark-border rounded px-3 py-1.5 text-sm text-white"
            value={modelType}
            onChange={(e) => handleModelTypeChange(e.target.value)}
          >
            {MODEL_TYPES.map((type) => (
              <option key={type.value} value={type.value}>
                {type.label}
              </option>
            ))}
          </select>

          {/* Training Method Dropdown */}
          <select
            className="bg-dark-bg border border-dark-border rounded px-3 py-1.5 text-sm text-white"
            value={trainingMethod}
            onChange={(e) => {
              setTrainingMethod(e.target.value);
              setConfig({ ...config, training_method: e.target.value });
            }}
          >
            {trainingMethods.map((method) => (
              <option key={method.value} value={method.value}>
                {method.label}
              </option>
            ))}
          </select>

          {/* Start Training Button */}
          <button
            onClick={handleCreateJob}
            disabled={!config || loading || starting}
            className="bg-green-600 hover:bg-green-700 text-white px-4 py-1.5 rounded text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {starting ? 'Starting...' : loading ? 'Loading...' : 'Start Training'}
          </button>
        </div>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="bg-red-500/10 border-b border-red-500/20 px-6 py-3">
          <p className="text-sm text-red-400">{error}</p>
        </div>
      )}

      {/* Content */}
      <div className="flex-1 overflow-auto p-6">
        <div className="grid grid-cols-4 gap-6">
          {/* JOB Section */}
          <Section title="JOB">
            <Field label="Training Name" tooltip="Name for this training job">
              <input
                type="text"
                value={config?.training_name || 'my_first_lora_v1'}
                onChange={(e) => setConfig({ ...config, training_name: e.target.value })}
                className="input w-full"
              />
            </Field>
            <Field label="Epochs" tooltip="Number of training epochs">
              <input
                type="text"
                value={config?.epochs || 100}
                onChange={(e) => setConfig({ ...config, epochs: parseInt(e.target.value) || 100 })}
                className="input w-full"
                min="1"
              />
            </Field>
            <Field label="Workspace Dir" tooltip="Directory for training workspace and backups">
              <input
                type="text"
                value={config?.workspace_dir || ''}
                onChange={(e) => setConfig({ ...config, workspace_dir: e.target.value })}
                className="input w-full"
                placeholder="workspace/run"
              />
            </Field>
            <div className="flex items-center gap-2 py-1">
              <input
                type="checkbox"
                id="continueLastBackup"
                checked={config?.continue_last_backup || false}
                onChange={(e) => setConfig({ ...config, continue_last_backup: e.target.checked })}
                className="rounded"
              />
              <label htmlFor="continueLastBackup" className="text-sm text-white">Continue from last backup</label>
            </div>
            <div className="flex items-center gap-2 py-1">
              <input
                type="checkbox"
                id="clearCache"
                checked={config?.clear_cache_before_training !== false}
                onChange={(e) => setConfig({ ...config, clear_cache_before_training: e.target.checked })}
                className="rounded"
              />
              <label htmlFor="clearCache" className="text-sm text-white">Clear cache before training</label>
            </div>
          </Section>

          {/* MODEL Section */}
          <Section title="MODEL">
            <Field label="Base Model Path" tooltip="Model path or HuggingFace ID">
              <input
                type="text"
                value={config?.base_model_name || ''}
                onChange={(e) => setConfig({ ...config, base_model_name: e.target.value })}
                className="input w-full"
                placeholder="e.g., black-forest-labs/FLUX.1-dev"
              />
            </Field>
            <Field label="Output Model Destination" tooltip="Where to save the trained model">
              <input
                type="text"
                value={config?.output_model_destination || ''}
                onChange={(e) => setConfig({ ...config, output_model_destination: e.target.value })}
                className="input w-full"
                placeholder="models/my_lora"
              />
            </Field>
          </Section>

          {/* QUANTIZATION Section */}
          <Section title="QUANTIZATION">
            <Field label="Transformer">
              <select
                className="input w-full"
                value={config?.weight_dtype || 'FLOAT_8'}
                onChange={(e) => setConfig({ ...config, weight_dtype: e.target.value })}
              >
                <option value="FLOAT_8">float8 (default)</option>
                <option value="BFLOAT_16">bfloat16</option>
                <option value="FLOAT_16">float16</option>
              </select>
            </Field>
            <Field label="Text Encoder">
              <select
                className="input w-full"
                value={config?.text_encoder_dtype || 'FLOAT_8'}
                onChange={(e) => setConfig({ ...config, text_encoder_dtype: e.target.value })}
              >
                <option value="FLOAT_8">float8 (default)</option>
                <option value="BFLOAT_16">bfloat16</option>
              </select>
            </Field>
          </Section>

          {/* TARGET Section */}
          <Section title="TARGET">
            <Field label="Target Type">
              <select
                className="input w-full"
                value={config?.lora_model_name || 'lora'}
                onChange={(e) => setConfig({ ...config, lora_model_name: e.target.value })}
              >
                <option value="lora">LoRA</option>
                <option value="lokr">LoKr</option>
                <option value="locon">LoCon</option>
              </select>
            </Field>
            <Field label="Linear Rank">
              <input
                type="text"
                value={config?.lora_rank || 32}
                onChange={(e) => setConfig({ ...config, lora_rank: parseInt(e.target.value) })}
                className="input w-full"
              />
            </Field>
          </Section>
        </div>

        {/* TRAINING Section */}
        <div className="mt-6 bg-dark-surface rounded-lg border border-dark-border">
          <div className="px-4 py-3 border-b border-dark-border">
            <h2 className="text-sm font-medium text-muted uppercase tracking-wider">Training</h2>
          </div>
          <div className="p-4 grid grid-cols-4 gap-6">
            <Field label="Batch Size">
              <input
                type="text"
                value={config?.batch_size || 1}
                onChange={(e) => setConfig({ ...config, batch_size: parseInt(e.target.value) })}
                className="input w-full"
              />
            </Field>
            {/* Optimizer with expandable settings */}
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <label className="text-sm text-muted w-32">Optimizer</label>
                <div className="flex-1 flex gap-1">
                  <select
                    className="input flex-1"
                    value={config?.optimizer?.optimizer || 'ADAMW_8BIT'}
                    onChange={(e) => setConfig({ ...config, optimizer: { ...config?.optimizer, optimizer: e.target.value } })}
                  >
                    <option value="ADAMW">AdamW</option>
                    <option value="ADAMW_8BIT">AdamW 8Bit</option>
                    <option value="ADAM">Adam</option>
                    <option value="ADAM_8BIT">Adam 8Bit</option>
                    <option value="SGD">SGD</option>
                    <option value="SGD_8BIT">SGD 8Bit</option>
                    <option value="ADAGRAD">Adagrad</option>
                    <option value="ADAGRAD_8BIT">Adagrad 8Bit</option>
                    <option value="RMSPROP">RMSprop</option>
                    <option value="RMSPROP_8BIT">RMSprop 8Bit</option>
                    <option value="LION">Lion</option>
                    <option value="LION_8BIT">Lion 8Bit</option>
                    <option value="PRODIGY">Prodigy</option>
                    <option value="DADAPT_ADAM">D-Adapt Adam</option>
                    <option value="DADAPT_SGD">D-Adapt SGD</option>
                    <option value="ADAFACTOR">AdaFactor</option>
                    <option value="CAME">CAME</option>
                    <option value="SCHEDULE_FREE_ADAMW">Schedule-Free AdamW</option>
                  </select>
                  <button
                    type="button"
                    onClick={() => toggleSettings('optimizer')}
                    className="px-2 py-1 bg-dark-bg border border-dark-border rounded hover:bg-dark-hover text-muted hover:text-white"
                    title="Optimizer Settings"
                  >
                    <MoreHorizontal className="w-4 h-4" />
                  </button>
                </div>
              </div>
              {expandedSettings.has('optimizer') && (
                <div className="ml-32 pl-2 border-l-2 border-dark-border space-y-2 py-2">
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <label className="text-xs text-muted block mb-1">Weight Decay</label>
                      <input
                        type="text"
                        value={config?.optimizer?.weight_decay || '0.01'}
                        onChange={(e) => setConfig({ ...config, optimizer: { ...config?.optimizer, weight_decay: e.target.value } })}
                        className="input w-full text-sm"
                      />
                    </div>
                    <div>
                      <label className="text-xs text-muted block mb-1">Eps</label>
                      <input
                        type="text"
                        value={config?.optimizer?.eps || '1e-8'}
                        onChange={(e) => setConfig({ ...config, optimizer: { ...config?.optimizer, eps: e.target.value } })}
                        className="input w-full text-sm"
                      />
                    </div>
                    <div>
                      <label className="text-xs text-muted block mb-1">Beta1</label>
                      <input
                        type="text"
                        value={config?.optimizer?.beta1 || '0.9'}
                        onChange={(e) => setConfig({ ...config, optimizer: { ...config?.optimizer, beta1: e.target.value } })}
                        className="input w-full text-sm"
                      />
                    </div>
                    <div>
                      <label className="text-xs text-muted block mb-1">Beta2</label>
                      <input
                        type="text"
                        value={config?.optimizer?.beta2 || '0.999'}
                        onChange={(e) => setConfig({ ...config, optimizer: { ...config?.optimizer, beta2: e.target.value } })}
                        className="input w-full text-sm"
                      />
                    </div>
                    <div>
                      <label className="text-xs text-muted block mb-1">Momentum</label>
                      <input
                        type="text"
                        value={config?.optimizer?.momentum || '0'}
                        onChange={(e) => setConfig({ ...config, optimizer: { ...config?.optimizer, momentum: e.target.value } })}
                        className="input w-full text-sm"
                      />
                    </div>
                    <div>
                      <label className="text-xs text-muted block mb-1">Dampening</label>
                      <input
                        type="text"
                        value={config?.optimizer?.dampening || '0'}
                        onChange={(e) => setConfig({ ...config, optimizer: { ...config?.optimizer, dampening: e.target.value } })}
                        className="input w-full text-sm"
                      />
                    </div>
                  </div>
                  <div className="flex items-center gap-4 text-sm">
                    <label className="flex items-center gap-2 text-muted">
                      <input
                        type="checkbox"
                        checked={config?.optimizer?.fused || false}
                        onChange={(e) => setConfig({ ...config, optimizer: { ...config?.optimizer, fused: e.target.checked } })}
                        className="rounded"
                      />
                      Fused
                    </label>
                    <label className="flex items-center gap-2 text-muted">
                      <input
                        type="checkbox"
                        checked={config?.optimizer?.stochastic_rounding || false}
                        onChange={(e) => setConfig({ ...config, optimizer: { ...config?.optimizer, stochastic_rounding: e.target.checked } })}
                        className="rounded"
                      />
                      Stochastic Rounding
                    </label>
                  </div>
                </div>
              )}
            </div>
            <Field label="Timestep Type">
              <select
                className="input w-full"
                value={config?.timestep_distribution || 'SIGMOID'}
                onChange={(e) => setConfig({ ...config, timestep_distribution: e.target.value })}
              >
                <option value="SIGMOID">Sigmoid</option>
                <option value="LINEAR">Linear</option>
              </select>
            </Field>
            <Field label="Use EMA">
              <Toggle
                defaultChecked={config?.ema || false}
                onChange={(checked) => setConfig({ ...config, ema: checked })}
              />
            </Field>

            <Field label="Gradient Accumulation">
              <input
                type="text"
                value={config?.gradient_accumulation_steps || 1}
                onChange={(e) => setConfig({ ...config, gradient_accumulation_steps: parseInt(e.target.value) })}
                className="input w-full"
              />
            </Field>
            {/* Learning Rate with expandable scheduler settings */}
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <label className="text-sm text-muted w-32">Learning Rate</label>
                <div className="flex-1 flex gap-1">
                  <input
                    type="text"
                    value={config?.learning_rate || '0.0001'}
                    onChange={(e) => setConfig({ ...config, learning_rate: e.target.value })}
                    className="input flex-1"
                  />
                  <button
                    type="button"
                    onClick={() => toggleSettings('scheduler')}
                    className="px-2 py-1 bg-dark-bg border border-dark-border rounded hover:bg-dark-hover text-muted hover:text-white"
                    title="Scheduler Settings"
                  >
                    <MoreHorizontal className="w-4 h-4" />
                  </button>
                </div>
              </div>
              {expandedSettings.has('scheduler') && (
                <div className="ml-32 pl-2 border-l-2 border-dark-border space-y-2 py-2">
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <label className="text-xs text-muted block mb-1">LR Scheduler</label>
                      <select
                        value={config?.lr_scheduler || 'CONSTANT'}
                        onChange={(e) => setConfig({ ...config, lr_scheduler: e.target.value })}
                        className="input w-full text-sm"
                      >
                        <option value="CONSTANT">Constant</option>
                        <option value="LINEAR">Linear</option>
                        <option value="COSINE">Cosine</option>
                        <option value="COSINE_WITH_RESTARTS">Cosine w/ Restarts</option>
                        <option value="POLYNOMIAL">Polynomial</option>
                        <option value="ADAFACTOR">AdaFactor</option>
                        <option value="REX">REX</option>
                      </select>
                    </div>
                    <div>
                      <label className="text-xs text-muted block mb-1">Warmup Steps</label>
                      <input
                        type="text"
                        value={config?.lr_warmup_steps || 0}
                        onChange={(e) => setConfig({ ...config, lr_warmup_steps: parseInt(e.target.value) })}
                        className="input w-full text-sm"
                      />
                    </div>
                    <div>
                      <label className="text-xs text-muted block mb-1">Min LR</label>
                      <input
                        type="text"
                        value={config?.lr_min || '0'}
                        onChange={(e) => setConfig({ ...config, lr_min: e.target.value })}
                        className="input w-full text-sm"
                      />
                    </div>
                    <div>
                      <label className="text-xs text-muted block mb-1">Cycles</label>
                      <input
                        type="text"
                        value={config?.lr_cycles || 1}
                        onChange={(e) => setConfig({ ...config, lr_cycles: parseInt(e.target.value) })}
                        className="input w-full text-sm"
                      />
                    </div>
                  </div>
                </div>
              )}
            </div>
            <Field label="Timestep Bias">
              <select
                className="input w-full"
                value={config?.timestep_bias_strategy || 'NONE'}
                onChange={(e) => setConfig({ ...config, timestep_bias_strategy: e.target.value })}
              >
                <option value="NONE">None</option>
                <option value="EARLIER">Earlier</option>
                <option value="LATER">Later</option>
              </select>
            </Field>
            <Field label="Unload TE">
              <Toggle
                defaultChecked={config?.text_encoder?.unload || false}
                onChange={(checked) => setConfig({ ...config, text_encoder: { ...config?.text_encoder, unload: checked } })}
              />
            </Field>

            <Field label="Steps">
              <input
                type="text"
                value={config?.epochs || 3000}
                onChange={(e) => setConfig({ ...config, epochs: parseInt(e.target.value) })}
                className="input w-full"
              />
            </Field>
            <Field label="Loss Type">
              <select
                className="input w-full"
                value={config?.loss_type || 'MSE'}
                onChange={(e) => setConfig({ ...config, loss_type: e.target.value })}
              >
                <option value="MSE">Mean Squared Error</option>
                <option value="HUBER">Huber Loss</option>
              </select>
            </Field>
            <Field label="Cache Text Embeddings">
              <Toggle
                defaultChecked={config?.latent_caching || false}
                onChange={(checked) => setConfig({ ...config, latent_caching: checked })}
              />
            </Field>
          </div>
        </div>

        {/* SAVE Section */}
        <div className="mt-6 bg-dark-surface rounded-lg border border-dark-border">
          <div className="px-4 py-3 border-b border-dark-border">
            <h2 className="text-sm font-medium text-muted uppercase tracking-wider">Save</h2>
          </div>
          <div className="p-4 grid grid-cols-4 gap-6">
            <Field label="Data Type">
              <select
                className="input w-full"
                value={config?.output_dtype || 'BFLOAT_16'}
                onChange={(e) => setConfig({ ...config, output_dtype: e.target.value })}
              >
                <option value="BFLOAT_16">BF16</option>
                <option value="FLOAT_16">FP16</option>
              </select>
            </Field>
            <Field label="Save Every">
              <input
                type="text"
                value={config?.save_every || 250}
                onChange={(e) => setConfig({ ...config, save_every: parseInt(e.target.value) })}
                className="input w-full"
              />
            </Field>
            <Field label="Max Step Saves to Keep">
              <input
                type="text"
                value={config?.max_step_saves_to_keep || 4}
                onChange={(e) => setConfig({ ...config, max_step_saves_to_keep: parseInt(e.target.value) })}
                className="input w-full"
              />
            </Field>
          </div>
        </div>

        {/* DATASETS Section */}
        <div className="mt-6 bg-dark-surface rounded-lg border border-dark-border">
          <div className="px-4 py-3 border-b border-dark-border flex items-center justify-between">
            <h2 className="text-sm font-medium text-muted uppercase tracking-wider">Datasets</h2>
          </div>
          <div className="p-4">
            {/* Dataset Item */}
            <div className="bg-dark-bg rounded-lg border border-dark-border p-4">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-medium text-white">Dataset 1</h3>
                <button className="text-danger hover:text-red-400">
                  <X className="w-5 h-5" />
                </button>
              </div>
              <div className="grid grid-cols-4 gap-4">
                <Field label="Target Dataset">
                  <select className="input w-full">
                    <option>01_aines</option>
                  </select>
                </Field>
                <Field label="Default Caption">
                  <input type="text" placeholder="eg. A photo of a cat" className="input w-full" />
                </Field>
                <div>
                  <label className="text-sm text-muted block mb-2">Settings</label>
                  <div className="space-y-2">
                    <label className="flex items-center gap-2 text-sm">
                      <Toggle />
                      <span>Cache Latents</span>
                    </label>
                    <label className="flex items-center gap-2 text-sm">
                      <Toggle />
                      <span>Is Regularization</span>
                    </label>
                  </div>
                </div>
                <div>
                  <label className="text-sm text-muted block mb-2">Resolutions</label>
                  <div className="space-y-2">
                    {['256', '512', '768', '1024', '1280', '1536'].map((res) => (
                      <label key={res} className="flex items-center gap-2 text-sm">
                        <Toggle defaultChecked={res === '1024' || res === '512'} />
                        <span>{res}</span>
                      </label>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            <button className="w-full mt-4 py-3 border border-dark-border border-dashed rounded-lg text-muted hover:text-white hover:border-primary transition-colors">
              Add Dataset
            </button>
          </div>
        </div>

        {/* SAMPLE Section */}
        <div className="mt-6 bg-dark-surface rounded-lg border border-dark-border">
          <div className="px-4 py-3 border-b border-dark-border">
            <h2 className="text-sm font-medium text-muted uppercase tracking-wider">Sample</h2>
          </div>
          <div className="p-4 grid grid-cols-4 gap-6">
            <Field label="Sample Every">
              <input
                type="text"
                value={config?.sample_every || 250}
                onChange={(e) => setConfig({ ...config, sample_every: parseInt(e.target.value) })}
                className="input w-full"
              />
            </Field>
            <Field label="Width">
              <input
                type="text"
                value={config?.sample_definition?.width || 1024}
                onChange={(e) => setConfig({
                  ...config,
                  sample_definition: { ...config?.sample_definition, width: parseInt(e.target.value) }
                })}
                className="input w-full"
              />
            </Field>
            <Field label="Seed">
              <input
                type="text"
                value={config?.sample_definition?.seed || 42}
                onChange={(e) => setConfig({
                  ...config,
                  sample_definition: { ...config?.sample_definition, seed: parseInt(e.target.value) }
                })}
                className="input w-full"
              />
            </Field>
            <div>
              <label className="text-sm text-muted block mb-2">Advanced Sampling</label>
              <div className="space-y-2">
                <label className="flex items-center gap-2 text-sm">
                  <Toggle
                    defaultChecked={config?.non_ema_sampling || false}
                    onChange={(checked) => setConfig({ ...config, non_ema_sampling: checked })}
                  />
                  <span>Skip First Sample</span>
                </label>
                <label className="flex items-center gap-2 text-sm">
                  <Toggle
                    defaultChecked={config?.sample_definition?.enabled === false}
                    onChange={(checked) => setConfig({
                      ...config,
                      sample_definition: { ...config?.sample_definition, enabled: !checked }
                    })}
                  />
                  <span>Disable Sampling</span>
                </label>
              </div>
            </div>

            {/* Sampler with expandable settings */}
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <label className="text-sm text-muted w-24">Sampler</label>
                <div className="flex-1 flex gap-1">
                  <select
                    className="input flex-1"
                    value={config?.sample_definition?.sampler || 'FLOW_MATCH'}
                    onChange={(e) => setConfig({
                      ...config,
                      sample_definition: { ...config?.sample_definition, sampler: e.target.value }
                    })}
                  >
                    <option value="FLOW_MATCH">FlowMatch</option>
                    <option value="EULER">Euler</option>
                    <option value="EULER_A">Euler Ancestral</option>
                    <option value="DPM_2">DPM2</option>
                    <option value="DPM_2_A">DPM2 Ancestral</option>
                    <option value="HEUN">Heun</option>
                    <option value="LMS">LMS</option>
                    <option value="PNDM">PNDM</option>
                    <option value="DDIM">DDIM</option>
                    <option value="DDPM">DDPM</option>
                    <option value="UNIPC">UniPC</option>
                    <option value="DPMPP_2M">DPM++ 2M</option>
                    <option value="DPMPP_2M_SDE">DPM++ 2M SDE</option>
                    <option value="DPMPP_3M_SDE">DPM++ 3M SDE</option>
                  </select>
                  <button
                    type="button"
                    onClick={() => toggleSettings('sampler')}
                    className="px-2 py-1 bg-dark-bg border border-dark-border rounded hover:bg-dark-hover text-muted hover:text-white"
                    title="Sampler Settings"
                  >
                    <MoreHorizontal className="w-4 h-4" />
                  </button>
                </div>
              </div>
              {expandedSettings.has('sampler') && (
                <div className="ml-24 pl-2 border-l-2 border-dark-border space-y-2 py-2">
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <label className="text-xs text-muted block mb-1">CFG Scale</label>
                      <input
                        type="text"
                        value={config?.sample_definition?.cfg_scale || '3.5'}
                        onChange={(e) => setConfig({
                          ...config,
                          sample_definition: { ...config?.sample_definition, cfg_scale: e.target.value }
                        })}
                        className="input w-full text-sm"
                      />
                    </div>
                    <div>
                      <label className="text-xs text-muted block mb-1">Steps</label>
                      <input
                        type="text"
                        value={config?.sample_definition?.steps || 20}
                        onChange={(e) => setConfig({
                          ...config,
                          sample_definition: { ...config?.sample_definition, steps: parseInt(e.target.value) }
                        })}
                        className="input w-full text-sm"
                      />
                    </div>
                  </div>
                  <div>
                    <label className="text-xs text-muted block mb-1">Prompt</label>
                    <textarea
                      value={config?.sample_definition?.prompt || ''}
                      onChange={(e) => setConfig({
                        ...config,
                        sample_definition: { ...config?.sample_definition, prompt: e.target.value }
                      })}
                      className="input w-full text-sm h-16 resize-none"
                      placeholder="Sample prompt..."
                    />
                  </div>
                  <div>
                    <label className="text-xs text-muted block mb-1">Negative Prompt</label>
                    <textarea
                      value={config?.sample_definition?.negative_prompt || ''}
                      onChange={(e) => setConfig({
                        ...config,
                        sample_definition: { ...config?.sample_definition, negative_prompt: e.target.value }
                      })}
                      className="input w-full text-sm h-16 resize-none"
                      placeholder="Negative prompt..."
                    />
                  </div>
                </div>
              )}
            </div>
            <Field label="Height">
              <input
                type="text"
                value={config?.sample_definition?.height || 1024}
                onChange={(e) => setConfig({
                  ...config,
                  sample_definition: { ...config?.sample_definition, height: parseInt(e.target.value) }
                })}
                className="input w-full"
              />
            </Field>
            <Field label="Walk Seed">
              <Toggle
                defaultChecked={config?.sample_definition?.walk_seed || true}
                onChange={(checked) => setConfig({
                  ...config,
                  sample_definition: { ...config?.sample_definition, walk_seed: checked }
                })}
              />
            </Field>

            <Field label="Guidance Scale">
              <input
                type="text"
                value={config?.sample_definition?.guidance_scale || 4}
                onChange={(e) => setConfig({
                  ...config,
                  sample_definition: { ...config?.sample_definition, guidance_scale: parseFloat(e.target.value) }
                })}
                className="input w-full"
              />
            </Field>

            <Field label="Sample Steps">
              <input
                type="text"
                value={config?.sample_definition?.sample_steps || 25}
                onChange={(e) => setConfig({
                  ...config,
                  sample_definition: { ...config?.sample_definition, sample_steps: parseInt(e.target.value) }
                })}
                className="input w-full"
              />
            </Field>
          </div>

          {/* Sample Prompts */}
          <div className="p-4 border-t border-dark-border">
            <label className="text-sm text-muted block mb-2">Sample Prompts</label>
            <div className="space-y-3">
              <PromptRow prompt="woman with red hair, playing chess at the park" />
              <PromptRow prompt="a woman holding a coffee cup, in a beanie, sitting at a cafe" />
            </div>
          </div>
        </div>
      </div>

      {/* Save Preset Dialog */}
      {showSaveDialog && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-dark-surface border border-dark-border rounded-lg p-6 w-96 shadow-xl">
            <h3 className="text-lg font-medium text-white mb-4">Save Configuration</h3>
            <div className="mb-4">
              <label className="text-sm text-muted block mb-2">Preset Name</label>
              <input
                type="text"
                value={savePresetName}
                onChange={(e) => setSavePresetName(e.target.value)}
                className="input w-full"
                placeholder="Enter preset name..."
                autoFocus
                onKeyDown={(e) => {
                  if (e.key === 'Enter') handleConfirmSave();
                  if (e.key === 'Escape') setShowSaveDialog(false);
                }}
              />
            </div>
            <div className="flex justify-end gap-2">
              <button
                onClick={() => setShowSaveDialog(false)}
                className="px-4 py-2 text-sm text-muted hover:text-white"
                disabled={saving}
              >
                Cancel
              </button>
              <button
                onClick={handleConfirmSave}
                disabled={saving || !savePresetName.trim()}
                className="px-4 py-2 bg-primary hover:bg-primary-hover text-white rounded text-sm font-medium disabled:opacity-50"
              >
                {saving ? 'Saving...' : 'Save'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-dark-surface rounded-lg border border-dark-border">
      <div className="px-4 py-3 border-b border-dark-border">
        <h2 className="text-sm font-medium text-muted uppercase tracking-wider">{title}</h2>
      </div>
      <div className="p-4 space-y-4">{children}</div>
    </div>
  );
}

function Field({
  label,
  tooltip,
  children,
}: {
  label: string;
  tooltip?: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <label className="text-sm text-muted flex items-center gap-1 mb-2">
        {label}
        {tooltip && <HelpCircle className="w-3.5 h-3.5 text-muted/50" />}
      </label>
      {children}
    </div>
  );
}

function Toggle({ defaultChecked = false, onChange }: { defaultChecked?: boolean; onChange?: (checked: boolean) => void }) {
  const [checked, setChecked] = useState(defaultChecked);

  const handleClick = () => {
    const newValue = !checked;
    setChecked(newValue);
    onChange?.(newValue);
  };

  return (
    <button
      type="button"
      onClick={handleClick}
      className={`
        relative w-9 h-5 rounded-full transition-colors
        ${checked ? 'bg-green-600' : 'bg-gray-600'}
      `}
    >
      <span
        className={`
          absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform
          ${checked ? 'translate-x-4' : 'translate-x-0'}
        `}
      />
    </button>
  );
}

function PromptRow({ prompt }: { prompt: string }) {
  return (
    <div className="bg-dark-bg rounded-lg border border-dark-border p-3">
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1">
          <label className="text-xs text-muted block mb-1">Prompt</label>
          <input type="text" defaultValue={prompt} className="input w-full text-sm" />
        </div>
        <button className="text-muted hover:text-danger mt-5">
          <X className="w-4 h-4" />
        </button>
      </div>
      <div className="grid grid-cols-4 gap-3 mt-3">
        <Field label="Width">
          <input type="text" defaultValue="1024 (default)" className="input w-full text-sm" />
        </Field>
        <Field label="Height">
          <input type="text" defaultValue="1024 (default)" className="input w-full text-sm" />
        </Field>
        <Field label="Seed">
          <input type="text" defaultValue="42 (default)" className="input w-full text-sm" />
        </Field>
        <Field label="LoRA Scale">
          <input type="text" defaultValue="1.0 (default)" className="input w-full text-sm" />
        </Field>
      </div>
    </div>
  );
}
