import { useState, useEffect } from 'react';
import { Plus, X, Copy, Shuffle, FileText, Upload } from 'lucide-react';
import { useConfigStore } from '../../stores/configStore';
import { databaseApi, DbSample } from '../../lib/api';
import { useDatabase } from '../../hooks/useDatabase';
const NOISE_SCHEDULERS = [
  'DDIM', 'EULER', 'EULER_A', 'DPMPP', 'DPMPP_SDE', 'UNIPC',
  'EULER_KARRAS', 'DPMPP_KARRAS', 'DPMPP_SDE_KARRAS', 'UNIPC_KARRAS'
];

const IMAGE_FORMATS = ['JPG', 'PNG'];
const TIME_UNITS = ['NEVER', 'EPOCH', 'STEP', 'SECOND', 'MINUTE', 'HOUR'];

interface Sample {
  id: string;
  enabled: boolean;
  prompt: string;
  negative_prompt: string;
  width: number;
  height: number;
  seed: number;
  random_seed: boolean;
  diffusion_steps: number;
  cfg_scale: number;
  noise_scheduler: string;
}

const createDefaultSample = (): Sample => ({
  id: `sample-${Date.now()}-${Math.random()}`,
  enabled: true,
  prompt: '',
  negative_prompt: '',
  width: 512,
  height: 512,
  seed: 42,
  random_seed: false,
  diffusion_steps: 20,
  cfg_scale: 7.0,
  noise_scheduler: 'DDIM',
});

export function SamplingView() {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [selectedSample, setSelectedSample] = useState<Sample | null>(null);
  const [samplesFilePath, setSamplesFilePath] = useState<string>('');
  const { config, updateConfig } = useConfigStore();

  // Database status
  const { dbEnabled } = useDatabase();

  // Import prompts dialog state
  const [showImportDialog, setShowImportDialog] = useState(false);
  const [importFilePath, setImportFilePath] = useState('/home/alex/OneTrainer/lady_prompts.txt');
  const [importSettings, setImportSettings] = useState({
    width: 1024,
    height: 1024,
    diffusion_steps: 20,
    cfg_scale: 3.5,
    noise_scheduler: 'EULER',
  });
  const [importPreview, setImportPreview] = useState<string[]>([]);
  const [importError, setImportError] = useState<string | null>(null);

  // Save samples to file when they change
  const saveSamplesToFile = async (samplesToSave: Sample[]) => {
    const filePath = samplesFilePath || (config as any)?.sample_definition_file_name;
    if (!filePath) return;

    try {
      // Convert Sample to file format
      const samplesData = samplesToSave.map(s => ({
        enabled: s.enabled,
        prompt: s.prompt,
        negative_prompt: s.negative_prompt,
        width: s.width,
        height: s.height,
        seed: s.seed,
        random_seed: s.random_seed,
        diffusion_steps: s.diffusion_steps,
        cfg_scale: s.cfg_scale,
        noise_scheduler: s.noise_scheduler,
        // The following fields are part of the original file format but not directly in our Sample interface
        // They are included here to maintain compatibility with the file structure if needed,
        // but are not explicitly mapped from the Sample object.
        __version: 0, // Assuming a default version
        frames: 1,
        length: 10.0,
        text_encoder_1_layer_skip: 0,
        text_encoder_2_layer_skip: 0,
        text_encoder_2_sequence_length: null,
        text_encoder_3_layer_skip: 0,
        text_encoder_4_layer_skip: 0,
        transformer_attention_mask: false,
        force_last_timestep: false,
        sample_inpainting: false,
        base_image_path: '',
        mask_image_path: '',
      }));

      await fetch(`/api/config/samples-file?file_path=${encodeURIComponent(filePath)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ samples: samplesData }),
      });
    } catch (err) {
      console.error('Failed to save samples to file:', err);
    }
  };

  // Load samples from database, config, or file - only on initial mount
  const [hasLoadedSamples, setHasLoadedSamples] = useState(false);

  useEffect(() => {
    // If we have DB samples already, don't reload
    if (samples.length > 0 && samples[0]?.id.startsWith('db-')) return;
    // If DB is enabled, always try to reload from DB
    // If already loaded from non-DB source and DB is NOT enabled, skip
    if (hasLoadedSamples && !dbEnabled) return;

    const loadSamples = async () => {
      // Try database first if enabled
      if (dbEnabled) {
        try {
          const res = await databaseApi.listSamples();
          if (res.data && res.data.length > 0) {
            const dbSamples: Sample[] = res.data.map((ds: DbSample) => ({
              id: `db-${ds.id}`,
              enabled: ds.enabled,
              prompt: ds.prompt,
              negative_prompt: ds.negative_prompt || '',
              width: ds.width,
              height: ds.height,
              seed: ds.seed,
              random_seed: ds.config?.random_seed || false,
              diffusion_steps: ds.config?.diffusion_steps || 20,
              cfg_scale: ds.config?.cfg_scale || 7.0,
              noise_scheduler: ds.config?.noise_scheduler || 'DDIM',
            }));
            setSamples(dbSamples);
            setHasLoadedSamples(true);
            if (dbSamples.length > 0) setSelectedSample(dbSamples[0]);
            return;
          }
        } catch (err) {
          console.error('Failed to load samples from database:', err);
          // Fall through to other methods
        }
      }

      const c = config as any;

      // Try loading from sample_definition_file_name if available
      if (c?.sample_definition_file_name) {
        setSamplesFilePath(c.sample_definition_file_name);
        try {
          const response = await fetch(`/api/config/samples-file?file_path=${encodeURIComponent(c.sample_definition_file_name)}`);
          const data = await response.json();
          const fileSamples = data.samples;
          if (Array.isArray(fileSamples) && fileSamples.length > 0) {
            const loadedSamples: Sample[] = fileSamples.map((fs: any, index: number) => ({
              id: `sample-${index}-${Date.now()}`,
              enabled: fs.enabled !== false,
              prompt: fs.prompt || '',
              negative_prompt: fs.negative_prompt || '',
              width: fs.width || 512,
              height: fs.height || 512,
              seed: fs.seed || 42,
              random_seed: fs.random_seed || false,
              diffusion_steps: fs.diffusion_steps || 20,
              cfg_scale: fs.cfg_scale || 7.0,
              noise_scheduler: fs.noise_scheduler || 'DDIM',
            }));
            setSamples(loadedSamples);
            setSelectedSample(loadedSamples[0]);
            setHasLoadedSamples(true);
            return;
          }
        } catch (err) {
          console.error('Failed to load samples from file:', err);
        }
      }

      // Try inline samples array from config
      if (c?.samples && Array.isArray(c.samples) && c.samples.length > 0) {
        const loadedSamples: Sample[] = c.samples.map((fs: any, index: number) => ({
          id: `sample-${index}-${Date.now()}`,
          enabled: fs.enabled !== false,
          prompt: fs.prompt || '',
          negative_prompt: fs.negative_prompt || '',
          width: fs.width || 512,
          height: fs.height || 512,
          seed: fs.seed || 42,
          random_seed: fs.random_seed || false,
          diffusion_steps: fs.diffusion_steps || 20,
          cfg_scale: fs.cfg_scale || 7.0,
          noise_scheduler: fs.noise_scheduler || 'DDIM',
        }));
        setSamples(loadedSamples);
        setSelectedSample(loadedSamples[0]);
        setHasLoadedSamples(true);
        return;
      }

      // Fallback to localStorage
      const saved = localStorage.getItem('onetrainer_samples');
      if (saved) {
        try {
          const parsed = JSON.parse(saved);
          if (Array.isArray(parsed) && parsed.length > 0) {
            setSamples(parsed);
            setSelectedSample(parsed[0]);
          }
        } catch (e) {
          console.error('Failed to parse saved samples:', e);
        }
      }

      setHasLoadedSamples(true);
    };

    if (config) {
      loadSamples();
    }
  }, [config, hasLoadedSamples, dbEnabled]);

  // Auto-save samples when they change
  useEffect(() => {
    // Don't save empty array on initial load - wait for samples to be loaded
    if (!hasLoadedSamples) {
      return;
    }

    localStorage.setItem('onetrainer_samples', JSON.stringify(samples));
    // Also save to file if we have a file path
    if (samplesFilePath && samples.length > 0) {
      saveSamplesToFile(samples);
    }
  }, [samples, samplesFilePath, hasLoadedSamples]);

  const handleAddSample = () => {
    const newSample = createDefaultSample();
    setSamples([...samples, newSample]);
    setSelectedSample(newSample);
  };

  const handleCloneSample = (sample: Sample) => {
    const cloned = { ...sample, id: `sample-${Date.now()}`, seed: Math.floor(Math.random() * 2147483647) };
    setSamples([...samples, cloned]);
  };

  const handleRemoveSample = (id: string) => {
    setSamples(samples.filter((s) => s.id !== id));
    if (selectedSample?.id === id) setSelectedSample(null);
  };

  const handleToggleEnabled = (id: string) => {
    setSamples(samples.map((s) => s.id === id ? { ...s, enabled: !s.enabled } : s));
  };

  const handleUpdateSample = (id: string, updates: Partial<Sample>) => {
    setSamples(samples.map((s) => s.id === id ? { ...s, ...updates } : s));
    if (selectedSample?.id === id) {
      setSelectedSample({ ...selectedSample, ...updates });
    }
  };

  const randomizeSeed = (id: string) => {
    handleUpdateSample(id, { seed: Math.floor(Math.random() * 2147483647) });
  };

  // Sample timing from config
  const sampleAfter = (config as any)?.sample_after ?? 100;
  const sampleAfterUnit = (config as any)?.sample_after_unit ?? 'STEP';
  const sampleSkipFirst = (config as any)?.sample_skip_first ?? 0;

  const handleTimingChange = (field: string, value: any) => {
    updateConfig({ [field]: value } as any);
  };

  // Additional sample config from TrainConfig
  const sampleImageFormat = (config as any)?.sample_image_format ?? 'JPG';
  const samplesToTensorboard = (config as any)?.samples_to_tensorboard ?? true;
  const nonEmaSampling = (config as any)?.non_ema_sampling ?? true;

  // Action handlers
  const handleSampleNow = async () => {
    try {
      await fetch('/api/training/sample', { method: 'POST' });
    } catch (err) {
      console.error('Failed to trigger sample:', err);
    }
  };

  const handleManualSample = async () => {
    // For manual sample, we might want to open a dialog or use selected sample config
    if (selectedSample) {
      try {
        await fetch('/api/training/sample', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ sample_config: selectedSample }),
        });
      } catch (err) {
        console.error('Failed to trigger manual sample:', err);
      }
    }
  };

  // Import prompts from file
  const handleLoadImportFile = async () => {
    if (!importFilePath.trim()) {
      setImportError('Please enter a file path');
      return;
    }
    setImportError(null);
    try {
      const response = await fetch(`/api/filesystem/read?path=${encodeURIComponent(importFilePath)}`);
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || 'Failed to read file');
      }
      const data = await response.json();
      // Split by double newlines or single newlines (filter empty lines)
      const lines = data.content.split(/\n\s*\n|\n/).filter((line: string) => line.trim().length > 0);
      setImportPreview(lines);
    } catch (err: any) {
      setImportError(err.message || 'Failed to load file');
      setImportPreview([]);
    }
  };

  const handleImportPrompts = () => {
    if (importPreview.length === 0) {
      setImportError('No prompts to import. Load a file first.');
      return;
    }

    const newSamples: Sample[] = importPreview.map((prompt, index) => ({
      id: `sample-${Date.now()}-${index}`,
      enabled: true,
      prompt: prompt.trim(),
      negative_prompt: '',
      width: importSettings.width,
      height: importSettings.height,
      seed: Math.floor(Math.random() * 2147483647),
      random_seed: true,
      diffusion_steps: importSettings.diffusion_steps,
      cfg_scale: importSettings.cfg_scale,
      noise_scheduler: importSettings.noise_scheduler,
    }));

    setSamples([...samples, ...newSamples]);
    setShowImportDialog(false);
    setImportPreview([]);
    if (newSamples.length > 0) {
      setSelectedSample(newSamples[0]);
    }
  };

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex flex-col border-b border-dark-border bg-dark-surface">
        <div className="flex items-center justify-between px-6 py-3">
          <h1 className="text-lg font-medium text-white">Sample Definitions</h1>
          <div className="flex items-center gap-3">
            {/* Action Buttons */}
            <button
              onClick={handleSampleNow}
              className="bg-cyan-600 hover:bg-cyan-500 text-white px-4 py-1.5 rounded-lg text-sm font-medium"
            >
              sample now
            </button>
            <button
              onClick={handleManualSample}
              className="bg-cyan-600 hover:bg-cyan-500 text-white px-4 py-1.5 rounded-lg text-sm font-medium"
            >
              manual sample
            </button>
            <button
              onClick={handleAddSample}
              className="bg-dark-border hover:bg-dark-hover text-white px-4 py-1.5 rounded-lg text-sm font-medium flex items-center gap-2"
            >
              <Plus className="w-4 h-4" />
              Add Sample
            </button>
            <button
              onClick={() => setShowImportDialog(true)}
              className="bg-primary hover:bg-primary-hover text-white px-4 py-1.5 rounded-lg text-sm font-medium flex items-center gap-2"
            >
              <FileText className="w-4 h-4" />
              Import Prompts
            </button>
          </div>
        </div>

        {/* Sample Settings Row */}
        <div className="flex items-center gap-6 px-6 py-2 border-t border-dark-border/50 bg-dark-bg/50">
          {/* Sample After */}
          <div className="flex items-center gap-2 text-sm">
            <span className="text-muted">Sample After</span>
            <input
              type="text"
              value={sampleAfter}
              onChange={(e) => handleTimingChange('sample_after', parseInt(e.target.value) || 10)}
              className="input w-16 text-sm"
              min="1"
            />
            <select
              value={sampleAfterUnit}
              onChange={(e) => handleTimingChange('sample_after_unit', e.target.value)}
              className="input text-sm"
            >
              {TIME_UNITS.map(unit => (
                <option key={unit} value={unit}>{unit}</option>
              ))}
            </select>
          </div>

          {/* Skip First */}
          <div className="flex items-center gap-2 text-sm">
            <span className="text-muted">Skip First</span>
            <input
              type="text"
              value={sampleSkipFirst}
              onChange={(e) => handleTimingChange('sample_skip_first', parseInt(e.target.value) || 0)}
              className="input w-16 text-sm"
              min="0"
            />
          </div>

          {/* Format */}
          <div className="flex items-center gap-2 text-sm">
            <span className="text-muted">Format</span>
            <select
              value={sampleImageFormat}
              onChange={(e) => handleTimingChange('sample_image_format', e.target.value)}
              className="input text-sm"
            >
              {IMAGE_FORMATS.map(fmt => (
                <option key={fmt} value={fmt}>{fmt}</option>
              ))}
            </select>
          </div>

          {/* Non-EMA Sampling Toggle */}
          <div className="flex items-center gap-2 text-sm">
            <span className="text-muted">Non-EMA Sampling</span>
            <button
              onClick={() => handleTimingChange('non_ema_sampling', !nonEmaSampling)}
              className={`w-9 h-5 rounded-full relative flex-shrink-0 ${nonEmaSampling ? 'bg-cyan-600' : 'bg-gray-600'}`}
            >
              <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform ${nonEmaSampling ? 'translate-x-4' : 'translate-x-0'}`} />
            </button>
          </div>

          {/* Samples to Tensorboard Toggle */}
          <div className="flex items-center gap-2 text-sm">
            <span className="text-muted">Samples to Tensorboard</span>
            <button
              onClick={() => handleTimingChange('samples_to_tensorboard', !samplesToTensorboard)}
              className={`w-9 h-5 rounded-full relative flex-shrink-0 ${samplesToTensorboard ? 'bg-cyan-600' : 'bg-gray-600'}`}
            >
              <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform ${samplesToTensorboard ? 'translate-x-4' : 'translate-x-0'}`} />
            </button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Samples List */}
        <div className="flex-1 overflow-auto">
          {samples.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-muted">
              <p className="text-sm mb-4">No sample definitions configured.</p>
              <button onClick={handleAddSample} className="px-4 py-2 bg-primary hover:bg-primary-hover text-white rounded-lg text-sm">
                Add First Sample
              </button>
            </div>
          ) : (
            <div className="divide-y divide-dark-border">
              {samples.map((sample) => (
                <div key={sample.id}
                  onClick={() => setSelectedSample(sample)}
                  className={`px-4 py-3 flex items-center gap-4 cursor-pointer hover:bg-dark-hover transition-colors
                    ${selectedSample?.id === sample.id ? 'bg-dark-hover' : ''}
                    ${!sample.enabled ? 'opacity-50' : ''}`}>

                  {/* Controls */}
                  <div className="flex items-center gap-2">
                    <button onClick={(e) => { e.stopPropagation(); handleRemoveSample(sample.id); }}
                      className="w-6 h-6 bg-red-600 hover:bg-red-500 text-white rounded text-xs flex items-center justify-center">
                      <X className="w-3 h-3" />
                    </button>
                    <button onClick={(e) => { e.stopPropagation(); handleCloneSample(sample); }}
                      className="w-6 h-6 bg-green-600 hover:bg-green-500 text-white rounded text-xs flex items-center justify-center">
                      <Copy className="w-3 h-3" />
                    </button>
                  </div>

                  {/* Enable Toggle */}
                  <button onClick={(e) => { e.stopPropagation(); handleToggleEnabled(sample.id); }}
                    className={`w-9 h-5 rounded-full relative flex-shrink-0 ${sample.enabled ? 'bg-green-600' : 'bg-gray-600'}`}>
                    <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform ${sample.enabled ? 'translate-x-4' : 'translate-x-0'}`} />
                  </button>

                  {/* Width/Height */}
                  <div className="flex items-center gap-1 text-xs text-muted w-24 flex-shrink-0">
                    <span>{sample.width}</span>
                    <span>×</span>
                    <span>{sample.height}</span>
                  </div>

                  {/* Seed */}
                  <div className="text-xs text-muted w-24 flex-shrink-0">
                    seed: {sample.seed}
                  </div>

                  {/* Prompt */}
                  <div className="flex-1 truncate text-sm text-white">
                    {sample.prompt || <span className="text-muted italic">No prompt</span>}
                  </div>

                  {/* Open Button */}
                  <button onClick={(e) => { e.stopPropagation(); setSelectedSample(sample); }}
                    className="px-3 py-1 text-xs bg-dark-border hover:bg-dark-hover text-white rounded">
                    ...
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Detail Panel */}
        {selectedSample && (
          <div className="w-96 border-l border-dark-border bg-dark-surface overflow-y-auto">
            <div className="p-4 border-b border-dark-border flex items-center justify-between">
              <h2 className="text-sm font-medium text-white">Sample Details</h2>
              <button onClick={() => setSelectedSample(null)} className="text-muted hover:text-white">
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="p-4 space-y-4">
              {/* Prompt */}
              <div>
                <label className="text-xs text-muted block mb-1">Prompt</label>
                <textarea value={selectedSample.prompt}
                  onChange={(e) => handleUpdateSample(selectedSample.id, { prompt: e.target.value })}
                  className="input w-full text-sm h-24 resize-none" placeholder="Enter prompt..." />
              </div>

              {/* Negative Prompt */}
              <div>
                <label className="text-xs text-muted block mb-1">Negative Prompt</label>
                <textarea value={selectedSample.negative_prompt}
                  onChange={(e) => handleUpdateSample(selectedSample.id, { negative_prompt: e.target.value })}
                  className="input w-full text-sm h-16 resize-none" placeholder="Enter negative prompt..." />
              </div>

              {/* Dimensions */}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-xs text-muted block mb-1">Width</label>
                  <input type="text" value={selectedSample.width}
                    onChange={(e) => handleUpdateSample(selectedSample.id, { width: parseInt(e.target.value) || 512 })}
                    className="input w-full text-sm" min="64" step="64" />
                </div>
                <div>
                  <label className="text-xs text-muted block mb-1">Height</label>
                  <input type="text" value={selectedSample.height}
                    onChange={(e) => handleUpdateSample(selectedSample.id, { height: parseInt(e.target.value) || 512 })}
                    className="input w-full text-sm" min="64" step="64" />
                </div>
              </div>
              {/* Resolution Presets */}
              <div>
                <label className="text-xs text-muted block mb-1">Resolution Presets</label>
                <div className="flex flex-wrap gap-1">
                  {[512, 768, 1024, 1536, 2048].map((size) => (
                    <button
                      key={size}
                      onClick={() => handleUpdateSample(selectedSample.id, { width: size, height: size })}
                      className={`px-2 py-1 text-xs rounded ${selectedSample.width === size && selectedSample.height === size ? 'bg-primary text-white' : 'bg-dark-border hover:bg-dark-hover text-white'}`}
                    >
                      {size}
                    </button>
                  ))}
                </div>
                <div className="flex flex-wrap gap-1 mt-1">
                  <button
                    onClick={() => handleUpdateSample(selectedSample.id, { width: 1024, height: 768 })}
                    className={`px-2 py-1 text-xs rounded ${selectedSample.width === 1024 && selectedSample.height === 768 ? 'bg-primary text-white' : 'bg-dark-border hover:bg-dark-hover text-white'}`}
                  >
                    1024×768
                  </button>
                  <button
                    onClick={() => handleUpdateSample(selectedSample.id, { width: 768, height: 1024 })}
                    className={`px-2 py-1 text-xs rounded ${selectedSample.width === 768 && selectedSample.height === 1024 ? 'bg-primary text-white' : 'bg-dark-border hover:bg-dark-hover text-white'}`}
                  >
                    768×1024
                  </button>
                  <button
                    onClick={() => handleUpdateSample(selectedSample.id, { width: 1536, height: 1024 })}
                    className={`px-2 py-1 text-xs rounded ${selectedSample.width === 1536 && selectedSample.height === 1024 ? 'bg-primary text-white' : 'bg-dark-border hover:bg-dark-hover text-white'}`}
                  >
                    1536×1024
                  </button>
                  <button
                    onClick={() => handleUpdateSample(selectedSample.id, { width: 1024, height: 1536 })}
                    className={`px-2 py-1 text-xs rounded ${selectedSample.width === 1024 && selectedSample.height === 1536 ? 'bg-primary text-white' : 'bg-dark-border hover:bg-dark-hover text-white'}`}
                  >
                    1024×1536
                  </button>
                </div>
              </div>

              {/* Seed */}
              <div>
                <label className="text-xs text-muted block mb-1">Seed</label>
                <div className="flex gap-2">
                  <input type="text" value={selectedSample.seed}
                    onChange={(e) => handleUpdateSample(selectedSample.id, { seed: e.target.value === '' ? 0 : parseInt(e.target.value) || selectedSample.seed })}
                    className="input flex-1 text-sm" />
                  <button onClick={() => randomizeSeed(selectedSample.id)}
                    className="p-2 bg-dark-border hover:bg-dark-hover rounded">
                    <Shuffle className="w-4 h-4 text-muted" />
                  </button>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <input type="checkbox" id="random_seed" checked={selectedSample.random_seed}
                  onChange={(e) => handleUpdateSample(selectedSample.id, { random_seed: e.target.checked })}
                  className="rounded" />
                <label htmlFor="random_seed" className="text-sm text-white">Random Seed</label>
              </div>

              {/* Sampling Parameters */}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-xs text-muted block mb-1">Diffusion Steps</label>
                  <input type="text" value={selectedSample.diffusion_steps}
                    onChange={(e) => handleUpdateSample(selectedSample.id, { diffusion_steps: parseInt(e.target.value) || 20 })}
                    className="input w-full text-sm" min="1" max="150" />
                </div>
                <div>
                  <label className="text-xs text-muted block mb-1">CFG Scale</label>
                  <input type="number" value={selectedSample.cfg_scale}
                    onChange={(e) => {
                      const val = e.target.value;
                      if (val === '' || val === '-' || val.endsWith('.')) {
                        // Allow intermediate states
                        handleUpdateSample(selectedSample.id, { cfg_scale: parseFloat(val) || selectedSample.cfg_scale });
                      } else {
                        handleUpdateSample(selectedSample.id, { cfg_scale: parseFloat(val) || 3.5 });
                      }
                    }}
                    className="input w-full text-sm" min="1" max="30" step="0.1" />
                </div>
              </div>

              <div>
                <label className="text-xs text-muted block mb-1">Noise Scheduler</label>
                <select value={selectedSample.noise_scheduler}
                  onChange={(e) => handleUpdateSample(selectedSample.id, { noise_scheduler: e.target.value })}
                  className="input w-full text-sm">
                  {NOISE_SCHEDULERS.map(s => <option key={s} value={s}>{s}</option>)}
                </select>
              </div>

              {/* Delete Button */}
              <button onClick={() => handleRemoveSample(selectedSample.id)}
                className="w-full py-2 bg-red-600/20 hover:bg-red-600/30 text-red-400 rounded text-sm flex items-center justify-center gap-2 mt-4">
                <X className="w-4 h-4" />
                Delete Sample
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Import Prompts Dialog */}
      {showImportDialog && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-8" onClick={() => setShowImportDialog(false)}>
          <div
            className="bg-dark-surface border border-dark-border rounded-xl shadow-2xl w-full max-w-3xl max-h-[85vh] flex flex-col"
            onClick={e => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-dark-border">
              <h2 className="text-lg font-semibold text-white">Import Prompts from File</h2>
              <button onClick={() => setShowImportDialog(false)} className="p-1 hover:bg-dark-hover rounded">
                <X className="w-5 h-5 text-muted" />
              </button>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-auto p-4 space-y-4">
              {/* File Path */}
              <div>
                <label className="text-sm text-muted block mb-1">Prompt File Path</label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={importFilePath}
                    onChange={(e) => setImportFilePath(e.target.value)}
                    placeholder="/path/to/prompts.txt"
                    className="input flex-1 text-sm"
                  />
                  <button
                    onClick={handleLoadImportFile}
                    className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded text-sm flex items-center gap-2"
                  >
                    <Upload className="w-4 h-4" />
                    Load
                  </button>
                </div>
                <p className="text-xs text-muted mt-1">Each prompt should be on its own line (blank lines are ignored)</p>
              </div>

              {/* Shared Settings */}
              <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                <div>
                  <label className="text-xs text-muted block mb-1">Width</label>
                  <input
                    type="number"
                    value={importSettings.width}
                    onChange={(e) => setImportSettings({ ...importSettings, width: parseInt(e.target.value) || 1024 })}
                    className="input w-full text-sm"
                  />
                </div>
                <div>
                  <label className="text-xs text-muted block mb-1">Height</label>
                  <input
                    type="number"
                    value={importSettings.height}
                    onChange={(e) => setImportSettings({ ...importSettings, height: parseInt(e.target.value) || 1024 })}
                    className="input w-full text-sm"
                  />
                </div>
                <div>
                  <label className="text-xs text-muted block mb-1">Steps</label>
                  <input
                    type="number"
                    value={importSettings.diffusion_steps}
                    onChange={(e) => setImportSettings({ ...importSettings, diffusion_steps: parseInt(e.target.value) || 20 })}
                    className="input w-full text-sm"
                  />
                </div>
                <div>
                  <label className="text-xs text-muted block mb-1">CFG</label>
                  <input
                    type="number"
                    value={importSettings.cfg_scale}
                    onChange={(e) => setImportSettings({ ...importSettings, cfg_scale: parseFloat(e.target.value) || 3.5 })}
                    className="input w-full text-sm"
                    step="0.1"
                  />
                </div>
                <div>
                  <label className="text-xs text-muted block mb-1">Scheduler</label>
                  <select
                    value={importSettings.noise_scheduler}
                    onChange={(e) => setImportSettings({ ...importSettings, noise_scheduler: e.target.value })}
                    className="input w-full text-sm"
                  >
                    {NOISE_SCHEDULERS.map(s => <option key={s} value={s}>{s}</option>)}
                  </select>
                </div>
              </div>

              {/* Resolution Presets */}
              <div>
                <label className="text-xs text-muted block mb-1">Resolution Presets</label>
                <div className="flex flex-wrap gap-1">
                  {[512, 768, 1024, 1536, 2048].map((size) => (
                    <button
                      key={size}
                      onClick={() => setImportSettings({ ...importSettings, width: size, height: size })}
                      className={`px-3 py-1 text-xs rounded ${importSettings.width === size && importSettings.height === size ? 'bg-primary text-white' : 'bg-dark-border hover:bg-dark-hover text-white'}`}
                    >
                      {size}×{size}
                    </button>
                  ))}
                  <button
                    onClick={() => setImportSettings({ ...importSettings, width: 1024, height: 768 })}
                    className={`px-3 py-1 text-xs rounded ${importSettings.width === 1024 && importSettings.height === 768 ? 'bg-primary text-white' : 'bg-dark-border hover:bg-dark-hover text-white'}`}
                  >
                    1024×768
                  </button>
                  <button
                    onClick={() => setImportSettings({ ...importSettings, width: 768, height: 1024 })}
                    className={`px-3 py-1 text-xs rounded ${importSettings.width === 768 && importSettings.height === 1024 ? 'bg-primary text-white' : 'bg-dark-border hover:bg-dark-hover text-white'}`}
                  >
                    768×1024
                  </button>
                </div>
              </div>

              {/* Error */}
              {importError && (
                <div className="p-3 bg-red-600/20 border border-red-600/50 rounded text-red-400 text-sm">
                  {importError}
                </div>
              )}

              {/* Preview */}
              {importPreview.length > 0 && (
                <div>
                  <label className="text-sm text-muted block mb-2">Preview ({importPreview.length} prompts)</label>
                  <div className="bg-dark-bg border border-dark-border rounded max-h-48 overflow-auto">
                    {importPreview.map((prompt, i) => (
                      <div key={i} className="px-3 py-2 text-sm text-white border-b border-dark-border/50 last:border-0">
                        <span className="text-muted mr-2">{i + 1}.</span>
                        <span className="truncate">{prompt.length > 100 ? prompt.substring(0, 100) + '...' : prompt}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Footer */}
            <div className="flex items-center justify-between p-4 border-t border-dark-border">
              <span className="text-sm text-muted">
                {importPreview.length > 0 ? `${importPreview.length} prompts ready to import` : 'Load a file to preview prompts'}
              </span>
              <div className="flex gap-2">
                <button
                  onClick={() => setShowImportDialog(false)}
                  className="px-4 py-2 text-sm text-muted hover:text-white"
                >
                  Cancel
                </button>
                <button
                  onClick={handleImportPrompts}
                  disabled={importPreview.length === 0}
                  className="px-4 py-2 bg-primary hover:bg-primary-hover text-white rounded text-sm disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Import {importPreview.length > 0 ? `${importPreview.length} Prompts` : 'Prompts'}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div >
  );
}
