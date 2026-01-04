import { useState, useEffect } from 'react';
import { useConfigStore } from '../../stores/configStore';

interface DataConfig {
  resolution: number;
  batch_size: number;
  gradient_accumulation_steps: number;
  dataloader_threads: number;
  latent_caching: boolean;
}

const DEFAULT_CONFIG: DataConfig = {
  resolution: 512,
  batch_size: 1,
  gradient_accumulation_steps: 1,
  dataloader_threads: 4,
  latent_caching: true,
};

export function DataView() {
  const { config: storeConfig, updateConfig } = useConfigStore();
  const [config, setConfigLocal] = useState<DataConfig>(DEFAULT_CONFIG);

  // Sync from store when it changes
  useEffect(() => {
    if (storeConfig && Object.keys(storeConfig).length > 0) {
      const c = storeConfig as any;
      setConfigLocal({
        resolution: c.resolution ?? DEFAULT_CONFIG.resolution,
        batch_size: c.batch_size ?? DEFAULT_CONFIG.batch_size,
        gradient_accumulation_steps: c.gradient_accumulation_steps ?? DEFAULT_CONFIG.gradient_accumulation_steps,
        dataloader_threads: c.dataloader_threads ?? DEFAULT_CONFIG.dataloader_threads,
        latent_caching: c.latent_caching ?? DEFAULT_CONFIG.latent_caching,
      });
    }
  }, [storeConfig]);

  const setConfig = (newConfig: DataConfig) => {
    setConfigLocal(newConfig);
    updateConfig(newConfig);
  };

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="h-14 flex items-center px-6 border-b border-dark-border bg-dark-surface">
        <h1 className="text-lg font-medium text-white">Data Configuration</h1>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-6">
        <div className="max-w-2xl space-y-6">
          {/* Dataset Settings */}
          <div className="bg-dark-surface rounded-lg border border-dark-border">
            <div className="px-4 py-3 border-b border-dark-border">
              <h2 className="text-sm font-medium text-muted uppercase tracking-wider">Dataset Settings</h2>
            </div>
            <div className="p-4 space-y-4">
              <div>
                <label className="text-sm text-muted block mb-2">Resolution</label>
                <input
                  type="text"
                  value={config.resolution}
                  onChange={(e) => setConfig({ ...config, resolution: parseInt(e.target.value) || 0 })}
                  className="input w-full"
                  min="64"
                  step="64"
                />
                <p className="text-xs text-muted mt-1">Image resolution for training (typically 512, 768, or 1024)</p>
              </div>

              <div>
                <label className="text-sm text-muted block mb-2">Batch Size</label>
                <input
                  type="text"
                  value={config.batch_size}
                  onChange={(e) => setConfig({ ...config, batch_size: parseInt(e.target.value) || 0 })}
                  className="input w-full"
                  min="1"
                />
                <p className="text-xs text-muted mt-1">Number of samples per training batch</p>
              </div>

              <div>
                <label className="text-sm text-muted block mb-2">Gradient Accumulation Steps</label>
                <input
                  type="text"
                  value={config.gradient_accumulation_steps}
                  onChange={(e) => setConfig({ ...config, gradient_accumulation_steps: parseInt(e.target.value) || 0 })}
                  className="input w-full"
                  min="1"
                />
                <p className="text-xs text-muted mt-1">Number of steps to accumulate gradients before updating weights</p>
              </div>

              <div>
                <label className="text-sm text-muted block mb-2">Dataloader Threads</label>
                <input
                  type="text"
                  value={config.dataloader_threads}
                  onChange={(e) => setConfig({ ...config, dataloader_threads: parseInt(e.target.value) || 0 })}
                  className="input w-full"
                  min="1"
                />
                <p className="text-xs text-muted mt-1">Number of worker threads for data loading</p>
              </div>
            </div>
          </div>

          {/* Performance Options */}
          <div className="bg-dark-surface rounded-lg border border-dark-border">
            <div className="px-4 py-3 border-b border-dark-border">
              <h2 className="text-sm font-medium text-muted uppercase tracking-wider">Performance Options</h2>
            </div>
            <div className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-sm text-white">Latent Caching</div>
                  <div className="text-xs text-muted">Cache VAE-encoded latents to speed up training</div>
                </div>
                <button
                  onClick={() => setConfig({ ...config, latent_caching: !config.latent_caching })}
                  className={`relative w-9 h-5 rounded-full transition-colors ${
                    config.latent_caching ? 'bg-green-600' : 'bg-gray-600'
                  }`}
                >
                  <span
                    className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform ${
                      config.latent_caching ? 'translate-x-4' : 'translate-x-0'
                    }`}
                  />
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
