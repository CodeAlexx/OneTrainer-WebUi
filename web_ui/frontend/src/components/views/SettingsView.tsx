import { useEffect, useState } from 'react';

interface Settings {
  output_dir: string;
  cache_dir: string;
  default_learning_rate: string;
  default_steps: number;
  default_optimizer: string;
  dark_mode: boolean;
  show_advanced: boolean;
}

const DEFAULT_SETTINGS: Settings = {
  output_dir: '/home/alex/ai-toolkit/output',
  cache_dir: '/home/alex/.cache/onetrainer',
  default_learning_rate: '0.0001',
  default_steps: 3000,
  default_optimizer: 'AdamW8Bit',
  dark_mode: true,
  show_advanced: false,
};

export function SettingsView() {
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [showSuccess, setShowSuccess] = useState(false);

  // Load settings from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem('onetrainer_settings');
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        setSettings({ ...DEFAULT_SETTINGS, ...parsed });
      } catch (e) {
        console.error('Failed to parse saved settings:', e);
      }
    }
  }, []);

  const handleSave = () => {
    localStorage.setItem('onetrainer_settings', JSON.stringify(settings));
    setShowSuccess(true);
    setTimeout(() => setShowSuccess(false), 3000);
  };

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="h-14 flex items-center px-6 border-b border-dark-border bg-dark-surface">
        <h1 className="text-lg font-medium text-white">Settings</h1>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-6">
        <div className="max-w-2xl space-y-6">
          {/* Success Message */}
          {showSuccess && (
            <div className="bg-green-500/10 border border-green-500/20 rounded-lg p-4">
              <div className="text-green-500 text-sm">Settings saved successfully!</div>
            </div>
          )}

          {/* General Settings */}
          <div className="bg-dark-surface rounded-lg border border-dark-border">
            <div className="px-4 py-3 border-b border-dark-border">
              <h2 className="text-sm font-medium text-muted uppercase tracking-wider">General</h2>
            </div>
            <div className="p-4 space-y-4">
              <div>
                <label className="text-sm text-muted block mb-2">Output Directory</label>
                <input
                  type="text"
                  value={settings.output_dir}
                  onChange={(e) => setSettings({ ...settings, output_dir: e.target.value })}
                  className="input w-full"
                />
              </div>
              <div>
                <label className="text-sm text-muted block mb-2">Cache Directory</label>
                <input
                  type="text"
                  value={settings.cache_dir}
                  onChange={(e) => setSettings({ ...settings, cache_dir: e.target.value })}
                  className="input w-full"
                />
              </div>
            </div>
          </div>

          {/* Training Defaults */}
          <div className="bg-dark-surface rounded-lg border border-dark-border">
            <div className="px-4 py-3 border-b border-dark-border">
              <h2 className="text-sm font-medium text-muted uppercase tracking-wider">Training Defaults</h2>
            </div>
            <div className="p-4 space-y-4">
              <div>
                <label className="text-sm text-muted block mb-2">Default Learning Rate</label>
                <input
                  type="text"
                  value={settings.default_learning_rate}
                  onChange={(e) => setSettings({ ...settings, default_learning_rate: e.target.value })}
                  className="input w-full"
                />
              </div>
              <div>
                <label className="text-sm text-muted block mb-2">Default Steps</label>
                <input
                  type="text"
                  value={settings.default_steps}
                  onChange={(e) => setSettings({ ...settings, default_steps: parseInt(e.target.value) || 0 })}
                  className="input w-full"
                />
              </div>
              <div>
                <label className="text-sm text-muted block mb-2">Default Optimizer</label>
                <select
                  value={settings.default_optimizer}
                  onChange={(e) => setSettings({ ...settings, default_optimizer: e.target.value })}
                  className="input w-full"
                >
                  <option>AdamW8Bit</option>
                  <option>Prodigy</option>
                  <option>AdaFactor</option>
                </select>
              </div>
            </div>
          </div>

          {/* Interface */}
          <div className="bg-dark-surface rounded-lg border border-dark-border">
            <div className="px-4 py-3 border-b border-dark-border">
              <h2 className="text-sm font-medium text-muted uppercase tracking-wider">Interface</h2>
            </div>
            <div className="p-4 space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-sm text-white">Dark Mode</div>
                  <div className="text-xs text-muted">Use dark theme (always on)</div>
                </div>
                <button
                  onClick={() => setSettings({ ...settings, dark_mode: !settings.dark_mode })}
                  className={`relative w-9 h-5 rounded-full transition-colors ${
                    settings.dark_mode ? 'bg-green-600' : 'bg-gray-600'
                  }`}
                >
                  <span
                    className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform ${
                      settings.dark_mode ? 'translate-x-4' : 'translate-x-0'
                    }`}
                  />
                </button>
              </div>
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-sm text-white">Show Advanced Options</div>
                  <div className="text-xs text-muted">Display advanced training options by default</div>
                </div>
                <button
                  onClick={() => setSettings({ ...settings, show_advanced: !settings.show_advanced })}
                  className={`relative w-9 h-5 rounded-full transition-colors ${
                    settings.show_advanced ? 'bg-green-600' : 'bg-gray-600'
                  }`}
                >
                  <span
                    className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform ${
                      settings.show_advanced ? 'translate-x-4' : 'translate-x-0'
                    }`}
                  />
                </button>
              </div>
            </div>
          </div>

          {/* About */}
          <div className="bg-dark-surface rounded-lg border border-dark-border">
            <div className="px-4 py-3 border-b border-dark-border">
              <h2 className="text-sm font-medium text-muted uppercase tracking-wider">About</h2>
            </div>
            <div className="p-4 space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-muted">Version</span>
                <span className="text-white">1.0.0 (Web UI)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted">OneTrainer</span>
                <a href="https://github.com/Nerogar/OneTrainer" className="text-primary hover:text-primary-light">
                  github.com/Nerogar/OneTrainer
                </a>
              </div>
            </div>
          </div>

          {/* Save Button */}
          <div className="flex justify-end">
            <button
              onClick={handleSave}
              className="px-6 py-2 bg-primary hover:bg-primary-light text-white rounded-md transition-colors font-medium"
            >
              Save Settings
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
