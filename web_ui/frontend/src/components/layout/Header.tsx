import { useState, useEffect, useRef } from 'react';
import { Download, Upload, Save, ChevronDown, Check, FolderOpen, Database, Grid } from 'lucide-react';
import { useConfigStore } from '../../stores/configStore';
import { MODEL_TYPES, getTrainingMethods } from '../../model_constants';
import { databaseApi, DbPreset } from '../../lib/api';
import { useDatabase } from '../../hooks/useDatabase';
import { PresetCardSelector } from '../PresetCardSelector';

interface PresetInfo {
  name: string;
  path: string;
  last_modified: string;
  id?: number; // Database presets have an ID
  isDbPreset?: boolean;
}

export function Header() {
  const { config, setConfig, updateConfig, currentPreset, setCurrentPreset } = useConfigStore();
  const { dbEnabled } = useDatabase();
  const [presets, setPresets] = useState<PresetInfo[]>([]);
  const [showPresetDropdown, setShowPresetDropdown] = useState(false);
  const [showCardSelector, setShowCardSelector] = useState(false);
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [savePresetName, setSavePresetName] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Fetch presets on mount and when database status changes
  useEffect(() => {
    fetchPresets();
  }, [dbEnabled]);

  // Close dropdown on outside click
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setShowPresetDropdown(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Clear message after 3 seconds
  useEffect(() => {
    if (message) {
      const timer = setTimeout(() => setMessage(null), 3000);
      return () => clearTimeout(timer);
    }
  }, [message]);

  const fetchPresets = async () => {
    try {
      // Try database first if enabled
      if (dbEnabled) {
        const dbRes = await databaseApi.listPresets();
        const dbPresets: PresetInfo[] = dbRes.data.map((p: DbPreset) => ({
          name: p.name,
          path: '',
          last_modified: p.updated_at,
          id: p.id,
          isDbPreset: true,
        }));
        setPresets(dbPresets);
        return;
      }

      // Fall back to JSON presets
      const response = await fetch('/api/config/presets');
      if (response.ok) {
        const data = await response.json();
        setPresets((data.presets || []).map((p: any) => ({ ...p, isDbPreset: false })));
      }
    } catch (error) {
      console.error('Failed to fetch presets:', error);
    }
  };

  const loadPreset = async (preset: PresetInfo) => {
    setIsLoading(true);
    try {
      let loadedConfig: any;

      // Load from database if it's a database preset
      if (preset.isDbPreset && preset.id) {
        const dbRes = await databaseApi.getPreset(preset.id);
        loadedConfig = dbRes.data.config;
      } else {
        // Load from JSON file
        const response = await fetch(`/api/config/presets/${encodeURIComponent(preset.name)}`);
        if (!response.ok) {
          const error = await response.json();
          throw new Error(error.detail || 'Failed to load preset');
        }
        const data = await response.json();
        loadedConfig = data.config;
      }

      setConfig(loadedConfig);
      setCurrentPreset(preset.name);
      setMessage({ type: 'success', text: `Loaded preset: ${preset.name}` });
    } catch (error: any) {
      setMessage({ type: 'error', text: error.message || 'Failed to load preset' });
    } finally {
      setIsLoading(false);
      setShowPresetDropdown(false);
    }
  };

  const savePreset = async (name: string) => {
    if (!name.trim()) {
      setMessage({ type: 'error', text: 'Please enter a preset name' });
      return;
    }

    setIsLoading(true);
    try {
      const response = await fetch(`/api/config/presets/${encodeURIComponent(name)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config }),
      });

      if (response.ok) {
        setCurrentPreset(name);
        setMessage({ type: 'success', text: `Saved preset: ${name}` });
        setShowSaveDialog(false);
        setSavePresetName('');
        fetchPresets(); // Refresh list
      } else {
        const error = await response.json();
        setMessage({ type: 'error', text: error.detail || 'Failed to save preset' });
      }
    } catch (error) {
      setMessage({ type: 'error', text: 'Failed to save preset' });
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileLoad = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;

      try {
        const text = await file.text();
        const configData = JSON.parse(text);
        setConfig(configData);
        setCurrentPreset(file.name.replace('.json', ''));
        setMessage({ type: 'success', text: `Loaded: ${file.name}` });
      } catch (error) {
        setMessage({ type: 'error', text: 'Invalid JSON file' });
      }
    };
    input.click();
  };

  const handleFileSave = () => {
    const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${currentPreset || 'config'}.json`;
    a.click();
    URL.revokeObjectURL(url);
    setMessage({ type: 'success', text: 'Downloaded config file' });
  };

  return (
    <header className="h-14 border-b border-dark-border bg-dark-surface px-6 flex items-center justify-between">
      <div className="flex items-center gap-4">
        <h1 className="text-xl font-bold text-primary">OneTrainer</h1>

        {/* Preset Selector */}
        <div className="flex items-center gap-1">
          <div className="relative" ref={dropdownRef}>
            <button
              onClick={() => setShowPresetDropdown(!showPresetDropdown)}
              className="flex items-center gap-2 px-3 py-1.5 bg-dark-bg border border-dark-border rounded hover:border-primary/50 text-sm min-w-[200px]"
            >
              <FolderOpen className="w-4 h-4 text-muted" />
              <span className="flex-1 text-left truncate">
                {currentPreset || 'Select preset...'}
              </span>
              <ChevronDown className="w-4 h-4 text-muted" />
            </button>

            {showPresetDropdown && (
              <div className="absolute top-full left-0 mt-1 w-72 bg-dark-surface border border-dark-border rounded-lg shadow-xl z-50 max-h-80 overflow-auto">
                {presets.length === 0 ? (
                  <div className="px-3 py-4 text-sm text-muted text-center">
                    {dbEnabled ? 'No presets in database. Migrate from JSON first.' : 'No presets found in training_presets/'}
                  </div>
                ) : (
                  presets.map((preset) => (
                    <button
                      key={preset.id || preset.name}
                      onClick={() => loadPreset(preset)}
                      className="w-full px-3 py-2 text-left hover:bg-dark-hover flex items-center gap-2 text-sm"
                    >
                      {preset.isDbPreset && <Database className="w-3 h-3 text-cyan-400" />}
                      {currentPreset === preset.name && (
                        <Check className="w-4 h-4 text-primary" />
                      )}
                      <span className={currentPreset === preset.name ? 'text-primary' : 'text-white'}>
                        {preset.name}
                      </span>
                    </button>
                  ))
                )}
              </div>
            )}
          </div>

          {/* Card View Button */}
          <button
            onClick={() => setShowCardSelector(true)}
            className="p-1.5 bg-dark-bg border border-dark-border rounded hover:border-primary/50"
            title="Open Card View"
          >
            <Grid className="w-4 h-4 text-muted" />
          </button>
        </div>

        {/* Status Message */}
        {message && (
          <span className={`text-sm ${message.type === 'success' ? 'text-green-400' : 'text-red-400'}`}>
            {message.text}
          </span>
        )}
      </div>

      <div className="flex items-center gap-2">
        {/* Model Type Selector */}
        <select
          value={(config as any)?.model_type || 'Z_IMAGE'}
          onChange={(e) => {
            const newType = e.target.value;
            const methods = getTrainingMethods(newType);
            // Default to first method if current is invalid
            const currentMethod = (config as any)?.training_method;
            const valid = methods.find(m => m.value === currentMethod);
            const newMethod = valid ? currentMethod : methods[0].value;

            updateConfig({ model_type: newType, training_method: newMethod });
          }}
          className="bg-dark-bg border border-dark-border rounded px-3 py-1.5 text-sm text-white max-w-[140px] truncate"
          title="Model Type"
        >
          {MODEL_TYPES.map(t => (
            <option key={t.value} value={t.value}>{t.label}</option>
          ))}
        </select>

        {/* Training Method Selector */}
        <select
          value={(config as any)?.training_method || 'LORA'}
          onChange={(e) => updateConfig({ training_method: e.target.value })}
          className="bg-dark-bg border border-dark-border rounded px-3 py-1.5 text-sm text-white max-w-[120px] truncate"
          title="Training Method"
        >
          {getTrainingMethods((config as any)?.model_type || 'Z_IMAGE').map(m => (
            <option key={m.value} value={m.value}>{m.label}</option>
          ))}
        </select>

        {/* Load from file */}
        <button
          onClick={handleFileLoad}
          disabled={isLoading}
          className="flex items-center gap-2 px-3 py-1.5 text-sm text-muted hover:text-white hover:bg-dark-hover rounded disabled:opacity-50"
        >
          <Upload className="h-4 w-4" />
          Load File
        </button>

        {/* Save to file */}
        <button
          onClick={handleFileSave}
          disabled={isLoading || !config}
          className="flex items-center gap-2 px-3 py-1.5 text-sm text-muted hover:text-white hover:bg-dark-hover rounded disabled:opacity-50"
        >
          <Download className="h-4 w-4" />
          Export
        </button>

        {/* Save Preset */}
        <button
          onClick={() => {
            setSavePresetName(currentPreset || '');
            setShowSaveDialog(true);
          }}
          disabled={isLoading || !config}
          className="flex items-center gap-2 px-3 py-1.5 text-sm bg-primary hover:bg-primary-hover text-white rounded disabled:opacity-50"
        >
          <Save className="h-4 w-4" />
          Save Preset
        </button>
      </div>

      {/* Save Dialog */}
      {showSaveDialog && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-dark-surface border border-dark-border rounded-lg p-6 w-96">
            <h2 className="text-lg font-medium text-white mb-4">Save Preset</h2>
            <input
              type="text"
              value={savePresetName}
              onChange={(e) => setSavePresetName(e.target.value)}
              placeholder="Preset name"
              className="w-full px-3 py-2 bg-dark-bg border border-dark-border rounded text-white mb-4"
              autoFocus
              onKeyDown={(e) => {
                if (e.key === 'Enter') savePreset(savePresetName);
                if (e.key === 'Escape') setShowSaveDialog(false);
              }}
            />
            <p className="text-xs text-muted mb-4">
              Will be saved to: training_presets/{savePresetName || '...'}.json
            </p>
            <div className="flex justify-end gap-2">
              <button
                onClick={() => setShowSaveDialog(false)}
                className="px-4 py-2 text-sm text-muted hover:text-white"
              >
                Cancel
              </button>
              <button
                onClick={() => savePreset(savePresetName)}
                disabled={!savePresetName.trim() || isLoading}
                className="px-4 py-2 text-sm bg-primary hover:bg-primary-hover text-white rounded disabled:opacity-50"
              >
                {isLoading ? 'Saving...' : 'Save'}
              </button>
            </div>
          </div>
        </div>
      )}
      {/* Card Selector Modal */}
      {showCardSelector && (
        <PresetCardSelector
          presets={presets}
          currentPreset={currentPreset}
          onSelect={(preset) => {
            loadPreset(preset);
            setShowCardSelector(false);
          }}
          onDelete={async (preset) => {
            try {
              const response = await fetch(`/api/config/presets/${encodeURIComponent(preset.name)}`, {
                method: 'DELETE',
              });
              if (response.ok) {
                setMessage({ type: 'success', text: `Deleted preset: ${preset.name}` });
                fetchPresets(); // Refresh list
                if (currentPreset === preset.name) {
                  setCurrentPreset(null);
                }
              } else {
                const error = await response.json();
                setMessage({ type: 'error', text: error.detail || 'Failed to delete preset' });
              }
            } catch (error) {
              setMessage({ type: 'error', text: 'Failed to delete preset' });
            }
          }}
          onClose={() => setShowCardSelector(false)}
        />
      )}
    </header>
  );
}
