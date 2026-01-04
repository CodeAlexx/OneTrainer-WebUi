import { useConfigStore } from '../../stores/configStore';

const TIME_UNITS = ['NEVER', 'EPOCH', 'STEP', 'SECOND', 'MINUTE', 'HOUR', 'ALWAYS'];

export function BackupView() {
  const { config, updateConfig } = useConfigStore();

  // Config values with defaults matching TrainConfig
  const backupAfter = (config as any)?.backup_after ?? 30;
  const backupAfterUnit = (config as any)?.backup_after_unit ?? 'MINUTE';
  const rollingBackup = (config as any)?.rolling_backup ?? false;
  const rollingBackupCount = (config as any)?.rolling_backup_count ?? 3;
  const backupBeforeSave = (config as any)?.backup_before_save ?? true;
  const saveEvery = (config as any)?.save_every ?? 0;
  const saveEveryUnit = (config as any)?.save_every_unit ?? 'NEVER';
  const saveSkipFirst = (config as any)?.save_skip_first ?? 0;
  const saveFilenamePrefix = (config as any)?.save_filename_prefix ?? '';

  const handleChange = (field: string, value: any) => {
    updateConfig({ [field]: value } as any);
  };

  // Action handlers
  const handleBackupNow = async () => {
    try {
      await fetch('/api/training/backup', { method: 'POST' });
    } catch (err) {
      console.error('Failed to trigger backup:', err);
    }
  };

  const handleSaveNow = async () => {
    try {
      await fetch('/api/training/save', { method: 'POST' });
    } catch (err) {
      console.error('Failed to trigger save:', err);
    }
  };

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="h-14 flex items-center px-6 border-b border-dark-border bg-dark-surface">
        <h1 className="text-lg font-medium text-white">Backup & Save Settings</h1>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-6">
        <div className="max-w-3xl space-y-6">

          {/* Backup Settings */}
          <div className="bg-dark-surface rounded-lg border border-dark-border">
            <div className="px-4 py-3 border-b border-dark-border flex items-center justify-between">
              <h2 className="text-sm font-medium text-muted uppercase tracking-wider">Backup Settings</h2>
              <button
                onClick={handleBackupNow}
                className="bg-cyan-600 hover:bg-cyan-500 text-white px-4 py-1.5 rounded-lg text-sm font-medium"
              >
                backup now
              </button>
            </div>
            <div className="p-4 space-y-4">
              {/* Backup After */}
              <div className="flex items-center gap-4">
                <label className="text-sm text-muted w-40">Backup After</label>
                <input
                  type="text"
                  value={backupAfter}
                  onChange={(e) => handleChange('backup_after', parseInt(e.target.value) || 0)}
                  className="input w-20 text-sm"
                />
                <select
                  value={backupAfterUnit}
                  onChange={(e) => handleChange('backup_after_unit', e.target.value)}
                  className="input text-sm"
                >
                  {TIME_UNITS.map(unit => (
                    <option key={unit} value={unit}>{unit}</option>
                  ))}
                </select>
              </div>

              {/* Rolling Backup */}
              <div className="flex items-center gap-4">
                <label className="text-sm text-muted w-40">Rolling Backup</label>
                <button
                  onClick={() => handleChange('rolling_backup', !rollingBackup)}
                  className={`w-9 h-5 rounded-full relative flex-shrink-0 ${rollingBackup ? 'bg-cyan-600' : 'bg-gray-600'}`}
                >
                  <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform ${rollingBackup ? 'translate-x-4' : 'translate-x-0'}`} />
                </button>
              </div>

              {/* Rolling Backup Count */}
              <div className="flex items-center gap-4">
                <label className="text-sm text-muted w-40">Rolling Backup Count</label>
                <input
                  type="text"
                  value={rollingBackupCount}
                  onChange={(e) => handleChange('rolling_backup_count', parseInt(e.target.value) || 3)}
                  className="input w-20 text-sm"
                  disabled={!rollingBackup}
                />
              </div>

              {/* Backup Before Save */}
              <div className="flex items-center gap-4">
                <label className="text-sm text-muted w-40">Backup Before Save</label>
                <button
                  onClick={() => handleChange('backup_before_save', !backupBeforeSave)}
                  className={`w-9 h-5 rounded-full relative flex-shrink-0 ${backupBeforeSave ? 'bg-cyan-600' : 'bg-gray-600'}`}
                >
                  <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform ${backupBeforeSave ? 'translate-x-4' : 'translate-x-0'}`} />
                </button>
              </div>
            </div>
          </div>

          {/* Save Settings */}
          <div className="bg-dark-surface rounded-lg border border-dark-border">
            <div className="px-4 py-3 border-b border-dark-border flex items-center justify-between">
              <h2 className="text-sm font-medium text-muted uppercase tracking-wider">Save Settings</h2>
              <button
                onClick={handleSaveNow}
                className="bg-cyan-600 hover:bg-cyan-500 text-white px-4 py-1.5 rounded-lg text-sm font-medium"
              >
                save now
              </button>
            </div>
            <div className="p-4 space-y-4">
              {/* Save Every */}
              <div className="flex items-center gap-4">
                <label className="text-sm text-muted w-40">Save Every</label>
                <input
                  type="text"
                  value={saveEvery}
                  onChange={(e) => handleChange('save_every', parseInt(e.target.value) || 0)}
                  className="input w-20 text-sm"
                />
                <select
                  value={saveEveryUnit}
                  onChange={(e) => handleChange('save_every_unit', e.target.value)}
                  className="input text-sm"
                >
                  {TIME_UNITS.map(unit => (
                    <option key={unit} value={unit}>{unit}</option>
                  ))}
                </select>
              </div>

              {/* Skip First */}
              <div className="flex items-center gap-4">
                <label className="text-sm text-muted w-40">Skip First</label>
                <input
                  type="text"
                  value={saveSkipFirst}
                  onChange={(e) => handleChange('save_skip_first', parseInt(e.target.value) || 0)}
                  className="input w-20 text-sm"
                />
              </div>

              {/* Save Filename Prefix */}
              <div className="flex items-center gap-4">
                <label className="text-sm text-muted w-40">Save Filename Prefix</label>
                <input
                  type="text"
                  value={saveFilenamePrefix}
                  onChange={(e) => handleChange('save_filename_prefix', e.target.value)}
                  className="input flex-1 text-sm"
                  placeholder="Enter prefix for saved files..."
                />
              </div>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}
