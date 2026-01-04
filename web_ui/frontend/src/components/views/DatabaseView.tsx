import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { databaseApi, DbStatus, DbPreset, DbVersion, DbTrainingRun } from '../../lib/api';

export function DatabaseView() {
  const [status, setStatus] = useState<DbStatus | null>(null);
  const [presets, setPresets] = useState<DbPreset[]>([]);
  const [trainingRuns, setTrainingRuns] = useState<DbTrainingRun[]>([]);
  const [selectedPreset, setSelectedPreset] = useState<DbPreset | null>(null);
  const [versions, setVersions] = useState<DbVersion[]>([]);
  const [loading, setLoading] = useState(true);
  const [migrating, setMigrating] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  useEffect(() => {
    loadStatus();
  }, []);

  const loadStatus = async () => {
    try {
      setLoading(true);
      const res = await databaseApi.getStatus();
      setStatus(res.data);

      if (res.data.initialized) {
        const [presetsRes, runsRes] = await Promise.all([
          databaseApi.listPresets(),
          databaseApi.listTrainingRuns(10)
        ]);
        setPresets(presetsRes.data);
        setTrainingRuns(runsRes.data);
      }
    } catch (err) {
      console.error('Failed to load database status:', err);
      setMessage({ type: 'error', text: 'Failed to load database status' });
    } finally {
      setLoading(false);
    }
  };

  const handleInit = async () => {
    try {
      setLoading(true);
      await databaseApi.init();
      setMessage({ type: 'success', text: 'Database initialized successfully' });
      await loadStatus();
    } catch (err: any) {
      setMessage({ type: 'error', text: err.response?.data?.detail || 'Failed to initialize database' });
    } finally {
      setLoading(false);
    }
  };

  const handleMigrate = async () => {
    try {
      setMigrating(true);
      setMessage(null);
      const res = await databaseApi.migrate();
      const data = res.data;

      let msg = 'Migration complete: ';
      if (data.presets?.summary) {
        msg += `Presets: ${data.presets.summary.migrated_count} migrated, ${data.presets.summary.skipped_count} skipped. `;
      }
      if (data.concepts?.summary) {
        msg += `Concepts: ${data.concepts.summary.migrated_count} migrated. `;
      }
      if (data.samples?.summary) {
        msg += `Samples: ${data.samples.summary.migrated_count} migrated.`;
      }

      setMessage({ type: 'success', text: msg });
      await loadStatus();
    } catch (err: any) {
      setMessage({ type: 'error', text: err.response?.data?.detail || 'Migration failed' });
    } finally {
      setMigrating(false);
    }
  };

  const loadPresetVersions = async (preset: DbPreset) => {
    try {
      setSelectedPreset(preset);
      const res = await databaseApi.getPresetVersions(preset.id);
      setVersions(res.data);
    } catch (err) {
      console.error('Failed to load versions:', err);
    }
  };

  const handleRollback = async (presetId: number, version: number) => {
    try {
      await databaseApi.rollbackPreset(presetId, version);
      setMessage({ type: 'success', text: `Rolled back to version ${version}` });
      await loadStatus();
      if (selectedPreset) {
        loadPresetVersions(selectedPreset);
      }
    } catch (err: any) {
      setMessage({ type: 'error', text: err.response?.data?.detail || 'Rollback failed' });
    }
  };

  const formatDate = (dateStr: string | null) => {
    if (!dateStr) return '-';
    return new Date(dateStr).toLocaleString();
  };

  const formatDuration = (seconds: number | null) => {
    if (!seconds) return '-';
    const hours = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    if (hours > 0) return `${hours}h ${mins}m`;
    if (mins > 0) return `${mins}m ${secs}s`;
    return `${secs}s`;
  };

  if (loading && !status) {
    return (
      <div className="p-6">
        <Card>
          <CardContent className="p-6">
            <p className="text-muted-foreground">Loading database status...</p>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Status Message */}
      {message && (
        <div className={`p-4 rounded-lg ${message.type === 'success' ? 'bg-green-900/20 text-green-400' : 'bg-red-900/20 text-red-400'}`}>
          {message.text}
        </div>
      )}

      {/* Database Status */}
      <Card>
        <CardHeader>
          <CardTitle>Database Status</CardTitle>
          <p className="text-sm text-muted-foreground">SQLite storage for presets, concepts, samples, and training history</p>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <p className="text-sm text-muted-foreground">Status</p>
              <p className={`font-medium ${status?.initialized ? 'text-green-400' : 'text-yellow-400'}`}>
                {status?.initialized ? 'Initialized' : 'Not Initialized'}
              </p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Path</p>
              <p className="font-mono text-sm truncate" title={status?.path || ''}>
                {status?.path || '-'}
              </p>
            </div>
            {status?.counts && (
              <>
                <div>
                  <p className="text-sm text-muted-foreground">Presets</p>
                  <p className="font-medium">{status.counts.presets}</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Training Runs</p>
                  <p className="font-medium">{status.counts.training_runs}</p>
                </div>
              </>
            )}
          </div>

          <div className="flex gap-2">
            {!status?.initialized && (
              <Button onClick={handleInit} disabled={loading}>
                Initialize Database
              </Button>
            )}
            <Button onClick={handleMigrate} disabled={migrating || !status?.initialized} variant="secondary">
              {migrating ? 'Migrating...' : 'Migrate from JSON'}
            </Button>
            <Button onClick={loadStatus} disabled={loading} variant="secondary">
              Refresh
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Presets */}
      {status?.initialized && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle>Presets ({presets.length})</CardTitle>
              <p className="text-sm text-muted-foreground">Training configuration presets stored in database</p>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {presets.map((preset) => (
                  <div
                    key={preset.id}
                    className={`p-3 rounded-lg border cursor-pointer transition-colors ${
                      selectedPreset?.id === preset.id
                        ? 'border-blue-500 bg-blue-500/10'
                        : 'border-border hover:border-muted-foreground'
                    }`}
                    onClick={() => loadPresetVersions(preset)}
                  >
                    <div className="flex justify-between items-start">
                      <div>
                        <p className="font-medium">{preset.name}</p>
                        <p className="text-sm text-muted-foreground">
                          {preset.model_type} / {preset.training_method}
                        </p>
                      </div>
                      <div className="flex gap-1">
                        {preset.is_builtin && (
                          <span className="text-xs px-2 py-1 rounded bg-blue-500/20 text-blue-400">builtin</span>
                        )}
                        {preset.is_favorite && (
                          <span className="text-xs px-2 py-1 rounded bg-yellow-500/20 text-yellow-400">fav</span>
                        )}
                      </div>
                    </div>
                    <p className="text-xs text-muted-foreground mt-1">
                      Updated: {formatDate(preset.updated_at)}
                    </p>
                  </div>
                ))}
                {presets.length === 0 && (
                  <p className="text-muted-foreground text-sm">No presets in database. Click "Migrate from JSON" to import.</p>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Version History */}
          <Card>
            <CardHeader>
              <CardTitle>Version History</CardTitle>
              <p className="text-sm text-muted-foreground">
                {selectedPreset ? `History for "${selectedPreset.name}"` : 'Select a preset to view history'}
              </p>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {versions.map((version) => (
                  <div key={version.id} className="p-3 rounded-lg border border-border">
                    <div className="flex justify-between items-start">
                      <div>
                        <p className="font-medium">Version {version.version}</p>
                        <p className="text-sm text-muted-foreground">
                          {version.change_type}
                          {version.change_description && `: ${version.change_description}`}
                        </p>
                        {version.changed_fields.length > 0 && (
                          <p className="text-xs text-muted-foreground">
                            Changed: {version.changed_fields.join(', ')}
                          </p>
                        )}
                      </div>
                      {version.version > 1 && selectedPreset && (
                        <Button
                          size="sm"
                          variant="secondary"
                          onClick={() => handleRollback(selectedPreset.id, version.version)}
                        >
                          Rollback
                        </Button>
                      )}
                    </div>
                    <p className="text-xs text-muted-foreground mt-1">
                      {formatDate(version.created_at)}
                      {version.created_by && ` by ${version.created_by}`}
                    </p>
                  </div>
                ))}
                {versions.length === 0 && selectedPreset && (
                  <p className="text-muted-foreground text-sm">No version history available.</p>
                )}
                {!selectedPreset && (
                  <p className="text-muted-foreground text-sm">Select a preset from the list to view its version history.</p>
                )}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Recent Training Runs */}
      {status?.initialized && trainingRuns.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Recent Training Runs</CardTitle>
            <p className="text-sm text-muted-foreground">History of training sessions</p>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border">
                    <th className="text-left p-2">Name</th>
                    <th className="text-left p-2">Status</th>
                    <th className="text-left p-2">Progress</th>
                    <th className="text-left p-2">Loss</th>
                    <th className="text-left p-2">Duration</th>
                    <th className="text-left p-2">Started</th>
                  </tr>
                </thead>
                <tbody>
                  {trainingRuns.map((run) => (
                    <tr key={run.id} className="border-b border-border/50">
                      <td className="p-2">{run.run_name || run.preset_name || `Run ${run.id}`}</td>
                      <td className="p-2">
                        <span className={`px-2 py-1 rounded text-xs ${
                          run.status === 'completed' ? 'bg-green-500/20 text-green-400' :
                          run.status === 'error' ? 'bg-red-500/20 text-red-400' :
                          run.status === 'training' ? 'bg-blue-500/20 text-blue-400' :
                          'bg-gray-500/20 text-gray-400'
                        }`}>
                          {run.status}
                        </span>
                      </td>
                      <td className="p-2">
                        {run.total_epochs
                          ? `${run.current_epoch}/${run.total_epochs} epochs`
                          : run.total_steps
                          ? `${run.current_step}/${run.total_steps} steps`
                          : '-'}
                      </td>
                      <td className="p-2">{run.final_loss?.toFixed(4) || '-'}</td>
                      <td className="p-2">{formatDuration(run.total_duration_seconds)}</td>
                      <td className="p-2">{formatDate(run.started_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
