import { useState, useEffect } from 'react';
import { Play, Square, RefreshCw, ExternalLink, FolderOpen, BarChart3, Clock } from 'lucide-react';
import { tensorboardApi, type TensorBoardStatus, type TensorBoardLog } from '../../lib/api';

export function TensorBoardView() {
  const [status, setStatus] = useState<TensorBoardStatus | null>(null);
  const [logs, setLogs] = useState<TensorBoardLog[]>([]);
  const [workspace, setWorkspace] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [starting, setStarting] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [selectedLog, setSelectedLog] = useState<string | null>(null);
  const [port, setPort] = useState(6006);
  const [error, setError] = useState<string | null>(null);

  // Fetch status and logs on mount
  useEffect(() => {
    fetchStatus();
    fetchLogs();
  }, []);

  const fetchStatus = async () => {
    try {
      const response = await tensorboardApi.getStatus();
      setStatus(response.data);
    } catch (err) {
      console.error('Failed to fetch TensorBoard status:', err);
    }
  };

  const fetchLogs = async () => {
    setLoading(true);
    try {
      const response = await tensorboardApi.getLogs();
      setLogs(response.data.logs);
      setWorkspace(response.data.workspace);
    } catch (err) {
      console.error('Failed to fetch logs:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleStart = async () => {
    setStarting(true);
    setError(null);
    try {
      const response = await tensorboardApi.start(selectedLog || undefined, port);
      setStatus(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to start TensorBoard');
    } finally {
      setStarting(false);
    }
  };

  const handleStop = async () => {
    setStopping(true);
    try {
      const response = await tensorboardApi.stop();
      setStatus(response.data);
    } catch (err) {
      console.error('Failed to stop TensorBoard:', err);
    } finally {
      setStopping(false);
    }
  };

  const formatTimeAgo = (timestamp: number) => {
    const seconds = Math.floor(Date.now() / 1000 - timestamp);
    if (seconds < 60) return 'just now';
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
    return `${Math.floor(seconds / 86400)}d ago`;
  };

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="h-14 flex items-center justify-between px-6 border-b border-dark-border bg-dark-surface">
        <div className="flex items-center gap-3">
          <BarChart3 className="w-5 h-5 text-primary" />
          <h1 className="text-lg font-medium text-white">TensorBoard</h1>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={fetchLogs}
            className="p-2 hover:bg-dark-hover rounded text-muted hover:text-white"
            title="Refresh"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-auto p-6">
        <div className="max-w-4xl mx-auto space-y-6">
          {/* Status Card */}
          <div className="bg-dark-surface rounded-lg border border-dark-border p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className={`w-3 h-3 rounded-full ${status?.running ? 'bg-green-500' : 'bg-gray-500'}`} />
                <div>
                  <h2 className="text-white font-medium">
                    TensorBoard Server {status?.running ? 'Running' : 'Stopped'}
                  </h2>
                  {status?.running && status?.url && (
                    <p className="text-sm text-muted">
                      {status.url} - {status.logdir}
                    </p>
                  )}
                </div>
              </div>
              <div className="flex items-center gap-2">
                {status?.running ? (
                  <>
                    <a
                      href={status.url || '#'}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-2 px-3 py-1.5 bg-primary hover:bg-primary/80 text-white rounded text-sm"
                    >
                      <ExternalLink className="w-4 h-4" />
                      Open TensorBoard
                    </a>
                    <button
                      onClick={handleStop}
                      disabled={stopping}
                      className="flex items-center gap-2 px-3 py-1.5 bg-red-600 hover:bg-red-700 text-white rounded text-sm disabled:opacity-50"
                    >
                      <Square className="w-4 h-4" />
                      {stopping ? 'Stopping...' : 'Stop'}
                    </button>
                  </>
                ) : (
                  <button
                    onClick={handleStart}
                    disabled={starting}
                    className="flex items-center gap-2 px-3 py-1.5 bg-green-600 hover:bg-green-700 text-white rounded text-sm disabled:opacity-50"
                  >
                    <Play className="w-4 h-4" />
                    {starting ? 'Starting...' : 'Start TensorBoard'}
                  </button>
                )}
              </div>
            </div>

            {error && (
              <div className="mt-3 p-3 bg-red-500/10 border border-red-500/30 rounded text-red-400 text-sm">
                {error}
              </div>
            )}
          </div>

          {/* Settings */}
          {!status?.running && (
            <div className="bg-dark-surface rounded-lg border border-dark-border p-4">
              <h3 className="text-sm font-medium text-white mb-3">Settings</h3>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-xs text-muted block mb-1">Port</label>
                  <input
                    type="text"
                    value={port}
                    onChange={(e) => setPort(parseInt(e.target.value) || 6006)}
                    className="input w-full"
                    min="1024"
                    max="65535"
                  />
                </div>
                <div>
                  <label className="text-xs text-muted block mb-1">Log Directory</label>
                  <select
                    value={selectedLog || ''}
                    onChange={(e) => setSelectedLog(e.target.value || null)}
                    className="input w-full"
                  >
                    <option value="">All Logs (workspace)</option>
                    {logs.map((log) => (
                      <option key={log.path} value={log.path}>
                        {log.name}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>
          )}

          {/* Embedded TensorBoard (iframe) */}
          {status?.running && status?.url && (
            <div className="bg-dark-surface rounded-lg border border-dark-border overflow-hidden">
              <div className="px-4 py-2 border-b border-dark-border flex items-center justify-between">
                <span className="text-sm text-muted">TensorBoard Dashboard</span>
                <a
                  href={status.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-xs text-primary hover:underline flex items-center gap-1"
                >
                  Open in new tab <ExternalLink className="w-3 h-3" />
                </a>
              </div>
              <iframe
                src={status.url}
                className="w-full h-[600px] bg-white"
                title="TensorBoard"
              />
            </div>
          )}

          {/* Available Logs */}
          <div className="bg-dark-surface rounded-lg border border-dark-border">
            <div className="px-4 py-3 border-b border-dark-border">
              <h3 className="text-sm font-medium text-white">Training Logs</h3>
              <p className="text-xs text-muted mt-1">
                {workspace}
              </p>
            </div>
            <div className="divide-y divide-dark-border">
              {loading ? (
                <div className="p-8 text-center text-muted">
                  <RefreshCw className="w-6 h-6 animate-spin mx-auto mb-2" />
                  Loading logs...
                </div>
              ) : logs.length === 0 ? (
                <div className="p-8 text-center text-muted">
                  <FolderOpen className="w-12 h-12 mx-auto mb-3 opacity-30" />
                  <p>No training logs found</p>
                  <p className="text-xs mt-1">Run a training with TensorBoard enabled to see logs here</p>
                </div>
              ) : (
                logs.map((log) => (
                  <div
                    key={log.path}
                    className={`p-4 hover:bg-dark-hover cursor-pointer transition-colors ${
                      selectedLog === log.path ? 'bg-primary/10 border-l-2 border-primary' : ''
                    }`}
                    onClick={() => setSelectedLog(selectedLog === log.path ? null : log.path)}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <BarChart3 className="w-5 h-5 text-primary" />
                        <div>
                          <h4 className="text-white font-medium">{log.name}</h4>
                          <p className="text-xs text-muted">{log.path}</p>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="flex items-center gap-1 text-sm text-muted">
                          <Clock className="w-3 h-3" />
                          {formatTimeAgo(log.modified)}
                        </div>
                        <div className="text-xs text-muted">
                          {log.event_count} event file{log.event_count !== 1 ? 's' : ''}
                        </div>
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Help */}
          <div className="bg-dark-surface rounded-lg border border-dark-border p-4">
            <h3 className="text-sm font-medium text-white mb-2">About TensorBoard</h3>
            <div className="text-sm text-muted space-y-2">
              <p>
                TensorBoard provides visualization of training metrics including loss curves, learning rate schedules,
                and sample images generated during training.
              </p>
              <p>
                To enable TensorBoard logging for your training runs, make sure the "TensorBoard" option is enabled
                in your training configuration.
              </p>
              <p className="text-xs">
                Logs are stored in: <code className="bg-dark-bg px-1 rounded">{workspace}</code>
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
