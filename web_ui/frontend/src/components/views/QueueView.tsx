import { Play, Trash2, GripVertical, Clock, CheckCircle, XCircle, AlertCircle, Square, RefreshCw, Plus, ChevronDown, ChevronUp } from 'lucide-react';
import { useEffect, useState, useCallback } from 'react';
import { queueApi, QueuedJob, trainingWs } from '../../lib/api';

type JobStatus = QueuedJob['status'];

const statusConfig: Record<JobStatus, { icon: React.ComponentType<any>; color: string; label: string }> = {
  pending: { icon: Clock, color: 'text-yellow-400', label: 'Pending' },
  running: { icon: RefreshCw, color: 'text-blue-400', label: 'Running' },
  completed: { icon: CheckCircle, color: 'text-success', label: 'Completed' },
  failed: { icon: XCircle, color: 'text-danger', label: 'Failed' },
  cancelled: { icon: AlertCircle, color: 'text-muted', label: 'Cancelled' },
};

function formatDate(dateStr: string | null): string {
  if (!dateStr) return '--';
  const date = new Date(dateStr);
  return date.toLocaleString();
}

function formatDuration(startStr: string | null, endStr: string | null): string {
  if (!startStr) return '--';
  const start = new Date(startStr);
  const end = endStr ? new Date(endStr) : new Date();
  const diffMs = end.getTime() - start.getTime();
  const diffSec = Math.floor(diffMs / 1000);
  const hours = Math.floor(diffSec / 3600);
  const mins = Math.floor((diffSec % 3600) / 60);
  const secs = diffSec % 60;
  if (hours > 0) return `${hours}h ${mins}m ${secs}s`;
  if (mins > 0) return `${mins}m ${secs}s`;
  return `${secs}s`;
}

interface JobCardProps {
  job: QueuedJob;
  onRemove: (id: string) => void;
  onCancel: (id: string) => void;
  onMove: (id: string, direction: 'up' | 'down') => void;
  canMoveUp: boolean;
  canMoveDown: boolean;
  isCurrentJob?: boolean;
}

function JobCard({ job, onRemove, onCancel, onMove, canMoveUp, canMoveDown, isCurrentJob }: JobCardProps) {
  const status = statusConfig[job.status];
  const StatusIcon = status.icon;
  const isRunning = job.status === 'running';
  const isPending = job.status === 'pending';

  const progressPercent = job.progress?.global_step && job.progress?.max_step
    ? Math.round((job.progress.global_step / job.progress.max_step) * 100)
    : 0;

  return (
    <div className={`border rounded-lg p-4 ${isCurrentJob ? 'border-primary bg-primary/5' : 'border-dark-border bg-dark-surface'}`}>
      <div className="flex items-start gap-3">
        {/* Drag handle for pending jobs */}
        {isPending && (
          <div className="flex flex-col gap-1 pt-1">
            <button
              onClick={() => onMove(job.id, 'up')}
              disabled={!canMoveUp}
              className="p-0.5 text-muted hover:text-white disabled:opacity-30 disabled:cursor-not-allowed"
              title="Move up"
            >
              <ChevronUp className="w-4 h-4" />
            </button>
            <GripVertical className="w-4 h-4 text-muted" />
            <button
              onClick={() => onMove(job.id, 'down')}
              disabled={!canMoveDown}
              className="p-0.5 text-muted hover:text-white disabled:opacity-30 disabled:cursor-not-allowed"
              title="Move down"
            >
              <ChevronDown className="w-4 h-4" />
            </button>
          </div>
        )}

        {/* Job info */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <StatusIcon className={`w-4 h-4 ${status.color} ${isRunning ? 'animate-spin' : ''}`} />
            <span className="text-white font-medium truncate">{job.name}</span>
            <span className={`text-xs px-2 py-0.5 rounded ${status.color} bg-white/5`}>
              {status.label}
            </span>
          </div>

          <div className="text-xs text-muted truncate mb-2">
            {job.config_path}
          </div>

          {/* Progress bar for running jobs */}
          {isRunning && job.progress && (
            <div className="mb-2">
              <div className="flex justify-between text-xs text-muted mb-1">
                <span>Step {job.progress.global_step} / {job.progress.max_step || '?'}</span>
                <span>{progressPercent}%</span>
              </div>
              <div className="h-1.5 bg-dark-bg rounded-full overflow-hidden">
                <div
                  className="h-full bg-primary rounded-full transition-all duration-300"
                  style={{ width: `${progressPercent}%` }}
                />
              </div>
              {job.progress.loss != null && (
                <div className="text-xs text-muted mt-1">
                  Loss: {job.progress.loss.toFixed(4)}
                  {job.progress.smooth_loss != null && ` (smooth: ${job.progress.smooth_loss.toFixed(4)})`}
                </div>
              )}
            </div>
          )}

          {/* Timestamps */}
          <div className="flex gap-4 text-xs text-muted">
            <span>Created: {formatDate(job.created_at)}</span>
            {job.started_at && <span>Duration: {formatDuration(job.started_at, job.completed_at)}</span>}
          </div>

          {/* Error message */}
          {job.error && (
            <div className="mt-2 text-xs text-danger bg-danger/10 rounded px-2 py-1">
              {job.error}
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="flex items-center gap-1">
          {isRunning && (
            <button
              onClick={() => onCancel(job.id)}
              className="p-2 text-danger hover:bg-danger/20 rounded"
              title="Cancel job"
            >
              <Square className="w-4 h-4" />
            </button>
          )}
          {isPending && (
            <button
              onClick={() => onRemove(job.id)}
              className="p-2 text-muted hover:text-danger hover:bg-dark-hover rounded"
              title="Remove from queue"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

export function QueueView() {
  const [jobs, setJobs] = useState<QueuedJob[]>([]);
  const [currentJob, setCurrentJob] = useState<QueuedJob | null>(null);
  const [history, setHistory] = useState<QueuedJob[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showHistory, setShowHistory] = useState(false);

  // Fetch queue and history
  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      const [queueResponse, historyResponse] = await Promise.all([
        queueApi.list(),
        queueApi.history(20),
      ]);
      setJobs(queueResponse.data.jobs);
      setCurrentJob(queueResponse.data.current_job);
      setHistory(historyResponse.data.jobs);
      setError(null);
    } catch (err: any) {
      console.error('Failed to fetch queue:', err);
      setError(err.response?.data?.detail || err.message || 'Failed to fetch queue');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // WebSocket updates
  useEffect(() => {
    const handleProgress = (data: any) => {
      // Update current job progress
      setCurrentJob(prev => prev ? { ...prev, progress: data } : null);
    };

    const handleStatus = () => {
      // Refresh queue on status change
      fetchData();
    };

    trainingWs.connect();
    trainingWs.on('progress', handleProgress);
    trainingWs.on('status', handleStatus);
    trainingWs.on('training_complete', fetchData);
    trainingWs.on('job_started', fetchData);

    return () => {
      trainingWs.off('progress', handleProgress);
      trainingWs.off('status', handleStatus);
      trainingWs.off('training_complete', fetchData);
      trainingWs.off('job_started', fetchData);
      trainingWs.disconnect();
    };
  }, [fetchData]);

  // Actions
  const handleStartNext = async () => {
    try {
      await queueApi.startNext();
      await fetchData();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to start next job');
    }
  };

  const handleRemove = async (jobId: string) => {
    try {
      await queueApi.remove(jobId);
      await fetchData();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to remove job');
    }
  };

  const handleCancel = async (jobId: string) => {
    try {
      await queueApi.cancel(jobId);
      await fetchData();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to cancel job');
    }
  };

  const handleMove = async (jobId: string, direction: 'up' | 'down') => {
    const jobIndex = jobs.findIndex(j => j.id === jobId);
    if (jobIndex === -1) return;

    const newPosition = direction === 'up' ? jobIndex - 1 : jobIndex + 1;
    if (newPosition < 0 || newPosition >= jobs.length) return;

    try {
      await queueApi.move(jobId, newPosition);
      await fetchData();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to move job');
    }
  };

  const handleClearHistory = async () => {
    try {
      await queueApi.clearHistory();
      setHistory([]);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to clear history');
    }
  };

  const pendingJobs = jobs.filter(j => j.status === 'pending');
  const hasRunningJob = currentJob?.status === 'running';

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="h-14 flex items-center justify-between px-6 border-b border-dark-border bg-dark-surface">
        <h1 className="text-lg font-medium text-white">Training Queue</h1>
        <div className="flex items-center gap-2">
          <button
            onClick={fetchData}
            disabled={loading}
            className="p-2 text-muted hover:text-white hover:bg-dark-hover rounded disabled:opacity-50"
            title="Refresh"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
          <button className="bg-primary hover:bg-primary-hover text-white px-4 py-1.5 rounded-lg text-sm font-medium flex items-center gap-2">
            <Plus className="w-4 h-4" />
            New Job
          </button>
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="px-6 py-3 bg-red-900/20 border-b border-red-800/50 flex items-center justify-between">
          <span className="text-red-400 text-sm">{error}</span>
          <button onClick={() => setError(null)} className="text-red-400 hover:text-red-300">
            <XCircle className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Main content */}
      <div className="flex-1 overflow-auto p-6 space-y-6">
        {/* Current Job Section */}
        <section>
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-medium text-muted uppercase tracking-wider">Current Job</h2>
            {!hasRunningJob && pendingJobs.length > 0 && (
              <button
                onClick={handleStartNext}
                className="bg-success hover:bg-green-600 text-white px-3 py-1 rounded text-sm font-medium flex items-center gap-1"
              >
                <Play className="w-3 h-3" />
                Start Next
              </button>
            )}
          </div>
          {currentJob ? (
            <JobCard
              job={currentJob}
              onRemove={handleRemove}
              onCancel={handleCancel}
              onMove={() => { }}
              canMoveUp={false}
              canMoveDown={false}
              isCurrentJob
            />
          ) : (
            <div className="text-muted text-sm py-8 text-center border border-dashed border-dark-border rounded-lg">
              No job currently running
            </div>
          )}
        </section>

        {/* Pending Queue Section */}
        <section>
          <h2 className="text-sm font-medium text-muted uppercase tracking-wider mb-3">
            Pending Queue ({pendingJobs.length})
          </h2>
          {pendingJobs.length > 0 ? (
            <div className="space-y-2">
              {pendingJobs.map((job, index) => (
                <JobCard
                  key={job.id}
                  job={job}
                  onRemove={handleRemove}
                  onCancel={handleCancel}
                  onMove={handleMove}
                  canMoveUp={index > 0}
                  canMoveDown={index < pendingJobs.length - 1}
                />
              ))}
            </div>
          ) : (
            <div className="text-muted text-sm py-8 text-center border border-dashed border-dark-border rounded-lg">
              No jobs in queue
            </div>
          )}
        </section>

        {/* History Section */}
        <section>
          <div className="flex items-center justify-between mb-3">
            <button
              onClick={() => setShowHistory(!showHistory)}
              className="flex items-center gap-2 text-sm font-medium text-muted uppercase tracking-wider hover:text-white"
            >
              {showHistory ? <ChevronDown className="w-4 h-4" /> : <ChevronUp className="w-4 h-4" />}
              History ({history.length})
            </button>
            {history.length > 0 && showHistory && (
              <button
                onClick={handleClearHistory}
                className="text-xs text-muted hover:text-danger"
              >
                Clear History
              </button>
            )}
          </div>
          {showHistory && (
            history.length > 0 ? (
              <div className="space-y-2">
                {history.map((job) => (
                  <JobCard
                    key={job.id}
                    job={job}
                    onRemove={() => { }}
                    onCancel={() => { }}
                    onMove={() => { }}
                    canMoveUp={false}
                    canMoveDown={false}
                  />
                ))}
              </div>
            ) : (
              <div className="text-muted text-sm py-4 text-center">
                No completed jobs
              </div>
            )
          )}
        </section>
      </div>
    </div>
  );
}
