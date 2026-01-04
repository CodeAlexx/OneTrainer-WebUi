import { useEffect, useState, useRef } from 'react';
import { Thermometer, Fan, Gauge, Cpu, HardDrive, Zap, Terminal, Trash2, TrendingDown, Play, Square, Loader2, ChevronDown, ChevronRight } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { systemApi, trainingApi, trainingWs, configApi, type SystemInfo, type TrainingProgress, type PresetInfo } from '../../lib/api';
import { useConfigStore } from '../../stores/configStore';

interface LossDataPoint {
  step: number;
  loss: number | null;
  smoothLoss: number | null;
}

interface LogEntry {
  id: number;
  timestamp: Date;
  type: 'step' | 'epoch' | 'sampling' | 'backup' | 'info' | 'error';
  message: string;
  progress?: number;
  details?: {
    current?: number;
    total?: number;
    elapsed?: string;
    remaining?: string;
    speed?: string;
    loss?: number | null;
    smoothLoss?: number | null;
  };
}

export function DashboardView() {
  const {
    config,
    currentPreset,
    setConfig: setStoreConfig,
    setCurrentPreset
  } = useConfigStore();
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  const [trainingLogs, setTrainingLogs] = useState<LogEntry[]>([]);
  const [trainingProgress, setTrainingProgress] = useState<TrainingProgress | null>(null);
  const [lossHistory, setLossHistory] = useState<LossDataPoint[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  // Preset selection
  const [presets, setPresets] = useState<PresetInfo[]>([]);
  const [selectedPreset, setSelectedPreset] = useState<string>(currentPreset || '');
  const [loadingPreset, setLoadingPreset] = useState(false);
  const [isStopping, setIsStopping] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState<string>('idle');
  const [lossChartCollapsed, setLossChartCollapsed] = useState(false);
  const logContainerRef = useRef<HTMLDivElement>(null);
  const logIdRef = useRef(0);

  // Get model type and training method from config (with debugging)
  const configModel = (config as any)?.model_type;
  const configMethod = (config as any)?.training_method;

  // Map model type enums to display names
  const modelTypeDisplay = configModel ?
    (configModel === 'Z_IMAGE' ? 'Z-Image' :
      configModel === 'FLUX_DEV_1' ? 'Flux' : configModel) : '';
  const trainingMethodDisplay = configMethod || '';

  useEffect(() => {
    const fetchSystemInfo = async () => {
      try {
        const response = await systemApi.getInfo();
        setSystemInfo(response.data);
        setLastUpdate(new Date());
        setError(null);
      } catch (err) {
        setError('Failed to fetch GPU information');
        console.error('Error fetching system info:', err);
      } finally {
        setLoading(false);
      }
    };

    // Initial fetch
    fetchSystemInfo();

    // Refresh every 5 seconds
    const interval = setInterval(fetchSystemInfo, 5000);

    return () => clearInterval(interval);
  }, []);

  // Fetch training status on mount to sync isTraining state
  useEffect(() => {
    const fetchTrainingStatus = async () => {
      try {
        const response = await trainingApi.getStatus();
        const status = response.data;
        // Explicitly set true OR false to ensure UI matches backend
        setIsTraining(!!status?.is_training);
        if (status?.status) {
          setTrainingStatus(status.status);
        }
      } catch (err) {
        // If backend is not available, that's fine
        console.log('Could not fetch training status');
      }
    };
    fetchTrainingStatus();
  }, []);

  // Subscribe to training WebSocket updates
  useEffect(() => {
    const handleProgress = (data: TrainingProgress) => {
      setTrainingProgress(data);
      // Removed implicit setIsTraining(true) - rely on training_state updates

      // Track loss history for chart (only when we have loss data)
      if (data.loss !== null || data.smooth_loss !== null) {
        setLossHistory(prev => {
          const newPoint: LossDataPoint = {
            step: data.current_step,
            loss: data.loss,
            smoothLoss: data.smooth_loss,
          };
          // Keep last 500 data points for performance
          return [...prev, newPoint].slice(-500);
        });
      }

      // Add log entry for progress updates
      const newEntry: LogEntry = {
        id: logIdRef.current++,
        timestamp: new Date(),
        type: 'step',
        message: `step: ${data.current_step}/${data.total_steps}`,
        progress: (data.current_step / data.total_steps) * 100,
        details: {
          current: data.current_step,
          total: data.total_steps,
          elapsed: data.elapsed_time || '00:00',
          remaining: data.remaining_time || '00:00',
          speed: data.samples_per_second ? `${data.samples_per_second.toFixed(2)}s/it` : '--',
          loss: data.loss,
          smoothLoss: data.smooth_loss,
        }
      };

      setTrainingLogs(prev => {
        // Skip if the last step entry is the same step (deduplicate)
        const lastStepEntry = [...prev].reverse().find(e => e.type === 'step');
        if (lastStepEntry?.details?.current === data.current_step) {
          return prev;
        }
        const updated = [...prev, newEntry];
        return updated.slice(-500);
      });
    };

    const handleSampling = (data: any) => {
      const newEntry: LogEntry = {
        id: logIdRef.current++,
        timestamp: new Date(),
        type: 'sampling',
        message: `sampling: ${data.current || 0}/${data.total || 0}`,
        progress: data.total ? (data.current / data.total) * 100 : 0,
        details: {
          current: data.current,
          total: data.total,
          elapsed: data.elapsed || '00:00',
          remaining: data.remaining || '00:00',
          speed: data.speed || '--',
        }
      };

      setTrainingLogs(prev => [...prev, newEntry].slice(-500));
    };

    const handleBackup = (data: any) => {
      const newEntry: LogEntry = {
        id: logIdRef.current++,
        timestamp: new Date(),
        type: 'backup',
        message: `Creating Backup ${data.path || ''}`,
      };

      setTrainingLogs(prev => [...prev, newEntry].slice(-500));
    };

    const handleLog = (data: any) => {
      const message = data.message || '';

      // Skip generic "Training ..." messages to reduce noise
      if (message === 'Training ...' || message === 'Training...') {
        return;
      }

      const newEntry: LogEntry = {
        id: logIdRef.current++,
        timestamp: new Date(),
        type: data.level === 'error' ? 'error' : 'info',
        message: message,
      };

      setTrainingLogs(prev => {
        // Skip duplicate consecutive messages
        const lastEntry = prev[prev.length - 1];
        if (lastEntry?.message === message && lastEntry?.type === newEntry.type) {
          return prev;
        }
        return [...prev, newEntry].slice(-500);
      });
    };

    // Handle state updates (start, stop, error)
    const handleStateUpdate = (data: any) => {
      console.log('Training state update:', data);
      if (data) {
        if (data.status) {
          setTrainingStatus(data.status);
        }

        if (data.status === 'idle' || data.status === 'stopped' || data.status === 'completed' || data.status === 'error') {
          setIsTraining(false);
        } else if (data.is_training !== undefined) {
          setIsTraining(data.is_training);
        }
      }
    };

    // Handle initial connection with backend state
    const handleConnected = (data: any) => {
      console.log('WebSocket connected, initial state:', data);
      // Ignore empty event from onopen
      if (Object.keys(data || {}).length === 0) return;

      if (data) {
        if (data.status) {
          setTrainingStatus(data.status);
        }

        if (data.status === 'idle' || data.status === 'stopped' || data.status === 'completed' || data.status === 'error') {
          setIsTraining(false);
        } else if (data.is_training !== undefined) {
          setIsTraining(data.is_training);
        }
      }
    };

    trainingWs.on('connected', handleConnected);
    trainingWs.on('training_state', handleStateUpdate);
    trainingWs.on('progress', handleProgress);
    trainingWs.on('sampling', handleSampling);
    trainingWs.on('backup', handleBackup);
    trainingWs.on('log', handleLog);

    return () => {
      trainingWs.off('connected', handleConnected);
      trainingWs.off('training_state', handleStateUpdate);
      trainingWs.off('progress', handleProgress);
      trainingWs.off('sampling', handleSampling);
      trainingWs.off('backup', handleBackup);
      trainingWs.off('log', handleLog);
    };
  }, []);

  // Auto-scroll log container
  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [trainingLogs]);

  const clearLogs = () => {
    setTrainingLogs([]);
    setLossHistory([]);
  };

  // Fetch presets on mount - use training_presets folder for custom presets
  useEffect(() => {
    const fetchPresets = async () => {
      try {
        // Look in training_presets folder for custom presets (not workspace/config which has auto-saves)
        const response = await configApi.getPresets();
        setPresets(response.data.presets);
      } catch (err) {
        console.error('Failed to load presets:', err);
      }
    };
    fetchPresets();
  }, []);

  // Sync selectedPreset with store
  useEffect(() => {
    if (currentPreset) {
      setSelectedPreset(currentPreset);
    }
  }, [currentPreset]);

  // Handle preset selection and load config
  const handlePresetChange = async (presetName: string) => {
    console.log('[Dashboard] handlePresetChange called with:', presetName);
    setSelectedPreset(presetName);
    if (!presetName) return;

    setLoadingPreset(true);
    try {
      // Load from default training_presets folder
      console.log('[Dashboard] Loading preset from API...');
      const response = await configApi.loadPreset(presetName);
      console.log('[Dashboard] API response:', response.data);
      const configData = response.data.config || response.data;
      console.log('[Dashboard] Setting config:', configData);
      setStoreConfig(configData);
      setCurrentPreset(presetName);
      console.log('[Dashboard] Preset loaded successfully');
    } catch (err) {
      console.error('Failed to load preset:', err);
    } finally {
      setLoadingPreset(false);
    }
  };

  const handleStartTraining = async () => {
    // If no config loaded, don't start
    if (!config || Object.keys(config).length === 0) {
      const msg = 'No configuration loaded. Please select a preset or load a config.';
      console.error(msg);
      setError(msg);
      return;
    }

    setIsStarting(true);
    setError(null); // Clear previous errors
    try {
      // Save config to temp file and start training
      const saveResponse = await configApi.saveTemp(config);
      const configPath = saveResponse.data.path;
      await trainingApi.start(configPath);
      setIsTraining(true);
    } catch (err: any) {
      console.error('Failed to start training:', err);
      // Extract detailed error message from backend response if available
      const errorMsg = err.response?.data?.detail || err.message || 'Failed to start training';
      setError(errorMsg);
    } finally {
      setIsStarting(false);
    }
  };

  const handleStopTraining = async () => {
    setIsStopping(true);
    try {
      await trainingApi.stop();
      setIsTraining(false);
    } catch (err: any) {
      // If backend says not running (409), sync UI state to false
      if (err.response?.status === 409) {
        setIsTraining(false);
      }
      console.error('Failed to stop training:', err);
    } finally {
      setIsStopping(false);
    }
  };



  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  const formatMemoryGB = (bytes: number) => {
    return (bytes / (1024 ** 3)).toFixed(1);
  };

  // Get first GPU or default values
  const gpu = systemInfo?.gpus?.[0];
  const memoryAllocatedGB = gpu ? formatMemoryGB(gpu.memory_allocated) : '--';
  const memoryTotalGB = gpu ? formatMemoryGB(gpu.memory_total) : '--';
  const memoryPercentage = gpu ? (gpu.memory_allocated / gpu.memory_total) * 100 : 0;
  const gpuUtilization = gpu?.utilization ?? 0;

  return (
    <div className="p-6">
      {/* Header with Training Controls */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-4">
          <h1 className="text-xl font-semibold text-white">Dashboard</h1>
          {/* Preset Selector */}
          <select
            className="bg-dark-bg border border-dark-border rounded px-3 py-1.5 text-sm text-white min-w-[200px]"
            value={selectedPreset}
            onChange={(e) => handlePresetChange(e.target.value)}
            disabled={loadingPreset || isTraining}
          >
            <option value="">Select preset...</option>
            {presets.map((preset) => (
              <option key={preset.name} value={preset.name}>
                {preset.name}
              </option>
            ))}
          </select>
          {loadingPreset && <Loader2 className="w-4 h-4 animate-spin text-muted" />}
          {/* Model Type and Training Method Badges */}
          <div className="flex items-center gap-2">
            {modelTypeDisplay && (
              <span className="px-2 py-1 bg-purple-600/20 text-purple-400 rounded text-xs font-medium">
                {modelTypeDisplay}
              </span>
            )}
            {trainingMethodDisplay && (
              <span className="px-2 py-1 bg-blue-600/20 text-blue-400 rounded text-xs font-medium">
                {trainingMethodDisplay}
              </span>
            )}
            {currentPreset && (
              <span className="px-2 py-1 bg-dark-hover text-muted rounded text-xs truncate max-w-48">
                {currentPreset}
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            {!isTraining ? (
              <button
                onClick={handleStartTraining}
                disabled={isStarting}
                className="flex items-center gap-2 px-4 py-2 bg-success hover:bg-success/80 disabled:bg-success/50 text-white rounded-lg font-medium transition-colors"
              >
                {isStarting ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Play className="w-4 h-4" />
                )}
                Start Training
              </button>
            ) : (
              <button
                onClick={handleStopTraining}
                disabled={isStopping}
                className="flex items-center gap-2 px-4 py-2 bg-danger hover:bg-danger/80 disabled:bg-danger/50 text-white rounded-lg font-medium transition-colors"
              >
                {isStopping ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Square className="w-4 h-4" />
                )}
                Stop Training
              </button>
            )}
          </div>
        </div>
        <span className="text-sm text-muted">
          Last updated: {loading ? '--' : formatTime(lastUpdate)}
        </span>
      </div>

      {/* Error State */}
      {error && (
        <div className="mb-6 p-4 bg-danger/10 border border-danger/20 rounded-lg text-danger">
          {error}
        </div>
      )}

      {/* GPU Monitor Card - Compact 2 rows */}
      <div className="bg-black rounded-lg border border-dark-border p-3">
        {loading ? (
          <div className="text-center text-muted py-2">Loading...</div>
        ) : gpu ? (
          <div className="space-y-2">
            {/* Row 1: GPU name and stats */}
            <div className="flex items-center gap-6">
              <div className="flex items-center gap-2 min-w-0">
                <Cpu className="w-4 h-4 text-primary flex-shrink-0" />
                <span className="text-white text-sm font-medium truncate">{gpu.name}</span>
              </div>
              <div className="flex items-center gap-4 text-sm">
                <div className="flex items-center gap-1.5">
                  <Thermometer className="w-3.5 h-3.5 text-danger" />
                  <span className={`font-medium ${gpu.temperature && gpu.temperature > 80 ? 'text-danger' : gpu.temperature && gpu.temperature > 65 ? 'text-warning' : 'text-success'}`}>
                    {gpu.temperature !== null ? `${gpu.temperature}°C` : '--'}
                  </span>
                </div>
                <div className="flex items-center gap-1.5">
                  <Fan className="w-3.5 h-3.5 text-success" />
                  <span className="text-white">{gpu.fan_speed !== null ? `${gpu.fan_speed}%` : '--'}</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <Zap className="w-3.5 h-3.5 text-warning" />
                  <span className="text-white">
                    {gpu.power_draw !== null && gpu.power_limit !== null
                      ? `${gpu.power_draw.toFixed(0)}/${gpu.power_limit.toFixed(0)}W`
                      : '--'}
                  </span>
                </div>
              </div>
            </div>
            {/* Row 2: Progress bars */}
            <div className="grid grid-cols-3 gap-4">
              <div>
                <div className="flex items-center justify-between text-xs mb-1">
                  <span className="text-muted flex items-center gap-1"><Gauge className="w-3 h-3" /> GPU</span>
                  <span className="text-white">{gpuUtilization ? `${gpuUtilization}%` : '--'}</span>
                </div>
                <div className="h-2 bg-dark-bg rounded-full overflow-hidden">
                  <div className="h-full bg-primary rounded-full" style={{ width: `${gpuUtilization || 0}%` }} />
                </div>
              </div>
              <div>
                <div className="flex items-center justify-between text-xs mb-1">
                  <span className="text-muted flex items-center gap-1"><HardDrive className="w-3 h-3" /> VRAM</span>
                  <span className="text-white">{memoryAllocatedGB}/{memoryTotalGB}GB</span>
                </div>
                <div className="h-2 bg-dark-bg rounded-full overflow-hidden">
                  <div className="h-full bg-cyan-500 rounded-full" style={{ width: `${memoryPercentage}%` }} />
                </div>
              </div>
              <div>
                <div className="flex items-center justify-between text-xs mb-1">
                  <span className="text-muted flex items-center gap-1"><Zap className="w-3 h-3" /> Power</span>
                  <span className="text-white">
                    {gpu.power_draw !== null && gpu.power_limit !== null
                      ? `${((gpu.power_draw / gpu.power_limit) * 100).toFixed(0)}%`
                      : '--'}
                  </span>
                </div>
                <div className="h-2 bg-dark-bg rounded-full overflow-hidden">
                  <div className="h-full bg-warning rounded-full" style={{ width: `${gpu.power_draw && gpu.power_limit ? (gpu.power_draw / gpu.power_limit) * 100 : 0}%` }} />
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="text-center text-muted py-2">No GPU detected</div>
        )}
      </div>

      {/* Loss Chart - Collapsible */}
      {lossHistory.length > 0 && (
        <div className="mt-6 bg-black rounded-lg border border-dark-border">
          <div
            className="px-4 py-3 border-b border-dark-border flex items-center gap-2 cursor-pointer hover:bg-dark-hover/30 transition-colors"
            onClick={() => setLossChartCollapsed(!lossChartCollapsed)}
          >
            {lossChartCollapsed ? (
              <ChevronRight className="w-4 h-4 text-muted" />
            ) : (
              <ChevronDown className="w-4 h-4 text-muted" />
            )}
            <TrendingDown className="w-4 h-4 text-muted" />
            <h2 className="text-sm font-medium text-muted uppercase tracking-wider">Loss Chart</h2>
            <span className="text-xs text-muted ml-auto">{lossHistory.length} data points</span>
          </div>
          {!lossChartCollapsed && (
            <div className="p-4">
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={lossHistory} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
                  <XAxis
                    dataKey="step"
                    stroke="#6b7280"
                    fontSize={11}
                    tickFormatter={(value) => value.toLocaleString()}
                  />
                  <YAxis
                    stroke="#6b7280"
                    fontSize={11}
                    tickFormatter={(value) => value.toFixed(3)}
                    domain={['auto', 'auto']}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1f2937',
                      border: '1px solid #374151',
                      borderRadius: '6px',
                      fontSize: '12px',
                    }}
                    labelStyle={{ color: '#9ca3af' }}
                    formatter={(value: number) => value?.toFixed(4)}
                    labelFormatter={(label) => `Step ${label.toLocaleString()}`}
                  />
                  <Legend
                    wrapperStyle={{ fontSize: '12px' }}
                  />
                  <Line
                    type="monotone"
                    dataKey="loss"
                    stroke="#eab308"
                    strokeWidth={1}
                    dot={false}
                    name="Loss"
                    connectNulls
                  />
                  <Line
                    type="monotone"
                    dataKey="smoothLoss"
                    stroke="#22c55e"
                    strokeWidth={2}
                    dot={false}
                    name="Smooth Loss"
                    connectNulls
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}

      {/* Training Console */}
      <div className="mt-6 bg-black rounded-lg border border-dark-border">
        <div className="px-4 py-3 border-b border-dark-border flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Terminal className="w-4 h-4 text-muted" />
            <h2 className="text-sm font-medium text-muted uppercase tracking-wider">Training Console</h2>
          </div>
          <button
            onClick={clearLogs}
            className="p-1.5 hover:bg-dark-hover rounded text-muted hover:text-white"
            title="Clear logs"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>

        {/* Current Training Progress Summary */}
        {trainingProgress && (
          <div className="px-4 py-3 border-b border-dark-border bg-dark-bg/50">
            <div className="grid grid-cols-6 gap-4 text-sm">
              <div>
                <span className="text-muted">Epoch:</span>
                <span className="text-white ml-2">
                  {trainingProgress.current_epoch}/{trainingProgress.total_epochs}
                </span>
              </div>
              <div>
                <span className="text-muted">Step:</span>
                <span className="text-white ml-2">
                  {trainingProgress.current_step}/{trainingProgress.total_steps}
                </span>
              </div>
              <div>
                <span className="text-muted">Loss:</span>
                <span className="text-yellow-400 ml-2">
                  {trainingProgress.loss?.toFixed(3) || '--'}
                </span>
              </div>
              <div>
                <span className="text-muted">Smooth Loss:</span>
                <span className="text-green-400 ml-2">
                  {trainingProgress.smooth_loss?.toFixed(3) || '--'}
                </span>
              </div>
              <div>
                <span className="text-muted">Speed:</span>
                <span className="text-cyan-400 ml-2">
                  {trainingProgress.samples_per_second ? `${trainingProgress.samples_per_second.toFixed(2)} it/s` : '--'}
                </span>
              </div>
              <div>
                <span className="text-muted">ETA:</span>
                <span className="text-purple-400 ml-2">
                  {trainingProgress.remaining_time || '--'}
                </span>
              </div>
            </div>
            {/* Progress Bar */}
            <div className="mt-3">
              <div className="h-2 bg-dark-bg rounded-full overflow-hidden">
                <div
                  className="h-full bg-primary rounded-full transition-all duration-300"
                  style={{ width: `${(trainingProgress.current_step / trainingProgress.total_steps) * 100}%` }}
                />
              </div>
            </div>
          </div>
        )}

        {/* Console Log Output */}
        <div
          ref={logContainerRef}
          className="h-80 overflow-y-auto p-4 font-mono text-sm bg-[#1a1a1a]"
        >
          {trainingLogs.length === 0 ? (
            <div className="text-muted text-center py-8">
              No training output yet. Start a training job to see progress here.
            </div>
          ) : (
            trainingLogs.map((log) => (
              <LogLine key={log.id} entry={log} />
            ))
          )}
        </div>
      </div>

      {/* Current Training Status */}
      <div className="mt-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm font-medium text-muted uppercase tracking-wider">Training Status</h2>
        </div>
        {isTraining ? (
          <div className="bg-black rounded-lg border border-dark-border p-4">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-2 h-2 bg-success rounded-full animate-pulse" />
              <span className="text-white font-medium">
                {trainingStatus === 'starting' ? 'Starting...' :
                  trainingStatus === 'training' ? 'Training in Progress' :
                    trainingStatus || 'Training in Progress'}
              </span>
              {currentPreset && (
                <span className="text-muted text-sm">({currentPreset})</span>
              )}
            </div>
            {trainingProgress ? (
              <>
                <div className="grid grid-cols-5 gap-4 text-sm mb-3">
                  <div>
                    <span className="text-muted">Progress:</span>
                    <span className="text-white ml-2">
                      {Math.round((trainingProgress.current_step / trainingProgress.total_steps) * 100)}%
                    </span>
                  </div>
                  <div>
                    <span className="text-muted">Epoch Step:</span>
                    <span className="text-cyan-400 ml-2">
                      {(() => {
                        const stepsPerEpoch = Math.ceil(trainingProgress.total_steps / trainingProgress.total_epochs);
                        const epochStep = ((trainingProgress.current_step - 1) % stepsPerEpoch) + 1;
                        return `${epochStep} of ${stepsPerEpoch}`;
                      })()}
                    </span>
                  </div>
                  <div>
                    <span className="text-muted">Epoch:</span>
                    <span className="text-white ml-2">{trainingProgress.current_epoch}/{trainingProgress.total_epochs}</span>
                  </div>
                  <div>
                    <span className="text-muted">Step:</span>
                    <span className="text-white ml-2">{trainingProgress.current_step}/{trainingProgress.total_steps}</span>
                  </div>
                  <div>
                    <span className="text-muted">Elapsed:</span>
                    <span className="text-white ml-2">{trainingProgress.elapsed_time || '--'}</span>
                  </div>
                  <div>
                    <span className="text-muted">ETA:</span>
                    <span className="text-purple-400 ml-2">{trainingProgress.remaining_time || '--'}</span>
                  </div>
                </div>
                <div className="h-2 bg-dark-bg rounded-full overflow-hidden">
                  <div
                    className="h-full bg-success rounded-full transition-all duration-300"
                    style={{ width: `${(trainingProgress.current_step / trainingProgress.total_steps) * 100}%` }}
                  />
                </div>
              </>
            ) : (
              <div className="text-muted text-sm">Waiting for progress updates...</div>
            )}
          </div>
        ) : (
          <div className="text-center text-muted py-12 bg-black rounded-lg border border-dark-border">
            No training in progress
          </div>
        )}
      </div>
    </div>
  );
}

// Log line component matching OneTrainer terminal output style
function LogLine({ entry }: { entry: LogEntry }) {
  const progressWidth = entry.progress || 0;

  if (entry.type === 'step' && entry.details) {
    return (
      <div className="flex items-center gap-2 py-0.5 text-white">
        <span className="text-cyan-400">step:</span>
        <span className="text-white">
          {entry.details.current}/{entry.details.total}
        </span>
        <div className="w-24 h-3 bg-dark-border rounded overflow-hidden">
          <div
            className="h-full bg-cyan-500"
            style={{ width: `${progressWidth}%` }}
          />
        </div>
        <span className="text-muted">{Math.round(entry.progress || 0)}%</span>
        <span className="text-muted">|</span>
        <span className="text-muted">
          [{entry.details.elapsed}&lt;{entry.details.remaining},
        </span>
        <span className="text-white">{entry.details.speed},</span>
        <span className="text-yellow-400">loss={entry.details.loss?.toFixed(3)},</span>
        <span className="text-green-400">smooth loss={entry.details.smoothLoss?.toFixed(3)}</span>
        <span className="text-muted">]</span>
      </div>
    );
  }

  if (entry.type === 'epoch' && entry.details) {
    return (
      <div className="flex items-center gap-2 py-0.5 text-white">
        <span className="text-purple-400">epoch:</span>
        <span className="text-white">
          {entry.details.current}/{entry.details.total}
        </span>
        <div className="w-24 h-3 bg-dark-border rounded overflow-hidden">
          <div
            className="h-full bg-purple-500"
            style={{ width: `${progressWidth}%` }}
          />
        </div>
        <span className="text-muted">{Math.round(entry.progress || 0)}%</span>
        <span className="text-muted">|</span>
        <span className="text-muted">
          [{entry.details.elapsed}&lt;{entry.details.remaining}, {entry.details.speed}]
        </span>
      </div>
    );
  }

  if (entry.type === 'sampling' && entry.details) {
    return (
      <div className="flex items-center gap-2 py-0.5 text-white">
        <span className="text-green-400">sampling:</span>
        <span className="text-white">{Math.round(entry.progress || 0)}%</span>
        <div className="w-32 h-3 bg-dark-border rounded overflow-hidden">
          <div
            className="h-full bg-green-500"
            style={{ width: `${progressWidth}%` }}
          />
        </div>
        <span className="text-muted">|</span>
        <span className="text-white">
          {entry.details.current}/{entry.details.total}
        </span>
        <span className="text-muted">
          [{entry.details.elapsed}&lt;{entry.details.remaining}, {entry.details.speed}]
        </span>
      </div>
    );
  }

  if (entry.type === 'backup') {
    return (
      <div className="py-0.5 text-yellow-300">
        {entry.message}
      </div>
    );
  }

  if (entry.type === 'error') {
    return (
      <div className="py-0.5 text-red-400">
        {entry.message}
      </div>
    );
  }

  // Default info type
  return (
    <div className="py-0.5 text-gray-300">
      {entry.message}
    </div>
  );
}
