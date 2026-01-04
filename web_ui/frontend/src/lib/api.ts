import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
  timeout: 3600000, // 1 hour (needed for large model downloads)
  headers: {
    'Content-Type': 'application/json',
  },
});

// Types
export interface GPUInfo {
  index: number;
  name: string;
  memory_total: number;
  memory_allocated: number;
  memory_reserved: number;
  memory_free: number;
  utilization: number | null;
  temperature: number | null;
  fan_speed: number | null;
  power_draw: number | null;
  power_limit: number | null;
}

export interface SystemInfo {
  gpus: GPUInfo[];
  cpu_count: number;
  memory_total: number;
  memory_available: number;
  python_version: string;
  torch_version: string;
  cuda_available: boolean;
  cuda_version: string;
}

export interface TrainingStatus {
  is_training: boolean;
  status: string;
  progress: {
    epoch: number;
    epoch_step: number;
    global_step: number;
  } | null;
  max_step: number;
  max_epoch: number;
  error: string | null;
}

export interface PresetInfo {
  name: string;
  path: string;
  model_type?: string;
}

export interface TrainingProgress {
  current_epoch: number;
  total_epochs: number;
  current_step: number;
  total_steps: number;
  loss: number | null;
  smooth_loss: number | null;
  learning_rate: number | null;
  samples_per_second: number | null;
  elapsed_time: string | null;
  remaining_time: string | null;
}

// System API
export const systemApi = {
  getInfo: () => api.get<SystemInfo>('/system/info'),
  getModels: () => api.get('/system/models'),
};

// Training API
export const trainingApi = {
  getStatus: () => api.get<TrainingStatus>('/training/status'),
  getProgress: () => api.get('/training/progress'),
  start: (configPath: string) => api.post('/training/start', { config_path: configPath }),
  stop: () => api.post('/training/stop'),
  pause: () => api.post('/training/pause'),
  resume: () => api.post('/training/resume'),
};

// Config API
export const configApi = {
  getPresets: (configDir?: string) => api.get<{ presets: PresetInfo[] }>('/config/presets', { params: { config_dir: configDir } }),
  loadPreset: (name: string, configDir?: string) => api.get(`/config/presets/${name}`, { params: { config_dir: configDir } }),
  savePreset: (name: string, config: any) => api.post(`/config/presets/${encodeURIComponent(name)}`, { config }),
  getCurrent: () => api.get('/config/current'),
  updateCurrent: (config: any) => api.put('/config/current', config),
  validate: (config: any) => api.post<{ valid: boolean; errors: string[]; warnings: string[] }>('/config/validate', { config }),
  saveTemp: (config: any) => api.post<{ path: string; success: boolean }>('/config/save-temp', { config }),
  getConceptsFile: (filePath: string) => api.get<{ concepts: any[]; file_path: string }>('/config/concepts-file', { params: { file_path: filePath } }),
};

// Filesystem API
export const filesystemApi = {
  browse: (path: string, extensions?: string, includeHidden?: boolean) =>
    api.get('/filesystem/browse', { params: { path, extensions, include_hidden: includeHidden } }),
  scan: (path: string, recursive?: boolean, includeImensions?: boolean, maxFiles?: number) =>
    api.get('/filesystem/scan', { params: { path, recursive, include_dimensions: includeImensions, max_files: maxFiles } }),
  validatePath: (path: string) =>
    api.get('/filesystem/validate-path', { params: { path } }),
};

// Samples API
export const samplesApi = {
  list: (samplesDir?: string, limit?: number) =>
    api.get('/samples', { params: { samples_dir: samplesDir, limit } }),
  get: (sampleId: string, samplesDir?: string) =>
    api.get(`/samples/${sampleId}`, { params: { samples_dir: samplesDir } }),
  generateDefault: () => api.post('/samples/generate/default'),
  generate: (params: any) => api.post('/samples/generate', params),
  // Tree browsing for samples view
  getTree: (samplesDir?: string) =>
    api.get<{ tree: TreeNode[]; root_path: string }>('/samples/tree', { params: { samples_dir: samplesDir } }),
  listImages: (path: string) =>
    api.get<{ images: SampleImage[]; count: number; directory: string }>('/samples/images', { params: { path } }),
};

export interface TreeNode {
  name: string;
  path: string;
  type: 'directory' | 'prompt' | 'image';
  children?: TreeNode[];
  image_count?: number;
}

export interface SampleImage {
  id: string;
  name: string;
  path: string;
  timestamp: string;
}

// Queue API
export interface QueuedJob {
  id: string;
  name: string;
  config_path: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  created_at: string | null;
  started_at: string | null;
  completed_at: string | null;
  error: string | null;
  progress: Record<string, any> | null;
}

export const queueApi = {
  list: () => api.get<{ jobs: QueuedJob[]; count: number; current_job: QueuedJob | null }>('/queue'),
  history: (limit?: number) => api.get<{ jobs: QueuedJob[]; count: number }>('/queue/history', { params: { limit } }),
  add: (name: string, configPath: string) => api.post<QueuedJob>('/queue', { name, config_path: configPath }),
  remove: (jobId: string) => api.delete(`/queue/${jobId}`),
  move: (jobId: string, position: number) => api.post(`/queue/${jobId}/move`, { position }),
  cancel: (jobId: string) => api.post(`/queue/${jobId}/cancel`),
  startNext: () => api.post('/queue/start-next'),
  clearHistory: () => api.delete('/queue/history'),
};

// Inference API
export interface InferenceState {
  model_loaded: boolean;
  model_path: string | null;
  model_type: string | null;
  lora_paths: string[];
  is_generating: boolean;
  generation_progress: number;
}

export interface GeneratedImage {
  id: string;
  path: string;
  prompt: string;
  negative_prompt: string;
  width: number;
  height: number;
  steps: number;
  guidance_scale: number;
  seed: number;
  created_at: string;
}

export interface GenerateParams {
  prompt: string;
  negative_prompt?: string;
  width?: number;
  height?: number;
  steps?: number;
  guidance_scale?: number;
  seed?: number;
  batch_size?: number;
  // Generation mode
  mode?: 'txt2img' | 'img2img' | 'inpainting' | 'edit' | 'video';
  // img2img / inpainting
  init_image_path?: string;
  mask_image_path?: string;
  strength?: number;
  // Video generation
  num_frames?: number;
  fps?: number;
  // Edit mode
  edit_instruction?: string;
  // Multi-image input (FLUX 2)
  reference_images?: string[];
}

// Plugin types
export interface PluginUIElement {
  id: string;
  label: string;
  type: string;
  default: any;
  options?: Record<string, any>;
}

export interface PluginInfo {
  name: string;
  display_name: string;
  version: string;
  description: string;
  author: string;
  type: string;
  enabled: boolean;
  loaded: boolean;
  supported_models: string[];
  ui_elements: PluginUIElement[];
}

export interface PluginsListResponse {
  plugins: PluginInfo[];
  available: string[];
}

// Plugins API
export const pluginsApi = {
  list: () => api.get<PluginsListResponse>('/plugins/'),
  load: (pluginName: string) => api.post('/plugins/load', { plugin_name: pluginName }),
  unload: (pluginName: string) => api.post('/plugins/unload', { plugin_name: pluginName }),
  enable: (pluginName: string) => api.post('/plugins/enable', { plugin_name: pluginName }),
  disable: (pluginName: string) => api.post('/plugins/disable', { plugin_name: pluginName }),
  loadAll: () => api.post('/plugins/load-all'),
  getInfo: (pluginName: string) => api.get<PluginInfo>(`/plugins/${pluginName}`),
};

export const inferenceApi = {
  getStatus: () => api.get<InferenceState>('/inference/status'),
  loadModel: (modelPath: string, modelType: string, loraPaths?: string[]) =>
    api.post('/inference/load', { model_path: modelPath, model_type: modelType, lora_paths: loraPaths }),
  unloadModel: () => api.post('/inference/unload'),
  generate: (params: GenerateParams) => api.post<{ success: boolean; image: GeneratedImage }>('/inference/generate', params),
  cancelGeneration: () => api.post('/inference/cancel'),
  getGallery: (limit?: number) => api.get<{ images: GeneratedImage[]; count: number }>('/inference/gallery', { params: { limit } }),
  getImage: (imageId: string) => `/api/inference/gallery/${imageId}`,
  deleteImage: (imageId: string) => api.delete(`/inference/gallery/${imageId}`),
  clearGallery: () => api.delete('/inference/gallery'),
};

// TensorBoard API
export interface TensorBoardStatus {
  running: boolean;
  port: number;
  logdir: string | null;
  url: string | null;
}

export interface TensorBoardLog {
  name: string;
  path: string;
  event_count: number;
  modified: number;
}

export const tensorboardApi = {
  getStatus: () => api.get<TensorBoardStatus>('/tensorboard/status'),
  start: (logdir?: string, port: number = 6006) =>
    api.post<TensorBoardStatus>('/tensorboard/start', { logdir, port }),
  stop: () => api.post<TensorBoardStatus>('/tensorboard/stop'),
  getLogs: () => api.get<{ logs: TensorBoardLog[]; workspace: string }>('/tensorboard/logs'),
};

// Caption API
export interface CaptionState {
  loaded: boolean;
  model_id: string | null;
  device: string;
  vram_used: string;
  dtype: string;
}

export interface CaptionBatchStatus {
  active: boolean;
  stats: {
    processed: number;
    skipped: number;
    failed: number;
  };
  current_file: string | null;
  last_caption: string | null;
  progress?: number;
}

export const captionApi = {
  getState: () => api.get<CaptionState>('/caption/state'),
  loadModel: (modelId: string, quantization: string, attnImpl: string) =>
    api.post<CaptionState>('/caption/load', { model_id: modelId, quantization, attn_impl: attnImpl }),
  unloadModel: () => api.post('/caption/unload'),
  generate: (mediaPath: string, prompt: string, maxTokens: number = 128, resolutionMode: string = 'auto') =>
    api.post<{ caption: string }>('/caption/generate', { media_path: mediaPath, prompt, max_tokens: maxTokens, resolution_mode: resolutionMode }),
  startBatch: (params: any) => api.post('/caption/batch/start', params),
  stopBatch: () => api.post('/caption/batch/stop'),
  getBatchStatus: () => api.get<CaptionBatchStatus>('/caption/batch/status'),
};

// WebSocket connection for real-time updates
export class TrainingWebSocket {
  private ws: WebSocket | null = null;
  private reconnectInterval: number = 3000;
  private listeners: Map<string, Set<(data: any) => void>> = new Map();
  private manualDisconnect: boolean = false;

  connect() {
    // Use the same host as the API - in development the backend is on port 8000
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    // Extract the base URL from the API config (handles both dev proxy and direct)
    const apiHost = window.location.hostname;
    const apiPort = window.location.port === '3000' || window.location.port === '3001' ? '8000' : window.location.port;
    const wsUrl = `${protocol}//${apiHost}:${apiPort}/ws`;

    this.manualDisconnect = false;
    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      console.log('WebSocket connected');
      this.emit('connected', {});
    };

    this.ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        this.emit(message.type, message.data);
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e);
      }
    };

    this.ws.onclose = () => {
      console.log('WebSocket disconnected');
      this.emit('disconnected', {});
      // Only auto-reconnect if not manually disconnected
      if (!this.manualDisconnect) {
        setTimeout(() => this.connect(), this.reconnectInterval);
      }
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }

  disconnect() {
    this.manualDisconnect = true;
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  reconnect() {
    console.log('Manual reconnect requested');
    this.disconnect();
    setTimeout(() => {
      this.connect();
    }, 100);
  }

  on(event: string, callback: (data: any) => void) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(callback);
  }

  off(event: string, callback: (data: any) => void) {
    this.listeners.get(event)?.delete(callback);
  }

  private emit(event: string, data: any) {
    this.listeners.get(event)?.forEach(callback => callback(data));
  }

  send(message: any) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    }
  }
}

export const trainingWs = new TrainingWebSocket();

// Database API
export interface DbStatus {
  enabled: boolean;
  initialized: boolean;
  path: string | null;
  counts: {
    presets: number;
    concepts: number;
    samples: number;
    training_runs: number;
  } | null;
  error?: string;
}

export interface DbPreset {
  id: number;
  name: string;
  description: string | null;
  model_type: string;
  training_method: string;
  base_model_name: string | null;
  config: Record<string, any>;
  config_version: number;
  is_builtin: boolean;
  is_favorite: boolean;
  tags: string[];
  created_at: string;
  updated_at: string;
}

export interface DbVersion {
  id: number;
  entity_type: string;
  entity_id: number;
  version: number;
  data: Record<string, any>;
  change_type: string;
  change_description: string | null;
  changed_fields: string[];
  created_at: string;
  created_by: string | null;
}

export interface DbConcept {
  id: number;
  name: string;
  path: string;
  concept_type: string;
  enabled: boolean;
  config: Record<string, any>;
  created_at: string;
  updated_at: string;
}

export interface DbSample {
  id: number;
  name: string | null;
  prompt: string;
  negative_prompt: string | null;
  width: number;
  height: number;
  seed: number;
  config: Record<string, any>;
  enabled: boolean;
  created_at: string;
  updated_at: string;
}

export interface DbTrainingRun {
  id: number;
  preset_id: number | null;
  preset_name: string | null;
  run_name: string | null;
  status: string;
  current_epoch: number;
  total_epochs: number | null;
  current_step: number;
  total_steps: number | null;
  final_loss: number | null;
  started_at: string | null;
  completed_at: string | null;
  total_duration_seconds: number | null;
  error_message: string | null;
  created_at: string;
}

export interface MigrationResult {
  migrated: Array<{ name: string; id: number }>;
  skipped: Array<{ name: string; reason: string }>;
  errors: Array<{ file: string; error: string }>;
  summary: {
    migrated_count: number;
    skipped_count: number;
    error_count: number;
  };
}

export const databaseApi = {
  // Status & init
  getStatus: () => api.get<DbStatus>('/db/status'),
  init: () => api.post<{ success: boolean; message: string }>('/db/init'),

  // Migration
  migrate: (includePresets = true, includeConcepts = true, includeSamples = true) =>
    api.post<{ presets?: MigrationResult; concepts?: MigrationResult; samples?: MigrationResult }>(
      '/db/migrate',
      null,
      { params: { include_presets: includePresets, include_concepts: includeConcepts, include_samples: includeSamples } }
    ),

  // Presets
  listPresets: (includeBuiltin = true, favoritesOnly = false) =>
    api.get<DbPreset[]>('/db/presets', { params: { include_builtin: includeBuiltin, favorites_only: favoritesOnly } }),
  getPreset: (id: number) => api.get<DbPreset>(`/db/presets/${id}`),
  createPreset: (name: string, config: Record<string, any>, description?: string) =>
    api.post<DbPreset>('/db/presets', config, { params: { name, description } }),
  updatePreset: (id: number, config: Record<string, any>, description?: string) =>
    api.put<DbPreset>(`/db/presets/${id}`, config, { params: { description } }),
  deletePreset: (id: number, hard = false) =>
    api.delete<{ success: boolean; message: string }>(`/db/presets/${id}`, { params: { hard } }),

  // Version history
  getPresetVersions: (presetId: number, limit = 50) =>
    api.get<DbVersion[]>(`/db/presets/${presetId}/versions`, { params: { limit } }),
  rollbackPreset: (presetId: number, version: number) =>
    api.post<DbPreset>(`/db/presets/${presetId}/rollback/${version}`),

  // Concepts
  listConcepts: (enabledOnly = false) =>
    api.get<DbConcept[]>('/db/concepts', { params: { enabled_only: enabledOnly } }),
  createConcept: (config: Record<string, any>) => api.post<DbConcept>('/db/concepts', config),
  updateConcept: (id: string, config: Record<string, any>) =>
    api.put<DbConcept>(`/db/concepts/${id.replace('db-', '')}`, config),
  deleteConcept: (id: string) => api.delete<{ success: boolean; message: string }>(`/db/concepts/${id.replace('db-', '')}`),

  // Samples
  listSamples: (enabledOnly = false) =>
    api.get<DbSample[]>('/db/samples', { params: { enabled_only: enabledOnly } }),
  createSample: (config: Record<string, any>, name?: string) =>
    api.post<DbSample>('/db/samples', config, { params: { name } }),

  // Training runs
  listTrainingRuns: (limit = 20, statusFilter?: string) =>
    api.get<DbTrainingRun[]>('/db/training-runs', { params: { limit, status_filter: statusFilter } }),
  getTrainingStats: () =>
    api.get<{ total_runs: number; status_counts: Record<string, number>; average_duration_seconds: number | null }>('/db/training-runs/stats'),

  // Export
  exportPresets: (outputDir: string, includeBuiltin = true) =>
    api.post<{ exported: Array<{ name: string; file: string }>; errors: any[] }>('/db/export/presets', null, { params: { output_dir: outputDir, include_builtin: includeBuiltin } }),
};

export default api;
