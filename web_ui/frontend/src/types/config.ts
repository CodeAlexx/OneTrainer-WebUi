export interface TrainingConfig {
  // General
  training_method: string;
  model_type: string;
  debug_mode: boolean;
  debug_dir: string;
  workspace_dir: string;
  cache_dir: string;

  // Model paths
  base_model_name: string;
  output_model_destination: string;
  output_model_format: string[];

  // Training parameters
  learning_rate: number;
  learning_rate_scheduler: string;
  learning_rate_warmup_steps: number;
  learning_rate_min_factor: number;
  learning_rate_cycles: number;
  train_batch_size: number;
  batch_size: number;
  gradient_accumulation_steps: number;
  max_epochs: number;
  epochs: number;
  resolution: string | number;

  // Optimizer
  optimizer: string;
  optimizer_defaults?: Record<string, any>;

  // EMA & Model
  ema: string;
  ema_decay: number;
  gradient_checkpointing: string;
  train_dtype: string;
  fallback_train_dtype: string;

  // Bucketing
  aspect_ratio_bucketing: boolean;
  bucket_quantization: number;
  aspect_tolerance: number;
  bucket_repeat_small: boolean;
  bucket_log_dropped: boolean;
  bucket_preset: string;
  bucket_balancing: string;
  bucket_min_size: number;
  bucket_merge_threshold: number;
  bucket_max_per_batch: number;
  bucket_custom_aspects: string;
  latent_caching: boolean;

  // Data
  concept_file_name: string;
  concepts?: Concept[];

  // Sampling
  sample_definition_file_name: string;
  samples?: Sample[];

  // LoRA specific
  peft_type: string;
  lora_rank: number;
  lora_alpha: number;
  lora_weight_dtype: string;

  // Diffusion-4K settings
  diffusion_4k_enabled: boolean;
  diffusion_4k_wavelet_loss_weight: number;
  diffusion_4k_wavelet_type: string;

  // Allow any additional fields from config
  [key: string]: any;
}

export interface Concept {
  name: string;
  path: string;
  prompt_source: string;
  enable_crop_jitter: boolean;
  enable_random_flip: boolean;
  random_rotate: boolean;
  random_brightness: boolean;
  random_contrast: boolean;
  random_saturation: boolean;
  random_hue: boolean;
}

export interface Sample {
  prompt: string;
  negative_prompt: string;
  seed: number;
  random_seed: boolean;
  width: number;
  height: number;
  steps: number;
}

export interface TrainingStatus {
  status: 'idle' | 'running' | 'paused' | 'stopped' | 'error';
  current_epoch?: number;
  total_epochs?: number;
  current_step?: number;
  total_steps?: number;
  loss?: number;
  learning_rate?: number;
  epoch_time_remaining?: number;
  total_time_remaining?: number;
  samples_per_second?: number;
  error_message?: string;
}

export interface Preset {
  name: string;
  path: string;
  modified: string;
}
