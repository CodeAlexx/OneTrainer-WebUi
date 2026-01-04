import { useState, useEffect } from 'react';
import { MoreHorizontal, X } from 'lucide-react';
import { useConfigStore } from '../../stores/configStore';
import { SamplesBrowserView } from './SamplesBrowserView';
import { LoRAView } from './LoRAView';

type TabType = 'overview' | 'samples' | 'config' | 'parameters' | 'lora' | 'diffusion4k' | 'buckets';

const OPTIMIZERS = [
  'ADAGRAD', 'ADAGRAD_8BIT', 'ADAM', 'ADAM_8BIT', 'ADAMW', 'ADAMW_8BIT', 'ADAMW_ADV',
  'AdEMAMix', 'AdEMAMix_8BIT', 'SIMPLIFIED_AdEMAMix', 'ADOPT', 'ADOPT_ADV',
  'LAMB', 'LAMB_8BIT', 'LARS', 'LARS_8BIT', 'LION', 'LION_8BIT', 'LION_ADV',
  'RMSPROP', 'RMSPROP_8BIT', 'SGD', 'SGD_8BIT', 'SCHEDULE_FREE_ADAMW', 'SCHEDULE_FREE_SGD',
  'DADAPT_ADA_GRAD', 'DADAPT_ADAM', 'DADAPT_ADAN', 'DADAPT_LION', 'DADAPT_SGD',
  'PRODIGY', 'PRODIGY_PLUS_SCHEDULE_FREE', 'PRODIGY_ADV', 'LION_PRODIGY_ADV',
  'ADAFACTOR', 'CAME', 'CAME_8BIT', 'MUON', 'MUON_ADV', 'ADAMUON_ADV',
  'ADABELIEF', 'TIGER', 'AIDA', 'YOGI',
];
const SCHEDULERS = ['CONSTANT', 'LINEAR', 'COSINE', 'COSINE_WITH_RESTARTS', 'COSINE_WITH_HARD_RESTARTS', 'REX', 'ADAFACTOR', 'CUSTOM'];
const EMA_MODES = ['OFF', 'CPU', 'GPU'];
const GRADIENT_CHECKPOINTING = ['OFF', 'ON', 'CPU_OFFLOADED'];
const TRAIN_DTYPES = ['FLOAT_32', 'FLOAT_16', 'BFLOAT_16', 'TFLOAT_32'];
const FALLBACK_DTYPES = ['FLOAT_32', 'BFLOAT_16'];
const LR_SCALERS = ['NONE', 'SQRT_ACCUM', 'LINEAR_ACCUM', 'SQRT_BATCH_LINEAR_ACCUM'];
const TIMESTEP_DIST = ['UNIFORM', 'SIGMOID', 'LOGIT_NORMAL', 'BETA', 'FLUX_SHIFT'];
const LOSS_WEIGHTS = ['CONSTANT', 'MIN_SNR_GAMMA', 'P2', 'DEBIASED', 'RESCALE_ZERO_TERMINAL_SNR', 'SIGMOID'];
const LOSS_SCALERS = ['NONE', 'BATCH_SIZE', 'ACCUM_STEPS', 'BATCH_AND_ACCUM'];
const TIME_UNITS = ['NEVER', 'EPOCH', 'STEP', 'SECOND', 'MINUTE'];
const LAYER_PRESETS = ['full', 'unet_only', 'text_encoder_only', 'attention_only', 'feedforward_only'];

// Bucket presets with common aspect ratios
const BUCKET_PRESETS = {
  default: [
    { w: 1, h: 1 }, { w: 4, h: 5 }, { w: 2, h: 3 }, { w: 4, h: 7 }, { w: 1, h: 2 },
    { w: 2, h: 5 }, { w: 1, h: 3 }, { w: 2, h: 7 }, { w: 1, h: 4 },
  ],
  photo: [
    { w: 1, h: 1 }, { w: 3, h: 2 }, { w: 4, h: 3 }, { w: 16, h: 9 }, { w: 5, h: 4 },
    { w: 2, h: 3 }, { w: 3, h: 4 }, { w: 9, h: 16 }, { w: 4, h: 5 },
  ],
  video: [
    { w: 16, h: 9 }, { w: 21, h: 9 }, { w: 4, h: 3 }, { w: 1, h: 1 },
    { w: 9, h: 16 }, { w: 9, h: 21 }, { w: 3, h: 4 },
  ],
  widescreen: [
    { w: 16, h: 9 }, { w: 21, h: 9 }, { w: 32, h: 9 }, { w: 2, h: 1 },
    { w: 9, h: 16 }, { w: 9, h: 21 }, { w: 9, h: 32 }, { w: 1, h: 2 },
  ],
  square: [{ w: 1, h: 1 }],
};

const QUANTIZATION_VALUES = [8, 16, 32, 64, 128];

export function TrainingView() {
  const [activeTab, setActiveTab] = useState<TabType>('parameters');
  const { config, currentPreset, updateConfig } = useConfigStore();

  // Base Parameters
  const [optimizer, setOptimizer] = useState('ADAMW');
  const [scheduler, setScheduler] = useState('CONSTANT');
  const [learningRate, setLearningRate] = useState('3e-4');
  const [warmupSteps, setWarmupSteps] = useState(200);
  const [lrMinFactor, setLrMinFactor] = useState(0.0);
  const [lrCycles, setLrCycles] = useState(1.0);
  const [epochs, setEpochs] = useState(100);
  const [batchSize, setBatchSize] = useState(1);
  const [accumSteps, setAccumSteps] = useState(1);
  const [lrScaler, setLrScaler] = useState('NONE');
  const [clipGradNorm, setClipGradNorm] = useState(1.0);

  // EMA & Model
  const [ema, setEma] = useState('OFF');
  const [emaDecay, setEmaDecay] = useState(0.999);
  const [emaUpdateInterval, setEmaUpdateInterval] = useState(5);
  const [gradCheckpoint, setGradCheckpoint] = useState('ON');
  const [layerOffload, setLayerOffload] = useState(0.0);
  const [musubiBlocksToSwap, setMusubiBlocksToSwap] = useState(0);
  const [blocksToSwap, setBlocksToSwap] = useState(0);
  const [trainDtype, setTrainDtype] = useState('BFLOAT_16');
  const [fallbackDtype, setFallbackDtype] = useState('BFLOAT_16');
  const [autocastCache, setAutocastCache] = useState(true);
  const [resolution, setResolution] = useState('1024');
  const [frames, setFrames] = useState('25');
  const [circularPadding, setCircularPadding] = useState(false);
  const [enableAsyncOffload, setEnableAsyncOffload] = useState(true);
  const [enableActivationOffload, setEnableActivationOffload] = useState(true);
  const [compile, setCompile] = useState(false);
  const [onlyCache, setOnlyCache] = useState(false);

  // Device & Threads
  const [dataloaderThreads, setDataloaderThreads] = useState(2);
  const [trainDevice, setTrainDevice] = useState('cuda');
  const [tempDevice, setTempDevice] = useState('cpu');

  // Multi-GPU
  const [multiGpu, setMultiGpu] = useState(false);
  const [deviceIndexes, setDeviceIndexes] = useState('');

  // Layer Filter
  const [layerFilter, setLayerFilter] = useState('');
  const [layerFilterPreset, setLayerFilterPreset] = useState('full');
  const [layerFilterRegex, setLayerFilterRegex] = useState(false);

  // Custom Scheduler
  const [customScheduler, setCustomScheduler] = useState('');

  // Backup Settings
  const [backupAfter, setBackupAfter] = useState(30);
  const [backupAfterUnit, setBackupAfterUnit] = useState('MINUTE');
  const [rollingBackup, setRollingBackup] = useState(false);
  const [rollingBackupCount, setRollingBackupCount] = useState(3);
  const [backupBeforeSave, setBackupBeforeSave] = useState(true);
  const [saveEvery, setSaveEvery] = useState(0);
  const [saveEveryUnit, setSaveEveryUnit] = useState('NEVER');
  const [saveFilenamePrefix, setSaveFilenamePrefix] = useState('');

  // Resume Training - accessed via config store, not local state

  // Text Encoder
  const [trainTE, setTrainTE] = useState(false);
  const [teDropout, setTeDropout] = useState(0.0);
  const [teStopAfter, setTeStopAfter] = useState('');
  const [teLR, setTeLR] = useState('');
  const [clipSkip, setClipSkip] = useState(0);

  // UNet/Transformer
  const [trainUnet, setTrainUnet] = useState(true);
  const [unetStopAfter, setUnetStopAfter] = useState('');
  const [unetLR, setUnetLR] = useState('');
  const [rescaleNoise, setRescaleNoise] = useState(false);
  const [forceAttentionMask, setForceAttentionMask] = useState(false);
  const [guidanceScale, setGuidanceScale] = useState(1.0);

  // Noise
  const [offsetNoise, setOffsetNoise] = useState(0.0);
  const [perturbNoise, setPerturbNoise] = useState(0.0);
  const [timestepDist, setTimestepDist] = useState('UNIFORM');
  const [minNoise, setMinNoise] = useState(0.0);
  const [maxNoise, setMaxNoise] = useState(1.0);
  const [noiseWeight, setNoiseWeight] = useState(0.0);
  const [noiseBias, setNoiseBias] = useState(0.0);
  const [timestepShift, setTimestepShift] = useState(0.0);
  const [generalizedOffsetNoise, setGeneralizedOffsetNoise] = useState(false);
  const [forceVPred, setForceVPred] = useState(false);
  const [forceEpsPred, setForceEpsPred] = useState(false);
  const [dynamicTimestepShift, setDynamicTimestepShift] = useState(false);

  // Masked Training
  const [maskedTraining, setMaskedTraining] = useState(false);
  const [unmaskedProb, setUnmaskedProb] = useState(0.1);
  const [unmaskedWeight, setUnmaskedWeight] = useState(0.1);
  const [normalizeMaskLoss, setNormalizeMaskLoss] = useState(false);
  const [maskedPriorWeight, setMaskedPriorWeight] = useState(0.0);
  const [customConditioningImage, setCustomConditioningImage] = useState(false);

  // Loss
  const [mseStrength, setMseStrength] = useState(1.0);
  const [maeStrength, setMaeStrength] = useState(0.0);
  const [logCoshStrength, setLogCoshStrength] = useState(0.0);
  const [huberStrength, setHuberStrength] = useState(0.0);
  const [huberDelta, setHuberDelta] = useState(1.0);
  const [vbLossStrength, setVbLossStrength] = useState(1.0);
  const [lossWeightFn, setLossWeightFn] = useState('CONSTANT');
  const [gamma, setGamma] = useState(5.0);
  const [lossScaler, setLossScaler] = useState('NONE');
  const [dropoutProb, setDropoutProb] = useState(0.0);

  // Embeddings
  const [embLR, setEmbLR] = useState('');
  const [preserveEmbNorm, setPreserveEmbNorm] = useState(false);

  // LoRA settings are managed in the dedicated LoRA view

  // Diffusion-4K settings
  const [diffusion4kEnabled, setDiffusion4kEnabled] = useState(false);
  const [diffusion4kWaveletWeight, setDiffusion4kWaveletWeight] = useState(1.0);
  const [diffusion4kWaveletType, setDiffusion4kWaveletType] = useState('haar');

  // Bucketing
  const [bucketingEnabled, setBucketingEnabled] = useState(true);
  const [bucketPreset, setBucketPreset] = useState<keyof typeof BUCKET_PRESETS>('default');
  const [customAspects, setCustomAspects] = useState<{ w: number; h: number }[]>([]);
  const [useCustomAspects, setUseCustomAspects] = useState(false);
  const [bucketQuantization, setBucketQuantization] = useState(64);
  const [minBucketSize, setMinBucketSize] = useState(1);
  const [bucketMergeThreshold, setBucketMergeThreshold] = useState(0);
  const [repeatSmallBuckets, setRepeatSmallBuckets] = useState(true);
  const [aspectTolerance, setAspectTolerance] = useState(0.15);
  const [bucketBalancing, setBucketBalancing] = useState('OFF');
  const [logDroppedSamples, setLogDroppedSamples] = useState(true);
  const [maxBucketsPerBatch, setMaxBucketsPerBatch] = useState(0);

  // Settings modal
  const [settingsModal, setSettingsModal] = useState<string | null>(null);
  const openSettings = (key: string) => setSettingsModal(key);
  const closeSettings = () => setSettingsModal(null);

  // Advanced optimizer params
  const [optWeightDecay, setOptWeightDecay] = useState(0.01);
  const [optEps, setOptEps] = useState(1e-8);
  const [optBeta1, setOptBeta1] = useState(0.9);
  const [optBeta2, setOptBeta2] = useState(0.999);
  const [optBeta3, setOptBeta3] = useState(0.0);
  const [optMomentum, setOptMomentum] = useState(0.0);
  const [optDampening, setOptDampening] = useState(0.0);
  const [optFused, setOptFused] = useState(false);
  const [optStochasticRounding, setOptStochasticRounding] = useState(true);
  const [optAmsgrad, setOptAmsgrad] = useState(false);
  const [optNesterov, setOptNesterov] = useState(false);
  const [optForeach, setOptForeach] = useState(false);
  const [optDecouple, setOptDecouple] = useState(true);
  const [optUseBiasCorrection, setOptUseBiasCorrection] = useState(true);
  const [optAdamWMode, setOptAdamWMode] = useState(true);
  const [optCautious, setOptCautious] = useState(false);
  const [optD0, setOptD0] = useState(1e-6);
  const [optDCoef, setOptDCoef] = useState(1.0);
  const [optGrowthRate, setOptGrowthRate] = useState(1000000000.0);
  const [optRelativeStep, setOptRelativeStep] = useState(false);
  const [optSafeguardWarmup, setOptSafeguardWarmup] = useState(false);
  const [optClipThreshold, setOptClipThreshold] = useState(1.0);

  // Helper to update nested config properties (e.g., optimizer.optimizer)
  const updateNestedConfig = (path: string, value: any) => {
    const keys = path.split('.');
    if (keys.length === 1) {
      updateConfig({ [keys[0]]: value });
    } else {
      // For nested properties like optimizer.optimizer
      const [parent, child] = keys;
      const currentParent = (config as any)?.[parent] || {};
      updateConfig({ [parent]: { ...currentParent, [child]: value } });
    }
  };

  // Sync with store config when it changes
  useEffect(() => {
    if (config && Object.keys(config).length > 0) {
      const c = config as any;  // Config from backend has many more fields
      // Base parameters - handle nested optimizer object
      if (c.optimizer?.optimizer) setOptimizer(c.optimizer.optimizer);
      else if (typeof c.optimizer === 'string') setOptimizer(c.optimizer);
      if (c.learning_rate_scheduler) setScheduler(c.learning_rate_scheduler);
      if (c.learning_rate !== undefined) setLearningRate(String(c.learning_rate));
      if (c.learning_rate_warmup_steps !== undefined) setWarmupSteps(c.learning_rate_warmup_steps);
      if (c.learning_rate_min_factor !== undefined) setLrMinFactor(c.learning_rate_min_factor);
      if (c.learning_rate_cycles !== undefined) setLrCycles(c.learning_rate_cycles);
      if (c.epochs !== undefined) setEpochs(c.epochs);
      if (c.batch_size !== undefined) setBatchSize(c.batch_size);
      if (c.gradient_accumulation_steps !== undefined) setAccumSteps(c.gradient_accumulation_steps);
      if (c.learning_rate_scaler) setLrScaler(c.learning_rate_scaler);

      // EMA & Model
      if (c.ema) setEma(c.ema);
      if (c.ema_decay !== undefined) setEmaDecay(c.ema_decay);
      if (c.gradient_checkpointing) setGradCheckpoint(c.gradient_checkpointing);
      if (c.musubi_blocks_to_swap !== undefined) setMusubiBlocksToSwap(c.musubi_blocks_to_swap);
      if (c.blocks_to_swap !== undefined) setBlocksToSwap(c.blocks_to_swap);
      if (c.train_dtype) setTrainDtype(c.train_dtype);
      if (c.fallback_train_dtype) setFallbackDtype(c.fallback_train_dtype);
      if (c.resolution !== undefined) setResolution(String(c.resolution));
      if (c.frames !== undefined) setFrames(String(c.frames));
      if (c.enable_async_offloading !== undefined) setEnableAsyncOffload(c.enable_async_offloading);
      if (c.enable_activation_offloading !== undefined) setEnableActivationOffload(c.enable_activation_offloading);
      if (c.compile !== undefined) setCompile(c.compile);
      if (c.only_cache !== undefined) setOnlyCache(c.only_cache);
      if (c.dataloader_threads !== undefined) setDataloaderThreads(c.dataloader_threads);
      if (c.train_device) setTrainDevice(c.train_device);
      if (c.temp_device) setTempDevice(c.temp_device);
      if (c.multi_gpu !== undefined) setMultiGpu(c.multi_gpu);
      if (c.device_indexes) setDeviceIndexes(c.device_indexes);
      if (c.layer_filter) setLayerFilter(c.layer_filter);
      if (c.layer_filter_preset) setLayerFilterPreset(c.layer_filter_preset);
      if (c.layer_filter_regex !== undefined) setLayerFilterRegex(c.layer_filter_regex);
      if (c.custom_learning_rate_scheduler) setCustomScheduler(c.custom_learning_rate_scheduler);
      if (c.dropout_probability !== undefined) setDropoutProb(c.dropout_probability);
      if (c.vb_loss_strength !== undefined) setVbLossStrength(c.vb_loss_strength);
      if (c.offset_noise_weight !== undefined) setOffsetNoise(c.offset_noise_weight);
      if (c.perturbation_noise_weight !== undefined) setPerturbNoise(c.perturbation_noise_weight);
      if (c.generalized_offset_noise !== undefined) setGeneralizedOffsetNoise(c.generalized_offset_noise);
      if (c.force_v_prediction !== undefined) setForceVPred(c.force_v_prediction);
      if (c.force_epsilon_prediction !== undefined) setForceEpsPred(c.force_epsilon_prediction);
      if (c.dynamic_timestep_shifting !== undefined) setDynamicTimestepShift(c.dynamic_timestep_shifting);
      if (c.timestep_distribution) setTimestepDist(c.timestep_distribution);
      if (c.min_noising_strength !== undefined) setMinNoise(c.min_noising_strength);
      if (c.max_noising_strength !== undefined) setMaxNoise(c.max_noising_strength);
      if (c.noising_weight !== undefined) setNoiseWeight(c.noising_weight);
      if (c.noising_bias !== undefined) setNoiseBias(c.noising_bias);
      if (c.timestep_shift !== undefined) setTimestepShift(c.timestep_shift);
      if (c.masked_training !== undefined) setMaskedTraining(c.masked_training);
      if (c.unmasked_probability !== undefined) setUnmaskedProb(c.unmasked_probability);
      if (c.unmasked_weight !== undefined) setUnmaskedWeight(c.unmasked_weight);
      if (c.normalize_masked_area_loss !== undefined) setNormalizeMaskLoss(c.normalize_masked_area_loss);
      if (c.masked_prior_preservation_weight !== undefined) setMaskedPriorWeight(c.masked_prior_preservation_weight);
      if (c.custom_conditioning_image !== undefined) setCustomConditioningImage(c.custom_conditioning_image);
      if (c.mse_strength !== undefined) setMseStrength(c.mse_strength);
      if (c.mae_strength !== undefined) setMaeStrength(c.mae_strength);
      if (c.log_cosh_strength !== undefined) setLogCoshStrength(c.log_cosh_strength);
      if (c.huber_strength !== undefined) setHuberStrength(c.huber_strength);
      if (c.huber_delta !== undefined) setHuberDelta(c.huber_delta);
      if (c.loss_weight_fn) setLossWeightFn(c.loss_weight_fn);
      if (c.loss_weight_strength !== undefined) setGamma(c.loss_weight_strength);
      if (c.loss_scaler) setLossScaler(c.loss_scaler);
      if (c.backup_after !== undefined) setBackupAfter(c.backup_after);
      if (c.backup_after_unit) setBackupAfterUnit(c.backup_after_unit);
      if (c.rolling_backup !== undefined) setRollingBackup(c.rolling_backup);
      if (c.rolling_backup_count !== undefined) setRollingBackupCount(c.rolling_backup_count);
      if (c.backup_before_save !== undefined) setBackupBeforeSave(c.backup_before_save);
      if (c.save_every !== undefined) setSaveEvery(c.save_every);
      if (c.save_every_unit) setSaveEveryUnit(c.save_every_unit);
      if (c.save_filename_prefix) setSaveFilenamePrefix(c.save_filename_prefix);
      if (c.clip_grad_norm !== undefined) setClipGradNorm(c.clip_grad_norm);
      if (c.ema_update_step_interval !== undefined) setEmaUpdateInterval(c.ema_update_step_interval);
      if (c.force_circular_padding !== undefined) setCircularPadding(c.force_circular_padding);

      // Diffusion-4K
      if (c.diffusion_4k_enabled !== undefined) setDiffusion4kEnabled(c.diffusion_4k_enabled);
      if (c.diffusion_4k_wavelet_loss_weight !== undefined) setDiffusion4kWaveletWeight(c.diffusion_4k_wavelet_loss_weight);
      if (c.diffusion_4k_wavelet_type) setDiffusion4kWaveletType(c.diffusion_4k_wavelet_type);

      // Bucketing - these fields are in TrainConfig.py
      if (c.aspect_ratio_bucketing !== undefined) setBucketingEnabled(c.aspect_ratio_bucketing);
      if (c.bucket_quantization !== undefined) setBucketQuantization(c.bucket_quantization);
      if (c.aspect_tolerance !== undefined) setAspectTolerance(c.aspect_tolerance);
      if (c.bucket_repeat_small !== undefined) setRepeatSmallBuckets(c.bucket_repeat_small);
      if (c.bucket_log_dropped !== undefined) setLogDroppedSamples(c.bucket_log_dropped);
      if (c.bucket_preset) setBucketPreset(c.bucket_preset as keyof typeof BUCKET_PRESETS);
      if (c.bucket_balancing) setBucketBalancing(c.bucket_balancing);
      if (c.bucket_min_size !== undefined) setMinBucketSize(c.bucket_min_size);
      if (c.bucket_merge_threshold !== undefined) setBucketMergeThreshold(c.bucket_merge_threshold);
      if (c.bucket_max_per_batch !== undefined) setMaxBucketsPerBatch(c.bucket_max_per_batch);
      if (c.bucket_custom_aspects) {
        setUseCustomAspects(true);
        // Parse "16:9,4:3,1:1" format
        const parsed = c.bucket_custom_aspects.split(',').filter((s: string) => s.trim()).map((s: string) => {
          const [w, h] = s.trim().split(':').map((n: string) => parseInt(n));
          return { w: w || 1, h: h || 1 };
        });
        setCustomAspects(parsed);
      }
    }
  }, [config]);

  const tabs: { id: TabType; label: string }[] = [
    { id: 'overview', label: 'Overview' },
    { id: 'samples', label: 'Samples' },
    { id: 'config', label: 'Config File' },
    { id: 'parameters', label: 'Parameters' },
    { id: 'lora', label: 'LoRA / Adapters' },
    { id: 'diffusion4k', label: 'Diffusion 4K' },
    { id: 'buckets', label: 'Buckets' },
  ];

  const Toggle = ({ value, onChange, label }: { value: boolean; onChange: (v: boolean) => void; label?: string }) => (
    <div className="flex items-center justify-between w-full py-1">
      {label && <span className="text-sm text-white flex-1">{label}</span>}
      <button
        onClick={() => onChange(!value)}
        className={`relative w-9 h-5 rounded-full flex-shrink-0 transition-colors ${value ? 'bg-green-600' : 'bg-gray-600'}`}
      >
        <span
          className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform duration-200 ${value ? 'translate-x-4' : 'translate-x-0'}`}
        />
      </button>
    </div>
  );

  // Number input that allows proper editing - updates config on blur only
  const NumberInput = ({
    value,
    onChange,
    configKey,
    step = '1',
    min,
    max,
    isFloat = false
  }: {
    value: number | string;
    onChange: (v: number) => void;
    configKey: string;
    step?: string;
    min?: string;
    max?: string;
    isFloat?: boolean;
  }) => {
    const [localValue, setLocalValue] = useState(String(value));

    useEffect(() => {
      setLocalValue(String(value));
    }, [value]);

    const handleBlur = () => {
      const parsed = isFloat ? parseFloat(localValue) : parseInt(localValue);
      const finalValue = isNaN(parsed) ? 0 : parsed;
      onChange(finalValue);
      updateConfig({ [configKey]: finalValue });
    };

    return (
      <input
        type="text" inputMode="decimal"
        value={localValue}
        onChange={(e) => setLocalValue(e.target.value)}
        onBlur={handleBlur}
        className="input w-full text-sm"
        step={step}
        min={min}
        max={max}
      />
    );
  };

  // Settings Modal Component
  const SettingsModal = ({ title, children }: { title: string; children: React.ReactNode }) => (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70" onClick={closeSettings}>
      <div className="bg-dark-surface rounded-lg border border-dark-border w-[600px] max-h-[80vh] overflow-hidden" onClick={e => e.stopPropagation()}>
        <div className="flex items-center justify-between px-4 py-3 border-b border-dark-border">
          <h2 className="text-lg font-medium text-white">{title}</h2>
          <button onClick={closeSettings} className="p-1 hover:bg-dark-hover rounded text-muted hover:text-white">
            <X className="w-5 h-5" />
          </button>
        </div>
        <div className="p-4 overflow-y-auto max-h-[calc(80vh-60px)]">
          {children}
        </div>
      </div>
    </div>
  );

  return (
    <div className="h-full flex flex-col">
      <div className="border-b border-dark-border bg-dark-surface">
        <div className="flex items-center gap-6 px-6">
          {tabs.map((tab) => (
            <button key={tab.id} onClick={() => setActiveTab(tab.id)}
              className={`py-4 text-sm font-medium border-b-2 transition-colors ${activeTab === tab.id ? 'border-white text-white' : 'border-transparent text-muted hover:text-white'}`}>
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      <div className="flex-1 overflow-auto">
        {activeTab === 'overview' && (
          <div className="p-6">
            <div className="bg-dark-surface rounded-lg border border-dark-border p-6">
              <h2 className="text-lg font-medium text-white mb-4">Training Overview</h2>
              {currentPreset && (
                <div className="mb-4 px-3 py-2 bg-dark-bg rounded text-sm">
                  <span className="text-muted">Loaded Preset:</span>
                  <span className="text-white ml-2">{currentPreset}</span>
                </div>
              )}
              <div className="grid grid-cols-3 gap-4 text-sm">
                {[['Status', 'Idle'], ['Epoch', '-'], ['Step', '-'], ['Loss', '-'], ['LR', learningRate], ['ETA', '-']].map(([k, v]) => (
                  <div key={k} className="flex justify-between py-2 border-b border-dark-border">
                    <span className="text-muted">{k}</span><span className="text-white">{v}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'samples' && (
          <div className="h-full">
            <SamplesBrowserView />
          </div>
        )}

        {activeTab === 'config' && (
          <div className="p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-sm font-medium text-muted uppercase">Full Configuration (JSON)</h2>
              {currentPreset && (
                <span className="text-xs text-muted px-2 py-1 bg-dark-bg rounded">Loaded: {currentPreset}</span>
              )}
            </div>
            <pre className="bg-dark-card rounded-lg border border-dark-border p-4 text-sm text-green-400 font-mono overflow-auto max-h-[70vh]">
              {config ? JSON.stringify(config, null, 2) : 'No config loaded. Select a preset from "New Job" to load configuration.'}
            </pre>
          </div>
        )}

        {activeTab === 'parameters' && (
          <div className="p-4">
            <div className="grid grid-cols-3 gap-4">
              {/* Column 1: Base Training */}
              <div className="space-y-4">
                <div className="bg-dark-surface rounded-lg border border-dark-border p-3 space-y-2">
                  <div className="flex items-center justify-between mb-2">
                    <h2 className="text-xs font-medium text-muted uppercase">Optimizer & LR</h2>
                    <button onClick={() => openSettings('optimizer')} className="p-1 hover:bg-dark-hover rounded text-muted hover:text-white" title="Advanced settings">
                      <MoreHorizontal className="w-4 h-4" />
                    </button>
                  </div>
                  <div><label className="text-xs text-muted block mb-1">Optimizer</label>
                    <select value={optimizer} onChange={(e) => { setOptimizer(e.target.value); updateNestedConfig('optimizer.optimizer', e.target.value); }} className="input w-full text-sm">{OPTIMIZERS.map(o => <option key={o}>{o}</option>)}</select></div>
                  <div><label className="text-xs text-muted block mb-1">LR Scheduler</label>
                    <select value={scheduler} onChange={(e) => { setScheduler(e.target.value); updateConfig({ learning_rate_scheduler: e.target.value }); }} className="input w-full text-sm">{SCHEDULERS.map(s => <option key={s}>{s}</option>)}</select></div>
                  <div><label className="text-xs text-muted block mb-1">Learning Rate</label>
                    <input type="text" inputMode="decimal" value={learningRate} onChange={(e) => { setLearningRate(e.target.value); updateConfig({ learning_rate: parseFloat(e.target.value) || 0 }); }} className="input w-full text-sm" /></div>
                  <div><label className="text-xs text-muted block mb-1">Warmup Steps</label>
                    <NumberInput value={warmupSteps} onChange={setWarmupSteps} configKey="learning_rate_warmup_steps" /></div>
                  <div><label className="text-xs text-muted block mb-1">LR Min Factor</label>
                    <input type="text" inputMode="decimal" value={lrMinFactor} onChange={(e) => { setLrMinFactor(+e.target.value || 0); updateConfig({ learning_rate_min_factor: +e.target.value || 0 }); }} className="input w-full text-sm" step="0.01" /></div>
                  <div><label className="text-xs text-muted block mb-1">LR Cycles</label>
                    <input type="text" inputMode="decimal" value={lrCycles} onChange={(e) => { setLrCycles(+e.target.value || 1); updateConfig({ learning_rate_cycles: +e.target.value || 1 }); }} className="input w-full text-sm" step="0.1" /></div>
                  <div><label className="text-xs text-muted block mb-1">Epochs</label>
                    <NumberInput value={epochs} onChange={setEpochs} configKey="epochs" /></div>
                  <div><label className="text-xs text-muted block mb-1">Batch Size</label>
                    <NumberInput value={batchSize} onChange={setBatchSize} configKey="batch_size" min="1" /></div>
                  <div><label className="text-xs text-muted block mb-1">Accumulation Steps</label>
                    <input type="text" inputMode="decimal" value={accumSteps} onChange={(e) => { setAccumSteps(+e.target.value || 1); updateConfig({ gradient_accumulation_steps: +e.target.value || 1 }); }} className="input w-full text-sm" min="1" /></div>
                  <div><label className="text-xs text-muted block mb-1">LR Scaler</label>
                    <select value={lrScaler} onChange={(e) => { setLrScaler(e.target.value); updateConfig({ learning_rate_scaler: e.target.value }); }} className="input w-full text-sm">{LR_SCALERS.map(s => <option key={s}>{s}</option>)}</select></div>
                  <div><label className="text-xs text-muted block mb-1">Clip Grad Norm</label>
                    <input type="text" inputMode="decimal" value={clipGradNorm} onChange={(e) => { setClipGradNorm(+e.target.value || 0); updateConfig({ clip_grad_norm: +e.target.value || 0 }); }} className="input w-full text-sm" step="0.1" /></div>
                  {scheduler === 'CUSTOM' && (
                    <div><label className="text-xs text-muted block mb-1">Custom Scheduler</label>
                      <input type="text" inputMode="decimal" value={customScheduler} onChange={(e) => { setCustomScheduler(e.target.value); updateConfig({ custom_learning_rate_scheduler: e.target.value }); }} className="input w-full text-sm" placeholder="Class path" /></div>
                  )}
                  <div><label className="text-xs text-muted block mb-1">Dropout Probability</label>
                    <input type="text" inputMode="decimal" value={dropoutProb} onChange={(e) => { setDropoutProb(+e.target.value || 0); updateConfig({ dropout_probability: +e.target.value || 0 }); }} className="input w-full text-sm" step="0.01" /></div>
                  <div><label className="text-xs text-muted block mb-1">Dataloader Threads</label>
                    <input type="text" inputMode="decimal" value={dataloaderThreads} onChange={(e) => { setDataloaderThreads(+e.target.value || 2); updateConfig({ dataloader_threads: +e.target.value || 2 }); }} className="input w-full text-sm" min="1" max="16" /></div>
                </div>

                <div className="bg-dark-surface rounded-lg border border-dark-border p-3 space-y-2">
                  <div className="flex items-center justify-between mb-2">
                    <h2 className="text-xs font-medium text-muted uppercase">Text Encoder</h2>
                    <button onClick={() => openSettings('te')} className="p-1 hover:bg-dark-hover rounded text-muted hover:text-white" title="Advanced settings">
                      <MoreHorizontal className="w-4 h-4" />
                    </button>
                  </div>
                  <Toggle value={trainTE} onChange={(v) => { setTrainTE(v); updateConfig({ train_text_encoder: v }); }} label="Train Text Encoder" />
                  <div><label className="text-xs text-muted block mb-1">Caption Dropout</label>
                    <input type="text" inputMode="decimal" value={teDropout} onChange={(e) => { setTeDropout(+e.target.value || 0); updateConfig({ caption_dropout: +e.target.value || 0 }); }} className="input w-full text-sm" step="0.01" /></div>
                  <div><label className="text-xs text-muted block mb-1">Stop Training After</label>
                    <input type="text" inputMode="decimal" value={teStopAfter} onChange={(e) => { setTeStopAfter(e.target.value); updateConfig({ text_encoder_stop_after: e.target.value }); }} className="input w-full text-sm" placeholder="epochs or steps" /></div>
                  <div><label className="text-xs text-muted block mb-1">TE Learning Rate</label>
                    <input type="text" inputMode="decimal" value={teLR} onChange={(e) => { setTeLR(e.target.value); updateConfig({ text_encoder_learning_rate: parseFloat(e.target.value) || null }); }} className="input w-full text-sm" placeholder="Override base LR" /></div>
                  <div><label className="text-xs text-muted block mb-1">Clip Skip</label>
                    <input type="text" inputMode="decimal" value={clipSkip} onChange={(e) => { setClipSkip(+e.target.value || 0); updateConfig({ clip_skip: +e.target.value || 0 }); }} className="input w-full text-sm" /></div>
                </div>

                <div className="bg-dark-surface rounded-lg border border-dark-border p-3 space-y-2">
                  <h2 className="text-xs font-medium text-muted uppercase mb-2">Embeddings</h2>
                  <div><label className="text-xs text-muted block mb-1">Embeddings LR</label>
                    <input type="text" inputMode="decimal" value={embLR} onChange={(e) => { setEmbLR(e.target.value); updateConfig({ embeddings_learning_rate: parseFloat(e.target.value) || null }); }} className="input w-full text-sm" /></div>
                  <Toggle value={preserveEmbNorm} onChange={(v) => { setPreserveEmbNorm(v); updateConfig({ preserve_embeddings_norm: v }); }} label="Preserve Embedding Norm" />
                </div>
              </div>

              {/* Column 2: Model Settings */}
              <div className="space-y-4">
                <div className="bg-dark-surface rounded-lg border border-dark-border p-3 space-y-2">
                  <div className="flex items-center justify-between mb-2">
                    <h2 className="text-xs font-medium text-muted uppercase">EMA & Checkpointing</h2>
                    <button onClick={() => openSettings('ema')} className="p-1 hover:bg-dark-hover rounded text-muted hover:text-white" title="Advanced settings">
                      <MoreHorizontal className="w-4 h-4" />
                    </button>
                  </div>
                  <div><label className="text-xs text-muted block mb-1">EMA</label>
                    <select value={ema} onChange={(e) => { setEma(e.target.value); updateConfig({ ema: e.target.value }); }} className="input w-full text-sm">{EMA_MODES.map(m => <option key={m}>{m}</option>)}</select></div>
                  <div><label className="text-xs text-muted block mb-1">EMA Decay</label>
                    <input type="text" inputMode="decimal" value={emaDecay} onChange={(e) => { setEmaDecay(+e.target.value || 0.999); updateConfig({ ema_decay: +e.target.value || 0.999 }); }} className="input w-full text-sm" step="0.0001" /></div>
                  <div><label className="text-xs text-muted block mb-1">EMA Update Interval</label>
                    <input type="text" inputMode="decimal" value={emaUpdateInterval} onChange={(e) => { setEmaUpdateInterval(+e.target.value || 1); updateConfig({ ema_update_step_interval: +e.target.value || 1 }); }} className="input w-full text-sm" /></div>
                  <div><label className="text-xs text-muted block mb-1">Gradient Checkpointing</label>
                    <select value={gradCheckpoint} onChange={(e) => { setGradCheckpoint(e.target.value); updateConfig({ gradient_checkpointing: e.target.value }); }} className="input w-full text-sm">{GRADIENT_CHECKPOINTING.map(g => <option key={g}>{g}</option>)}</select></div>
                  <div><label className="text-xs text-muted block mb-1">Layer Offload Fraction</label>
                    <input type="text" inputMode="decimal" value={layerOffload} onChange={(e) => { setLayerOffload(+e.target.value || 0); updateConfig({ layer_offload_fraction: +e.target.value || 0 }); }} className="input w-full text-sm" step="0.1" min="0" max="1" /></div>
                  <div className="grid grid-cols-2 gap-2">
                    <div><label className="text-xs text-muted block mb-1" title="Block swap for training (K5, ZImage, Qwen). 0=auto">Block Swap (Train)</label>
                      <input type="number" value={musubiBlocksToSwap} onChange={(e) => { setMusubiBlocksToSwap(+e.target.value || 0); updateConfig({ musubi_blocks_to_swap: +e.target.value || 0 }); }} className="input w-full text-sm" min="0" /></div>
                    <div><label className="text-xs text-muted block mb-1" title="Block swap for inference (video models)">Block Swap (Infer)</label>
                      <input type="number" value={blocksToSwap} onChange={(e) => { setBlocksToSwap(+e.target.value || 0); updateConfig({ blocks_to_swap: +e.target.value || 0 }); }} className="input w-full text-sm" min="0" /></div>
                  </div>
                  <div><label className="text-xs text-muted block mb-1">Train Data Type</label>
                    <select value={trainDtype} onChange={(e) => { setTrainDtype(e.target.value); updateConfig({ train_dtype: e.target.value }); }} className="input w-full text-sm">{TRAIN_DTYPES.map(d => <option key={d}>{d}</option>)}</select></div>
                  <div><label className="text-xs text-muted block mb-1">Fallback Data Type</label>
                    <select value={fallbackDtype} onChange={(e) => { setFallbackDtype(e.target.value); updateConfig({ fallback_train_dtype: e.target.value }); }} className="input w-full text-sm">{FALLBACK_DTYPES.map(d => <option key={d}>{d}</option>)}</select></div>
                  <Toggle value={autocastCache} onChange={(v) => { setAutocastCache(v); updateConfig({ autocast_cache: v }); }} label="Autocast Cache" />
                  <div className="grid grid-cols-2 gap-2">
                    <div><label className="text-xs text-muted block mb-1">Resolution</label>
                      <input type="text" inputMode="decimal" value={resolution} onChange={(e) => { setResolution(e.target.value); updateConfig({ resolution: e.target.value }); }} className="input w-full text-sm" /></div>
                    <div><label className="text-xs text-muted block mb-1">Frames (Video)</label>
                      <input type="text" inputMode="decimal" value={frames} onChange={(e) => { setFrames(e.target.value); updateConfig({ frames: e.target.value }); }} className="input w-full text-sm" /></div>
                  </div>
                  <Toggle value={circularPadding} onChange={(v) => { setCircularPadding(v); updateConfig({ force_circular_padding: v }); }} label="Force Circular Padding" />
                  <Toggle value={enableAsyncOffload} onChange={(v) => { setEnableAsyncOffload(v); updateConfig({ enable_async_offloading: v }); }} label="Async Offloading" />
                  <Toggle value={enableActivationOffload} onChange={(v) => { setEnableActivationOffload(v); updateConfig({ enable_activation_offloading: v }); }} label="Activation Offloading" />
                  <Toggle value={compile} onChange={(v) => { setCompile(v); updateConfig({ compile: v }); }} label="Compile Model (torch)" />
                  <Toggle value={onlyCache} onChange={(v) => { setOnlyCache(v); updateConfig({ only_cache: v }); }} label="Only Cache (skip training)" />
                </div>

                <div className="bg-dark-surface rounded-lg border border-dark-border p-3 space-y-2">
                  <div className="flex items-center justify-between mb-2">
                    <h2 className="text-xs font-medium text-muted uppercase">Transformer / UNet</h2>
                    <button onClick={() => openSettings('unet')} className="p-1 hover:bg-dark-hover rounded text-muted hover:text-white" title="Advanced settings">
                      <MoreHorizontal className="w-4 h-4" />
                    </button>
                  </div>
                  <Toggle value={trainUnet} onChange={(v) => { setTrainUnet(v); updateConfig({ train_unet: v }); }} label="Train Transformer" />
                  <div><label className="text-xs text-muted block mb-1">Stop Training After</label>
                    <input type="text" inputMode="decimal" value={unetStopAfter} onChange={(e) => { setUnetStopAfter(e.target.value); updateConfig({ unet_stop_after: e.target.value }); }} className="input w-full text-sm" /></div>
                  <div><label className="text-xs text-muted block mb-1">Transformer LR</label>
                    <input type="text" inputMode="decimal" value={unetLR} onChange={(e) => { setUnetLR(e.target.value); updateConfig({ unet_learning_rate: parseFloat(e.target.value) || null }); }} className="input w-full text-sm" /></div>
                  <Toggle value={forceAttentionMask} onChange={(v) => { setForceAttentionMask(v); updateConfig({ force_attention_mask: v }); }} label="Force Attention Mask" />
                  <div><label className="text-xs text-muted block mb-1">Guidance Scale</label>
                    <input type="text" inputMode="decimal" value={guidanceScale} onChange={(e) => { setGuidanceScale(+e.target.value || 1); updateConfig({ guidance_scale: +e.target.value || 1 }); }} className="input w-full text-sm" step="0.5" /></div>
                  <Toggle value={rescaleNoise} onChange={(v) => { setRescaleNoise(v); updateConfig({ rescale_noise: v }); }} label="Rescale Noise + V-pred" />
                </div>

                <div className="bg-dark-surface rounded-lg border border-dark-border p-3 space-y-2">
                  <div className="flex items-center justify-between mb-2">
                    <h2 className="text-xs font-medium text-muted uppercase">Noise</h2>
                    <button onClick={() => openSettings('noise')} className="p-1 hover:bg-dark-hover rounded text-muted hover:text-white" title="Advanced settings">
                      <MoreHorizontal className="w-4 h-4" />
                    </button>
                  </div>
                  <div><label className="text-xs text-muted block mb-1">Offset Noise Weight</label>
                    <input type="text" inputMode="decimal" value={offsetNoise} onChange={(e) => { setOffsetNoise(+e.target.value || 0); updateConfig({ offset_noise_weight: +e.target.value || 0 }); }} className="input w-full text-sm" step="0.01" /></div>
                  <div><label className="text-xs text-muted block mb-1">Perturbation Noise</label>
                    <input type="text" inputMode="decimal" value={perturbNoise} onChange={(e) => { setPerturbNoise(+e.target.value || 0); updateConfig({ perturbation_noise_weight: +e.target.value || 0 }); }} className="input w-full text-sm" step="0.01" /></div>
                  <div><label className="text-xs text-muted block mb-1">Timestep Distribution</label>
                    <select value={timestepDist} onChange={(e) => { setTimestepDist(e.target.value); updateConfig({ timestep_distribution: e.target.value }); }} className="input w-full text-sm">{TIMESTEP_DIST.map(t => <option key={t}>{t}</option>)}</select></div>
                  <div className="grid grid-cols-2 gap-2">
                    <div><label className="text-xs text-muted block mb-1">Min Noise</label>
                      <input type="text" inputMode="decimal" value={minNoise} onChange={(e) => { setMinNoise(+e.target.value || 0); updateConfig({ min_noising_strength: +e.target.value || 0 }); }} className="input w-full text-sm" step="0.01" /></div>
                    <div><label className="text-xs text-muted block mb-1">Max Noise</label>
                      <input type="text" inputMode="decimal" value={maxNoise} onChange={(e) => { setMaxNoise(+e.target.value || 1); updateConfig({ max_noising_strength: +e.target.value || 1 }); }} className="input w-full text-sm" step="0.01" /></div>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <div><label className="text-xs text-muted block mb-1">Noise Weight</label>
                      <input type="text" inputMode="decimal" value={noiseWeight} onChange={(e) => { setNoiseWeight(+e.target.value || 0); updateConfig({ noising_weight: +e.target.value || 0 }); }} className="input w-full text-sm" step="0.1" /></div>
                    <div><label className="text-xs text-muted block mb-1">Noise Bias</label>
                      <input type="text" inputMode="decimal" value={noiseBias} onChange={(e) => { setNoiseBias(+e.target.value || 0); updateConfig({ noising_bias: +e.target.value || 0 }); }} className="input w-full text-sm" step="0.1" /></div>
                  </div>
                  <div><label className="text-xs text-muted block mb-1">Timestep Shift</label>
                    <input type="text" inputMode="decimal" value={timestepShift} onChange={(e) => { const v = parseFloat(e.target.value) || 0; setTimestepShift(v); updateConfig({ timestep_shift: v }); }} className="input w-full text-sm" /></div>
                  <Toggle value={generalizedOffsetNoise} onChange={(v) => { setGeneralizedOffsetNoise(v); updateConfig({ generalized_offset_noise: v }); }} label="Generalized Offset Noise" />
                  <Toggle value={forceVPred} onChange={(v) => { setForceVPred(v); updateConfig({ force_v_prediction: v }); }} label="Force V-Prediction" />
                  <Toggle value={forceEpsPred} onChange={(v) => { setForceEpsPred(v); updateConfig({ force_epsilon_prediction: v }); }} label="Force Epsilon Prediction" />
                  <Toggle value={dynamicTimestepShift} onChange={(v) => { setDynamicTimestepShift(v); updateConfig({ dynamic_timestep_shifting: v }); }} label="Dynamic Timestep Shifting" />
                </div>

                <div className="bg-dark-surface rounded-lg border border-dark-border p-3 space-y-2">
                  <h2 className="text-xs font-medium text-muted uppercase mb-2">Layer Filter</h2>
                  <div><label className="text-xs text-muted block mb-1">Preset</label>
                    <select value={layerFilterPreset} onChange={(e) => { setLayerFilterPreset(e.target.value); updateConfig({ layer_filter_preset: e.target.value }); }} className="input w-full text-sm">{LAYER_PRESETS.map(p => <option key={p}>{p}</option>)}</select></div>
                  <div><label className="text-xs text-muted block mb-1">Custom Filter</label>
                    <input type="text" inputMode="decimal" value={layerFilter} onChange={(e) => { setLayerFilter(e.target.value); updateConfig({ layer_filter: e.target.value }); }} className="input w-full text-sm" placeholder="Comma-separated layers" /></div>
                  <Toggle value={layerFilterRegex} onChange={(v) => { setLayerFilterRegex(v); updateConfig({ layer_filter_regex: v }); }} label="Use Regex" />
                </div>
              </div>

              {/* Column 3: Loss & Masked */}
              <div className="space-y-4">
                <div className="bg-dark-surface rounded-lg border border-dark-border p-3 space-y-2">
                  <div className="flex items-center justify-between mb-2">
                    <h2 className="text-xs font-medium text-muted uppercase">Masked Training</h2>
                    <button onClick={() => openSettings('masked')} className="p-1 hover:bg-dark-hover rounded text-muted hover:text-white" title="Advanced settings">
                      <MoreHorizontal className="w-4 h-4" />
                    </button>
                  </div>
                  <Toggle value={maskedTraining} onChange={(v) => { setMaskedTraining(v); updateConfig({ masked_training: v }); }} label="Masked Training" />
                  <div className="grid grid-cols-2 gap-2">
                    <div><label className="text-xs text-muted block mb-1">Unmasked Prob</label>
                      <input type="text" inputMode="decimal" value={unmaskedProb} onChange={(e) => { setUnmaskedProb(+e.target.value || 0); updateConfig({ unmasked_probability: +e.target.value || 0 }); }} className="input w-full text-sm" step="0.01" /></div>
                    <div><label className="text-xs text-muted block mb-1">Unmasked Weight</label>
                      <input type="text" inputMode="decimal" value={unmaskedWeight} onChange={(e) => { setUnmaskedWeight(+e.target.value || 0); updateConfig({ unmasked_weight: +e.target.value || 0 }); }} className="input w-full text-sm" step="0.01" /></div>
                  </div>
                  <div><label className="text-xs text-muted block mb-1">Prior Preservation Weight</label>
                    <input type="text" inputMode="decimal" value={maskedPriorWeight} onChange={(e) => { setMaskedPriorWeight(+e.target.value || 0); updateConfig({ masked_prior_preservation_weight: +e.target.value || 0 }); }} className="input w-full text-sm" step="0.01" /></div>
                  <Toggle value={normalizeMaskLoss} onChange={(v) => { setNormalizeMaskLoss(v); updateConfig({ normalize_masked_area_loss: v }); }} label="Normalize Masked Loss" />
                  <Toggle value={customConditioningImage} onChange={(v) => { setCustomConditioningImage(v); updateConfig({ custom_conditioning_image: v }); }} label="Custom Conditioning Image" />
                </div>

                <div className="bg-dark-surface rounded-lg border border-dark-border p-3 space-y-2">
                  <div className="flex items-center justify-between mb-2">
                    <h2 className="text-xs font-medium text-muted uppercase">Loss</h2>
                    <button onClick={() => openSettings('loss')} className="p-1 hover:bg-dark-hover rounded text-muted hover:text-white" title="Advanced settings">
                      <MoreHorizontal className="w-4 h-4" />
                    </button>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <div><label className="text-xs text-muted block mb-1">MSE</label>
                      <input type="text" inputMode="decimal" value={mseStrength} onChange={(e) => { setMseStrength(+e.target.value || 0); updateConfig({ mse_strength: +e.target.value || 0 }); }} className="input w-full text-sm" step="0.1" /></div>
                    <div><label className="text-xs text-muted block mb-1">MAE</label>
                      <input type="text" inputMode="decimal" value={maeStrength} onChange={(e) => { setMaeStrength(+e.target.value || 0); updateConfig({ mae_strength: +e.target.value || 0 }); }} className="input w-full text-sm" step="0.1" /></div>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <div><label className="text-xs text-muted block mb-1">Log-Cosh</label>
                      <input type="text" inputMode="decimal" value={logCoshStrength} onChange={(e) => { setLogCoshStrength(+e.target.value || 0); updateConfig({ log_cosh_strength: +e.target.value || 0 }); }} className="input w-full text-sm" step="0.1" /></div>
                    <div><label className="text-xs text-muted block mb-1">Huber</label>
                      <input type="text" inputMode="decimal" value={huberStrength} onChange={(e) => { setHuberStrength(+e.target.value || 0); updateConfig({ huber_strength: +e.target.value || 0 }); }} className="input w-full text-sm" step="0.1" /></div>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <div><label className="text-xs text-muted block mb-1">Huber Delta</label>
                      <input type="text" inputMode="decimal" value={huberDelta} onChange={(e) => { setHuberDelta(+e.target.value || 1); updateConfig({ huber_delta: +e.target.value || 1 }); }} className="input w-full text-sm" step="0.1" /></div>
                    <div><label className="text-xs text-muted block mb-1">VB Loss</label>
                      <input type="text" inputMode="decimal" value={vbLossStrength} onChange={(e) => { setVbLossStrength(+e.target.value || 1); updateConfig({ vb_loss_strength: +e.target.value || 1 }); }} className="input w-full text-sm" step="0.1" /></div>
                  </div>
                  <div><label className="text-xs text-muted block mb-1">Loss Weight Function</label>
                    <select value={lossWeightFn} onChange={(e) => { setLossWeightFn(e.target.value); updateConfig({ loss_weight_fn: e.target.value }); }} className="input w-full text-sm">{LOSS_WEIGHTS.map(l => <option key={l}>{l}</option>)}</select></div>
                  <div><label className="text-xs text-muted block mb-1">Gamma (SNR/P2)</label>
                    <input type="text" inputMode="decimal" value={gamma} onChange={(e) => { setGamma(+e.target.value || 5); updateConfig({ loss_weight_strength: +e.target.value || 5 }); }} className="input w-full text-sm" step="0.5" /></div>
                  <div><label className="text-xs text-muted block mb-1">Loss Scaler</label>
                    <select value={lossScaler} onChange={(e) => { setLossScaler(e.target.value); updateConfig({ loss_scaler: e.target.value }); }} className="input w-full text-sm">{LOSS_SCALERS.map(l => <option key={l}>{l}</option>)}</select></div>
                </div>

                <div className="bg-dark-surface rounded-lg border border-dark-border p-3 space-y-2">
                  <div className="flex items-center justify-between mb-2">
                    <h2 className="text-xs font-medium text-muted uppercase">Device & Multi-GPU</h2>
                    <button onClick={() => openSettings('device')} className="p-1 hover:bg-dark-hover rounded text-muted hover:text-white" title="Advanced settings">
                      <MoreHorizontal className="w-4 h-4" />
                    </button>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <div><label className="text-xs text-muted block mb-1">Train Device</label>
                      <input type="text" inputMode="decimal" value={trainDevice} onChange={(e) => { setTrainDevice(e.target.value); updateConfig({ train_device: e.target.value }); }} className="input w-full text-sm" /></div>
                    <div><label className="text-xs text-muted block mb-1">Temp Device</label>
                      <input type="text" inputMode="decimal" value={tempDevice} onChange={(e) => { setTempDevice(e.target.value); updateConfig({ temp_device: e.target.value }); }} className="input w-full text-sm" /></div>
                  </div>
                  <Toggle value={multiGpu} onChange={(v) => { setMultiGpu(v); updateConfig({ multi_gpu: v }); }} label="Multi-GPU Training" />
                  {multiGpu && (
                    <div><label className="text-xs text-muted block mb-1">Device Indexes</label>
                      <input type="text" inputMode="decimal" value={deviceIndexes} onChange={(e) => { setDeviceIndexes(e.target.value); updateConfig({ device_indexes: e.target.value }); }} className="input w-full text-sm" placeholder="0,1,2,3" /></div>
                  )}
                </div>

                <div className="bg-dark-surface rounded-lg border border-dark-border p-3 space-y-2">
                  <div className="flex items-center justify-between mb-2">
                    <h2 className="text-xs font-medium text-muted uppercase">Backup & Save</h2>
                    <button onClick={() => openSettings('backup')} className="p-1 hover:bg-dark-hover rounded text-muted hover:text-white" title="Advanced settings">
                      <MoreHorizontal className="w-4 h-4" />
                    </button>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <div><label className="text-xs text-muted block mb-1">Backup After</label>
                      <input type="text" inputMode="decimal" value={backupAfter} onChange={(e) => { setBackupAfter(+e.target.value || 30); updateConfig({ backup_after: +e.target.value || 30 }); }} className="input w-full text-sm" /></div>
                    <div><label className="text-xs text-muted block mb-1">Unit</label>
                      <select value={backupAfterUnit} onChange={(e) => { setBackupAfterUnit(e.target.value); updateConfig({ backup_after_unit: e.target.value }); }} className="input w-full text-sm">{TIME_UNITS.map(u => <option key={u}>{u}</option>)}</select></div>
                  </div>
                  <Toggle value={rollingBackup} onChange={(v) => { setRollingBackup(v); updateConfig({ rolling_backup: v }); }} label="Rolling Backup" />
                  {rollingBackup && (
                    <div><label className="text-xs text-muted block mb-1">Backup Count</label>
                      <input type="text" inputMode="decimal" value={rollingBackupCount} onChange={(e) => { setRollingBackupCount(+e.target.value || 3); updateConfig({ rolling_backup_count: +e.target.value || 3 }); }} className="input w-full text-sm" min="1" /></div>
                  )}
                  <Toggle value={backupBeforeSave} onChange={(v) => { setBackupBeforeSave(v); updateConfig({ backup_before_save: v }); }} label="Backup Before Save" />
                  <div className="grid grid-cols-2 gap-2">
                    <div><label className="text-xs text-muted block mb-1">Save Every</label>
                      <input type="text" inputMode="decimal" value={saveEvery} onChange={(e) => { setSaveEvery(+e.target.value || 0); updateConfig({ save_every: +e.target.value || 0 }); }} className="input w-full text-sm" /></div>
                    <div><label className="text-xs text-muted block mb-1">Unit</label>
                      <select value={saveEveryUnit} onChange={(e) => { setSaveEveryUnit(e.target.value); updateConfig({ save_every_unit: e.target.value }); }} className="input w-full text-sm">{TIME_UNITS.map(u => <option key={u}>{u}</option>)}</select></div>
                  </div>
                  <div><label className="text-xs text-muted block mb-1">Filename Prefix</label>
                    <input type="text" inputMode="decimal" value={saveFilenamePrefix} onChange={(e) => { setSaveFilenamePrefix(e.target.value); updateConfig({ save_filename_prefix: e.target.value }); }} className="input w-full text-sm" placeholder="model_" /></div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'lora' && (
          <div className="h-full">
            <LoRAView />
          </div>
        )}

        {activeTab === 'diffusion4k' && (
          <div className="p-6">
            <div className="max-w-2xl space-y-6">
              <div className="bg-dark-surface rounded-lg border border-dark-border p-6">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h2 className="text-lg font-medium text-white">Diffusion-4K Wavelet Loss</h2>
                    <p className="text-sm text-muted">Enhance high-frequency detail preservation using wavelet-based loss (from arXiv:2503.18352)</p>
                  </div>
                  <button
                    onClick={() => {
                      setDiffusion4kEnabled(!diffusion4kEnabled);
                      updateConfig({ diffusion_4k_enabled: !diffusion4kEnabled });
                    }}
                    className={`relative w-12 h-6 rounded-full flex-shrink-0 transition-colors ${diffusion4kEnabled ? 'bg-green-600' : 'bg-gray-600'}`}
                  >
                    <span className={`absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white transition-transform duration-200 ${diffusion4kEnabled ? 'translate-x-6' : 'translate-x-0'}`} />
                  </button>
                </div>

                {diffusion4kEnabled && (
                  <div className="space-y-4 pt-4 border-t border-dark-border">
                    <div>
                      <label className="text-xs text-muted block mb-1">Wavelet Loss Weight</label>
                      <input
                        type="text" inputMode="decimal"
                        value={diffusion4kWaveletWeight}
                        onChange={(e) => {
                          const val = parseFloat(e.target.value) || 1.0;
                          setDiffusion4kWaveletWeight(val);
                          updateConfig({ diffusion_4k_wavelet_loss_weight: val });
                        }}
                        className="input w-32 text-sm"
                        min="0"
                        max="10"
                        step="0.1"
                      />
                      <p className="text-xs text-muted mt-1">Weight for wavelet-decomposed loss (1.0 = equal to MSE loss)</p>
                    </div>

                    <div>
                      <label className="text-xs text-muted block mb-1">Wavelet Type</label>
                      <select
                        value={diffusion4kWaveletType}
                        onChange={(e) => {
                          setDiffusion4kWaveletType(e.target.value);
                          updateConfig({ diffusion_4k_wavelet_type: e.target.value });
                        }}
                        className="input w-32 text-sm"
                      >
                        <option value="haar">Haar</option>
                        <option value="db1">Daubechies-1</option>
                      </select>
                      <p className="text-xs text-muted mt-1">Haar is recommended as per Diffusion-4K paper</p>
                    </div>

                    <div className="bg-dark-card rounded-lg p-4 mt-4">
                      <h3 className="text-sm font-medium text-white mb-2">How it works</h3>
                      <p className="text-xs text-muted">
                        Applies Discrete Wavelet Transform (DWT) to decompose latents into low-frequency (LL) and high-frequency (LH, HL, HH) components.
                        Loss is computed on all subbands, emphasizing fine details and textures during training.
                      </p>
                    </div>
                  </div>
                )}
              </div>

              {/* Resolution preset for 4K sampling */}
              <div className="bg-dark-surface rounded-lg border border-dark-border p-6">
                <h2 className="text-lg font-medium text-white mb-2">4K Resolution Presets</h2>
                <p className="text-sm text-muted mb-4">Quick resolution presets for high-resolution training and sampling</p>
                <div className="flex gap-2">
                  {[{ label: '1024', w: 1024, h: 1024 }, { label: '2048', w: 2048, h: 2048 }, { label: '4096 (4K)', w: 4096, h: 4096 }].map((preset) => (
                    <button
                      key={preset.label}
                      onClick={() => updateConfig({ resolution: String(preset.w) })}
                      className="px-4 py-2 bg-dark-border hover:bg-dark-hover text-white rounded text-sm"
                    >
                      {preset.label}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'buckets' && (
          <div className="p-6 space-y-6">
            {/* Enable/Disable */}
            <div className="bg-dark-surface rounded-lg border border-dark-border p-4">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h2 className="text-lg font-medium text-white">Aspect Ratio Bucketing</h2>
                  <p className="text-sm text-muted">Group images by aspect ratio to minimize cropping</p>
                </div>
                <Toggle value={bucketingEnabled} onChange={(v) => {
                  setBucketingEnabled(v);
                  updateConfig({ aspect_ratio_bucketing: v });
                }} />
              </div>

              {bucketingEnabled && (
                <div className="grid grid-cols-2 gap-6">
                  {/* Left Column - Settings */}
                  <div className="space-y-4">
                    {/* Aspect Ratio Presets */}
                    <div className="bg-dark-card rounded-lg border border-dark-border p-3">
                      <h3 className="text-sm font-medium text-white mb-3">Aspect Ratio Presets</h3>
                      <div className="space-y-3">
                        <div>
                          <label className="text-xs text-muted block mb-1">Preset</label>
                          <select
                            value={bucketPreset}
                            onChange={(e) => {
                              const v = e.target.value as keyof typeof BUCKET_PRESETS;
                              setBucketPreset(v);
                              updateConfig({ bucket_preset: v });
                            }}
                            disabled={useCustomAspects}
                            className="input w-full text-sm"
                          >
                            <option value="default">Default (Original OneTrainer)</option>
                            <option value="photo">Photo (3:2, 4:3, 16:9, etc.)</option>
                            <option value="video">Video (16:9, 21:9, etc.)</option>
                            <option value="widescreen">Widescreen (Ultrawide)</option>
                            <option value="square">Square Only (1:1)</option>
                          </select>
                        </div>

                        <div className="text-xs text-muted">
                          <span className="font-medium">Included ratios: </span>
                          {BUCKET_PRESETS[bucketPreset].map((r, i) => (
                            <span key={i} className="inline-block bg-dark-bg px-1.5 py-0.5 rounded mr-1 mb-1">
                              {r.w}:{r.h}
                            </span>
                          ))}
                        </div>

                        <Toggle value={useCustomAspects} onChange={(v) => {
                          setUseCustomAspects(v);
                          if (!v) {
                            updateConfig({ bucket_custom_aspects: '' });
                          }
                        }} label="Use Custom Aspects" />

                        {useCustomAspects && (
                          <div className="space-y-2">
                            <label className="text-xs text-muted block">Custom Ratios (one per line, format: W:H)</label>
                            <textarea
                              className="input w-full h-24 text-sm font-mono"
                              placeholder="16:9&#10;4:3&#10;1:1&#10;3:4&#10;9:16"
                              value={customAspects.map(a => `${a.w}:${a.h}`).join('\n')}
                              onChange={(e) => {
                                const lines = e.target.value.split('\n').filter(l => l.trim());
                                const parsed = lines.map(l => {
                                  const [w, h] = l.split(':').map(n => parseInt(n.trim()));
                                  return { w: w || 1, h: h || 1 };
                                }).filter(a => !isNaN(a.w) && !isNaN(a.h));
                                setCustomAspects(parsed);
                                // Save as comma-separated string
                                updateConfig({ bucket_custom_aspects: parsed.map(a => `${a.w}:${a.h}`).join(',') });
                              }}
                            />
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Bucket Parameters */}
                    <div className="bg-dark-card rounded-lg border border-dark-border p-3">
                      <h3 className="text-sm font-medium text-white mb-3">Bucket Parameters</h3>
                      <div className="grid grid-cols-2 gap-3">
                        <div>
                          <label className="text-xs text-muted block mb-1">Quantization</label>
                          <select
                            value={bucketQuantization}
                            onChange={(e) => {
                              const v = parseInt(e.target.value);
                              setBucketQuantization(v);
                              updateConfig({ bucket_quantization: v });
                            }}
                            className="input w-full text-sm"
                          >
                            {QUANTIZATION_VALUES.map(v => (
                              <option key={v} value={v}>{v}px</option>
                            ))}
                          </select>
                          <p className="text-xs text-muted mt-1">Resolution granularity</p>
                        </div>

                        <div>
                          <label className="text-xs text-muted block mb-1">Aspect Tolerance</label>
                          <input
                            type="text" inputMode="decimal"
                            value={aspectTolerance}
                            onChange={(e) => {
                              const v = parseFloat(e.target.value);
                              setAspectTolerance(v);
                              updateConfig({ aspect_tolerance: v });
                            }}
                            step="0.01"
                            min="0"
                            max="0.5"
                            className="input w-full text-sm"
                          />
                          <p className="text-xs text-muted mt-1">Max crop % (0.15 = 15%)</p>
                        </div>

                        <div>
                          <label className="text-xs text-muted block mb-1">Min Bucket Size</label>
                          <input
                            type="text" inputMode="decimal"
                            value={minBucketSize}
                            onChange={(e) => {
                              const v = parseInt(e.target.value);
                              setMinBucketSize(v);
                              updateConfig({ bucket_min_size: v });
                            }}
                            min="1"
                            className="input w-full text-sm"
                          />
                          <p className="text-xs text-muted mt-1">Min images per bucket</p>
                        </div>

                        <div>
                          <label className="text-xs text-muted block mb-1">Merge Threshold</label>
                          <input
                            type="text" inputMode="decimal"
                            value={bucketMergeThreshold}
                            onChange={(e) => {
                              const v = parseInt(e.target.value);
                              setBucketMergeThreshold(v);
                              updateConfig({ bucket_merge_threshold: v });
                            }}
                            min="0"
                            className="input w-full text-sm"
                          />
                          <p className="text-xs text-muted mt-1">Merge if &lt; N images (0=off)</p>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Right Column - Balancing & Advanced */}
                  <div className="space-y-4">
                    {/* Balancing */}
                    <div className="bg-dark-card rounded-lg border border-dark-border p-3">
                      <h3 className="text-sm font-medium text-white mb-3">Bucket Balancing</h3>
                      <div className="space-y-3">
                        <div>
                          <label className="text-xs text-muted block mb-1">Balancing Mode</label>
                          <select
                            value={bucketBalancing}
                            onChange={(e) => {
                              const v = e.target.value;
                              setBucketBalancing(v);
                              updateConfig({ bucket_balancing: v });
                            }}
                            className="input w-full text-sm"
                          >
                            <option value="OFF">Off - Use buckets as-is</option>
                            <option value="OVERSAMPLE">Oversample - Repeat minority buckets</option>
                            <option value="WEIGHTED">Weighted - Inverse frequency sampling</option>
                          </select>
                        </div>

                        <Toggle
                          value={repeatSmallBuckets}
                          onChange={(v) => {
                            setRepeatSmallBuckets(v);
                            updateConfig({ bucket_repeat_small: v });
                          }}
                          label="Repeat Samples in Small Buckets"
                        />
                        <p className="text-xs text-muted -mt-2 ml-4">
                          Duplicate samples to fill batches instead of dropping
                        </p>

                        <div>
                          <label className="text-xs text-muted block mb-1">Max Buckets Per Batch</label>
                          <input
                            type="text" inputMode="decimal"
                            value={maxBucketsPerBatch}
                            onChange={(e) => {
                              const v = parseInt(e.target.value);
                              setMaxBucketsPerBatch(v);
                              updateConfig({ bucket_max_per_batch: v });
                            }}
                            min="0"
                            className="input w-full text-sm"
                          />
                          <p className="text-xs text-muted mt-1">Limit bucket diversity per batch (0=unlimited)</p>
                        </div>
                      </div>
                    </div>

                    {/* Logging & Debug */}
                    <div className="bg-dark-card rounded-lg border border-dark-border p-3">
                      <h3 className="text-sm font-medium text-white mb-3">Logging & Debug</h3>
                      <div className="space-y-2">
                        <Toggle
                          value={logDroppedSamples}
                          onChange={(v) => {
                            setLogDroppedSamples(v);
                            updateConfig({ bucket_log_dropped: v });
                          }}
                          label="Log Dropped Samples"
                        />
                        <p className="text-xs text-muted ml-4">
                          Warn when images are dropped due to bucket constraints
                        </p>
                      </div>
                    </div>

                    {/* Config Summary */}
                    <div className="bg-dark-card rounded-lg border border-dark-border p-3">
                      <h3 className="text-sm font-medium text-white mb-2">Config Summary</h3>
                      <pre className="text-xs text-green-400 font-mono bg-dark-bg rounded p-2 overflow-auto">
                        {`aspect_ratio_bucketing: ${bucketingEnabled}
bucket_preset: ${useCustomAspects ? 'custom' : bucketPreset}
bucket_quantization: ${bucketQuantization}
aspect_tolerance: ${aspectTolerance}
bucket_min_size: ${minBucketSize}
bucket_merge_threshold: ${bucketMergeThreshold}
bucket_balancing: ${bucketBalancing}
bucket_max_per_batch: ${maxBucketsPerBatch}
bucket_repeat_small: ${repeatSmallBuckets}
bucket_log_dropped: ${logDroppedSamples}${useCustomAspects ? `\nbucket_custom_aspects: ${customAspects.map(a => `${a.w}:${a.h}`).join(',')}` : ''}`}
                      </pre>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Bucket Preview / Stats */}
            {bucketingEnabled && (
              <div className="bg-dark-surface rounded-lg border border-dark-border p-4">
                <h3 className="text-sm font-medium text-white mb-3">Bucket Preview</h3>
                <p className="text-xs text-muted mb-4">
                  Visual representation of aspect ratio buckets. Load a dataset concept to see actual distribution.
                </p>
                <div className="grid grid-cols-9 gap-2">
                  {(useCustomAspects ? customAspects : BUCKET_PRESETS[bucketPreset]).map((ratio, i) => {
                    const aspect = ratio.w / ratio.h;
                    const isLandscape = aspect >= 1;
                    const width = isLandscape ? 60 : 60 * aspect;
                    const height = isLandscape ? 60 / aspect : 60;
                    return (
                      <div key={i} className="flex flex-col items-center">
                        <div
                          className="bg-primary/30 border border-primary/50 rounded flex items-center justify-center text-xs text-primary"
                          style={{ width: `${width}px`, height: `${height}px` }}
                        >
                          {ratio.w}:{ratio.h}
                        </div>
                        <span className="text-xs text-muted mt-1">{aspect.toFixed(2)}</span>
                      </div>
                    );
                  })}
                </div>
                {/* Landscape + Portrait indicator */}
                <div className="mt-4 flex gap-4 text-xs text-muted">
                  <span>
                    <span className="inline-block w-3 h-3 bg-blue-500/30 border border-blue-500/50 rounded mr-1" />
                    Landscape ({(useCustomAspects ? customAspects : BUCKET_PRESETS[bucketPreset]).filter(r => r.w >= r.h).length})
                  </span>
                  <span>
                    <span className="inline-block w-3 h-3 bg-purple-500/30 border border-purple-500/50 rounded mr-1" />
                    Portrait ({(useCustomAspects ? customAspects : BUCKET_PRESETS[bucketPreset]).filter(r => r.w < r.h).length})
                  </span>
                  <span>
                    Total: {(useCustomAspects ? customAspects : BUCKET_PRESETS[bucketPreset]).length * 2 -
                      (useCustomAspects ? customAspects : BUCKET_PRESETS[bucketPreset]).filter(r => r.w === r.h).length} buckets
                    (including flipped)
                  </span>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Settings Modals */}
      {settingsModal === 'optimizer' && (
        <SettingsModal title="Optimizer Settings">
          <div className="space-y-4">
            <div className="grid grid-cols-3 gap-3">
              <div><label className="text-xs text-muted block mb-1">Weight Decay</label>
                <input type="text" inputMode="decimal" value={optWeightDecay} onChange={(e) => setOptWeightDecay(+e.target.value)} className="input w-full text-sm" step="0.001" /></div>
              <div><label className="text-xs text-muted block mb-1">Eps</label>
                <input type="text" inputMode="decimal" value={optEps} onChange={(e) => setOptEps(+e.target.value || 1e-8)} className="input w-full text-sm" /></div>
              <div><label className="text-xs text-muted block mb-1">Beta1</label>
                <input type="text" inputMode="decimal" value={optBeta1} onChange={(e) => setOptBeta1(+e.target.value)} className="input w-full text-sm" step="0.01" /></div>
              <div><label className="text-xs text-muted block mb-1">Beta2</label>
                <input type="text" inputMode="decimal" value={optBeta2} onChange={(e) => setOptBeta2(+e.target.value)} className="input w-full text-sm" step="0.001" /></div>
              <div><label className="text-xs text-muted block mb-1">Beta3</label>
                <input type="text" inputMode="decimal" value={optBeta3} onChange={(e) => setOptBeta3(+e.target.value)} className="input w-full text-sm" step="0.01" /></div>
              <div><label className="text-xs text-muted block mb-1">Momentum</label>
                <input type="text" inputMode="decimal" value={optMomentum} onChange={(e) => setOptMomentum(+e.target.value)} className="input w-full text-sm" step="0.01" /></div>
              <div><label className="text-xs text-muted block mb-1">Dampening</label>
                <input type="text" inputMode="decimal" value={optDampening} onChange={(e) => setOptDampening(+e.target.value)} className="input w-full text-sm" step="0.01" /></div>
              <div><label className="text-xs text-muted block mb-1">D0 (Prodigy)</label>
                <input type="text" inputMode="decimal" value={optD0} onChange={(e) => setOptD0(+e.target.value)} className="input w-full text-sm" /></div>
              <div><label className="text-xs text-muted block mb-1">D Coef</label>
                <input type="text" inputMode="decimal" value={optDCoef} onChange={(e) => setOptDCoef(+e.target.value)} className="input w-full text-sm" step="0.1" /></div>
              <div><label className="text-xs text-muted block mb-1">Growth Rate</label>
                <input type="text" inputMode="decimal" value={optGrowthRate} onChange={(e) => setOptGrowthRate(+e.target.value)} className="input w-full text-sm" /></div>
              <div><label className="text-xs text-muted block mb-1">Clip Threshold</label>
                <input type="text" inputMode="decimal" value={optClipThreshold} onChange={(e) => setOptClipThreshold(+e.target.value)} className="input w-full text-sm" step="0.1" /></div>
            </div>
            <div className="border-t border-dark-border pt-3">
              <h3 className="text-xs font-medium text-muted uppercase mb-2">Options</h3>
              <div className="grid grid-cols-3 gap-2">
                <Toggle value={optFused} onChange={setOptFused} label="Fused" />
                <Toggle value={optStochasticRounding} onChange={setOptStochasticRounding} label="Stochastic Rounding" />
                <Toggle value={optAmsgrad} onChange={setOptAmsgrad} label="AMSGrad" />
                <Toggle value={optNesterov} onChange={setOptNesterov} label="Nesterov" />
                <Toggle value={optForeach} onChange={setOptForeach} label="Foreach" />
                <Toggle value={optDecouple} onChange={setOptDecouple} label="Decouple" />
                <Toggle value={optUseBiasCorrection} onChange={setOptUseBiasCorrection} label="Bias Correction" />
                <Toggle value={optAdamWMode} onChange={setOptAdamWMode} label="AdamW Mode" />
                <Toggle value={optCautious} onChange={setOptCautious} label="Cautious" />
                <Toggle value={optRelativeStep} onChange={setOptRelativeStep} label="Relative Step" />
                <Toggle value={optSafeguardWarmup} onChange={setOptSafeguardWarmup} label="Safeguard Warmup" />
              </div>
            </div>
          </div>
        </SettingsModal>
      )}

      {settingsModal === 'ema' && (
        <SettingsModal title="EMA & Checkpointing Settings">
          <div className="space-y-3">
            <p className="text-sm text-muted">Advanced EMA settings are shown in the main panel. Click ... for more options in future updates.</p>
          </div>
        </SettingsModal>
      )}

      {settingsModal === 'te' && (
        <SettingsModal title="Text Encoder Settings">
          <div className="space-y-3">
            <p className="text-sm text-muted">Text encoder settings are shown in the main panel.</p>
          </div>
        </SettingsModal>
      )}

      {settingsModal === 'unet' && (
        <SettingsModal title="Transformer / UNet Settings">
          <div className="space-y-3">
            <p className="text-sm text-muted">Transformer settings are shown in the main panel.</p>
          </div>
        </SettingsModal>
      )}

      {settingsModal === 'noise' && (
        <SettingsModal title="Noise Settings">
          <div className="space-y-3">
            <p className="text-sm text-muted">Noise settings are shown in the main panel.</p>
          </div>
        </SettingsModal>
      )}

      {settingsModal === 'masked' && (
        <SettingsModal title="Masked Training Settings">
          <div className="space-y-3">
            <p className="text-sm text-muted">Masked training settings are shown in the main panel.</p>
          </div>
        </SettingsModal>
      )}

      {settingsModal === 'loss' && (
        <SettingsModal title="Loss Settings">
          <div className="space-y-3">
            <p className="text-sm text-muted">Loss settings are shown in the main panel.</p>
          </div>
        </SettingsModal>
      )}

      {settingsModal === 'device' && (
        <SettingsModal title="Device & Multi-GPU Settings">
          <div className="space-y-3">
            <p className="text-sm text-muted">Device settings control where training computations run.</p>
            <div className="grid grid-cols-2 gap-3">
              <div><label className="text-xs text-muted block mb-1">Train Device</label>
                <input type="text" inputMode="decimal" value={trainDevice} onChange={(e) => setTrainDevice(e.target.value)} className="input w-full text-sm" /></div>
              <div><label className="text-xs text-muted block mb-1">Temp Device</label>
                <input type="text" inputMode="decimal" value={tempDevice} onChange={(e) => setTempDevice(e.target.value)} className="input w-full text-sm" /></div>
            </div>
            <Toggle value={multiGpu} onChange={setMultiGpu} label="Multi-GPU Training" />
            {multiGpu && (
              <div><label className="text-xs text-muted block mb-1">Device Indexes (comma-separated)</label>
                <input type="text" inputMode="decimal" value={deviceIndexes} onChange={(e) => setDeviceIndexes(e.target.value)} className="input w-full text-sm" placeholder="0,1,2,3" /></div>
            )}
          </div>
        </SettingsModal>
      )}

      {settingsModal === 'backup' && (
        <SettingsModal title="Backup & Save Settings">
          <div className="space-y-3">
            <h3 className="text-xs font-medium text-muted uppercase">Backup</h3>
            <div className="grid grid-cols-2 gap-3">
              <div><label className="text-xs text-muted block mb-1">Backup After</label>
                <input type="text" inputMode="decimal" value={backupAfter} onChange={(e) => setBackupAfter(+e.target.value || 30)} className="input w-full text-sm" /></div>
              <div><label className="text-xs text-muted block mb-1">Unit</label>
                <select value={backupAfterUnit} onChange={(e) => setBackupAfterUnit(e.target.value)} className="input w-full text-sm">{TIME_UNITS.map(u => <option key={u}>{u}</option>)}</select></div>
            </div>
            <Toggle value={rollingBackup} onChange={setRollingBackup} label="Rolling Backup" />
            {rollingBackup && (
              <div><label className="text-xs text-muted block mb-1">Backup Count</label>
                <input type="text" inputMode="decimal" value={rollingBackupCount} onChange={(e) => setRollingBackupCount(+e.target.value || 3)} className="input w-full text-sm" min="1" /></div>
            )}
            <Toggle value={backupBeforeSave} onChange={setBackupBeforeSave} label="Backup Before Save" />
            <div className="border-t border-dark-border pt-3 mt-3">
              <h3 className="text-xs font-medium text-muted uppercase mb-2">Auto-Save</h3>
              <div className="grid grid-cols-2 gap-3">
                <div><label className="text-xs text-muted block mb-1">Save Every</label>
                  <input type="text" inputMode="decimal" value={saveEvery} onChange={(e) => setSaveEvery(+e.target.value || 0)} className="input w-full text-sm" /></div>
                <div><label className="text-xs text-muted block mb-1">Unit</label>
                  <select value={saveEveryUnit} onChange={(e) => setSaveEveryUnit(e.target.value)} className="input w-full text-sm">{TIME_UNITS.map(u => <option key={u}>{u}</option>)}</select></div>
              </div>
              <div><label className="text-xs text-muted block mb-1">Filename Prefix</label>
                <input type="text" inputMode="decimal" value={saveFilenamePrefix} onChange={(e) => setSaveFilenamePrefix(e.target.value)} className="input w-full text-sm" placeholder="model_" /></div>
            </div>
          </div>
        </SettingsModal>
      )}
    </div>
  );
}
