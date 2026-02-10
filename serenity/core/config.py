"""Configuration objects and helpers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from serenity.core.interfaces import ModelType
from serenity.core.enums import (
    DataType,
    EMAMode,
    GradientCheckpointingMethod,
    ImageFormat,
    LearningRateScaler,
    LearningRateScheduler,
    LossScaler,
    LossWeight,
    ModelFormat,
    Optimizer,
    PeftType,
    TimeUnit,
    TimestepDistribution,
)
from serenity.core.concept_config import ConceptConfig


class TrainingMethod(str, Enum):
    LORA = "lora"
    FINE_TUNE = "fine_tune"
    FINE_TUNE_VAE = "fine_tune_vae"
    EMBEDDING = "embedding"


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------

def _coerce_model_type(value: ModelType | str) -> ModelType:
    if isinstance(value, ModelType):
        return value

    normalized = str(value).lower()
    # Allow common aliases
    alias_map = {
        "z_image": ModelType.ZIMAGE,
        "zimage": ModelType.ZIMAGE,
        "z-image": ModelType.ZIMAGE,
        "sd_15": ModelType.SD15,
        "sd15_inpaint": ModelType.SD15_INPAINTING,
        "sd_15_inpainting": ModelType.SD15_INPAINTING,
        "sd15_inpainting": ModelType.SD15_INPAINTING,
        "sd_20": ModelType.SD20,
        "sd2": ModelType.SD20,
        "sd2_0": ModelType.SD20,
        "sd_20_base": ModelType.SD20_BASE,
        "sd20_base": ModelType.SD20_BASE,
        "sd_20_inpainting": ModelType.SD20_INPAINTING,
        "sd20_inpainting": ModelType.SD20_INPAINTING,
        "sd_20_depth": ModelType.SD20_DEPTH,
        "sd20_depth": ModelType.SD20_DEPTH,
        "sd_21": ModelType.SD21,
        "sd21": ModelType.SD21,
        "sd_21_base": ModelType.SD21_BASE,
        "sd21_base": ModelType.SD21_BASE,
        "sdxl_base": ModelType.SDXL_10_BASE,
        "sdxl_10_base_inpainting": ModelType.SDXL_INPAINTING,
        "sdxl_inpainting": ModelType.SDXL_INPAINTING,
        "sdxl_inpaint": ModelType.SDXL_INPAINTING,
        "sd3": ModelType.SD3,
        "sd_3": ModelType.SD3,
        "sd35": ModelType.SD35,
        "sd_35": ModelType.SD35,
        "sd3.5": ModelType.SD35,
        "stable_diffusion_3": ModelType.SD3,
        "stable_diffusion_35": ModelType.SD35,
        "stable_diffusion_3.5": ModelType.SD35,
        "flux2_klein_4b": ModelType.FLUX_2_KLEIN_4B,
        "flux2_klein_9b": ModelType.FLUX_2_KLEIN_9B,
        "flux_2_klein_4b": ModelType.FLUX_2_KLEIN_4B,
        "flux_2_klein_9b": ModelType.FLUX_2_KLEIN_9B,
        "flux_fill": ModelType.FLUX_FILL_DEV,
        "flux_fill_dev": ModelType.FLUX_FILL_DEV,
        "flux_fill_dev_1": ModelType.FLUX_FILL_DEV,
        "flux_2": ModelType.FLUX_2,
        "flux2": ModelType.FLUX_2,
        "wuerstchen": ModelType.WUERSTCHEN_2,
        "wuerstchen_2": ModelType.WUERSTCHEN_2,
        "stable_cascade": ModelType.STABLE_CASCADE_1,
        "stable_cascade_1": ModelType.STABLE_CASCADE_1,
        "pixart": ModelType.PIXART_ALPHA,
        "pixart_alpha": ModelType.PIXART_ALPHA,
        "pixart_sigma": ModelType.PIXART_SIGMA,
        "sana": ModelType.SANA,
        "hunyuan_video": ModelType.HUNYUAN_VIDEO,
        "hidream": ModelType.HI_DREAM_FULL,
        "hi_dream_full": ModelType.HI_DREAM_FULL,
        "chroma": ModelType.CHROMA_1,
        "chroma_1": ModelType.CHROMA_1,
        "ltx2": ModelType.LTX2,
        "ltx": ModelType.LTX2,
        "ltx_video": ModelType.LTX2,
        "ltxvideo": ModelType.LTX2,
    }
    if normalized in alias_map:
        return alias_map[normalized]

    try:
        return ModelType(normalized)
    except ValueError as exc:
        raise ValueError(f"Unknown model type: {value}") from exc


def _coerce_training_method(value: TrainingMethod | str) -> TrainingMethod:
    if isinstance(value, TrainingMethod):
        return value

    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"finetune", "full", "full_finetune", "full_fine_tune"}:
        normalized = TrainingMethod.FINE_TUNE.value
    if normalized in {"vae", "vae_finetune", "fine_tune_vae", "finetune_vae"}:
        normalized = TrainingMethod.FINE_TUNE_VAE.value
    try:
        return TrainingMethod(normalized)
    except ValueError as exc:
        raise ValueError(f"Unknown training method: {value}") from exc


def _coerce_enum(value: Any, enum_cls: type) -> Any:
    """Generic enum coercion: pass through if already the right type, else convert."""
    if isinstance(value, enum_cls):
        return value
    if value is None:
        return value
    try:
        return enum_cls(value)
    except (ValueError, KeyError):
        # Try case-insensitive lookup for str enums
        if isinstance(value, str):
            for member in enum_cls:
                if member.value.lower() == value.lower():
                    return member
        raise


# ---------------------------------------------------------------------------
# Sub-config: TrainModelPartConfig
# ---------------------------------------------------------------------------

@dataclass
class TrainModelPartConfig:
    """Per-component (text encoder, transformer, VAE, etc.) training settings."""

    model_name: str = ""
    include: bool = True
    train: bool = True
    stop_training_after: int | None = None
    stop_training_after_unit: TimeUnit | str = TimeUnit.NEVER
    learning_rate: float | None = None
    weight_dtype: DataType | str = DataType.FLOAT_32
    dropout_probability: float = 0.0
    train_embedding: bool = True
    attention_mask: bool = False
    guidance_scale: float = 1.0

    def __post_init__(self) -> None:
        self.stop_training_after_unit = _coerce_enum(
            self.stop_training_after_unit, TimeUnit
        )
        self.weight_dtype = _coerce_enum(self.weight_dtype, DataType)


# ---------------------------------------------------------------------------
# Sub-config: TrainOptimizerConfig
# ---------------------------------------------------------------------------

@dataclass
class TrainOptimizerConfig:
    """Optimizer hyperparameters."""

    optimizer: Optimizer | str = Optimizer.ADAMW

    # Core params
    adam_w_mode: bool = False
    alpha: float | None = None
    amsgrad: bool = False
    beta1: float | None = None
    beta2: float | None = None
    beta3: float | None = None
    bias_correction: bool = False
    block_wise: bool = False
    capturable: bool = False
    centered: bool = False
    clip_threshold: float | None = None
    d0: float | None = None
    d_coef: float | None = None
    dampening: float | None = None
    decay_rate: float | None = None
    decouple: bool = False
    differentiable: bool = False
    eps: float | None = None
    eps2: float | None = None
    foreach: bool | None = None
    fsdp_in_use: bool = False
    fused: bool = False
    fused_back_pass: bool = False
    growth_rate: float | None = None
    initial_accumulator_value: int | None = None
    initial_accumulator: float | None = None
    is_paged: bool = False
    log_every: int | None = None
    lr_decay: float | None = None
    max_unorm: float | None = None
    maximize: bool = False
    min_8bit_size: int | None = None
    quant_block_size: int | None = None
    momentum: float | None = None
    nesterov: bool = False
    no_prox: bool = False
    optim_bits: int | None = None
    percentile_clipping: int | None = None
    r: float | None = None
    relative_step: bool = False
    safeguard_warmup: bool = False
    scale_parameter: bool = False
    stochastic_rounding: bool = True
    use_bias_correction: bool = False
    use_triton: bool = False
    warmup_init: bool = False
    weight_decay: float | None = None
    weight_lr_power: float | None = None

    # Advanced flags
    decoupled_decay: bool = False
    fixed_decay: bool = False
    weight_decouple: bool = False
    rectify: bool = False
    degenerated_to_sgd: bool = False
    k: int | None = None
    xi: float | None = None
    n_sma_threshold: int | None = None
    ams_bound: bool = False
    adanorm: bool = False
    adam_debias: bool = False
    slice_p: int | None = None
    cautious: bool = False
    weight_decay_by_lr: bool = True
    prodigy_steps: int | None = None
    use_speed: bool = False
    split_groups: bool = True
    split_groups_mean: bool = True
    factored: bool = True
    factored_fp32: bool = True
    use_stableadamw: bool = True
    use_cautious: bool = False
    use_grams: bool = False
    use_adopt: bool = False
    d_limiter: bool | None = True
    use_schedulefree: bool | None = True
    use_orthograd: bool = False
    nnmf_factor: bool = False
    orthogonal_gradient: bool = False
    use_atan2: bool = False
    use_ademamix: bool = False
    beta3_ema: float | None = None
    alpha_grad: float | None = None
    beta1_warmup: int | None = None
    min_beta1: float | None = None
    simplified_ademamix: bool = False
    cautious_mask: bool = False
    grams_moment: bool = False
    kourkoutas_beta: bool = False
    k_warmup_steps: int | None = None
    schedulefree_c: float | None = None
    ns_steps: int | None = None

    # Muon settings
    muon_with_aux_adam: bool = False
    muon_hidden_layers: str | None = None
    muon_adam_regex: bool = False
    muon_adam_lr: float | None = None
    muon_te1_adam_lr: float | None = None
    muon_te2_adam_lr: float | None = None
    muon_adam_config: dict | None = None
    rms_rescaling: bool | None = True
    normuon_variant: bool = False
    beta2_normuon: float | None = None
    normuon_eps: float | None = None
    low_rank_ortho: bool = False
    ortho_rank: int | None = None
    accelerated_ns: bool = False
    cautious_wd: bool = False
    approx_mars: bool = False
    kappa_p: float | None = None
    auto_kappa_p: bool = False
    compile: bool = False

    def __post_init__(self) -> None:
        self.optimizer = _coerce_enum(self.optimizer, Optimizer)


# ---------------------------------------------------------------------------
# Main: TrainConfig
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Training configuration used by the pipeline and trainer."""

    # ---- Required fields (no default) ----
    model_type: ModelType | str
    training_method: TrainingMethod | str
    transformer_path: str
    output_dir: str
    concepts: list[Any]

    # ---- General settings ----
    debug_mode: bool = False
    debug_dir: str = "debug"
    workspace_dir: str = "workspace/run"
    cache_dir: str = "workspace-cache/run"

    # Tensorboard
    tensorboard: bool = True
    tensorboard_expose: bool = False
    tensorboard_always_on: bool = False
    tensorboard_port: int = 6006

    # Validation
    validation: bool = False
    validate_after: float = 1.0
    validate_after_unit: TimeUnit | str = TimeUnit.EPOCH
    continue_last_backup: bool = False

    # ---- Multi-GPU ----
    multi_gpu: bool = False
    device_indexes: str = ""
    fused_gradient_reduce: bool = True
    async_gradient_reduce: bool = True
    async_gradient_reduce_buffer: int = 100

    # ---- Model settings ----
    base_model_name: str = ""
    output_dtype: DataType | str = DataType.FLOAT_32
    output_model_format: ModelFormat | str = ModelFormat.SAFETENSORS
    output_model_destination: str = "models/model.safetensors"
    gradient_checkpointing: GradientCheckpointingMethod | str = "off"
    enable_async_offloading: bool = False
    enable_activation_offloading: bool = False
    layer_offload_fraction: float = 0.0
    force_circular_padding: bool = False
    compile: bool = False

    # ---- Data settings ----
    concept_file_name: str = "training_concepts/concepts.json"
    aspect_ratio_bucketing: bool = True
    latent_caching: bool = True
    clear_cache_before_training: bool = True

    # ---- Training settings ----
    learning_rate_scheduler: LearningRateScheduler | str = LearningRateScheduler.CONSTANT
    custom_learning_rate_scheduler: str | None = None
    scheduler_params: list[dict[str, str]] = field(default_factory=list)
    learning_rate: float = 1e-4
    learning_rate_warmup_steps: float = 200.0
    learning_rate_cycles: float = 1.0
    learning_rate_min_factor: float = 0.0
    epochs: int = 100
    max_train_steps: int | None = None
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    ema: EMAMode | str = EMAMode.OFF
    ema_decay: float = 0.999
    ema_update_step_interval: int = 5
    dataloader_threads: int = 2
    train_device: str = "cuda"
    temp_device: str = "cpu"
    train_dtype: DataType | str = DataType.FLOAT_16
    fallback_train_dtype: DataType | str = DataType.BFLOAT_16
    enable_autocast_cache: bool = True
    only_cache: bool = False
    resolution: str = "512"
    frames: str = "25"
    seed: int = 42

    # Loss settings
    mse_strength: float = 1.0
    mae_strength: float = 0.0
    log_cosh_strength: float = 0.0
    huber_strength: float = 0.0
    huber_delta: float = 1.0
    vb_loss_strength: float = 1.0
    loss_weight_fn: LossWeight | str = LossWeight.CONSTANT
    loss_weight_strength: float = 5.0
    dropout_probability: float = 0.0
    loss_scaler: LossScaler | str = LossScaler.NONE
    learning_rate_scaler: LearningRateScaler | str = LearningRateScaler.NONE
    clip_grad_norm: float | None = 1.0

    # Layer filter (for LoRA layer selection)
    layer_filter: str = ""
    layer_filter_preset: str = "full"
    layer_filter_regex: bool = False

    # ---- Noise settings ----
    offset_noise_weight: float = 0.0
    generalized_offset_noise: bool = False
    perturbation_noise_weight: float = 0.0
    rescale_noise_scheduler_to_zero_terminal_snr: bool = False
    force_v_prediction: bool = False
    force_epsilon_prediction: bool = False
    timestep_distribution: TimestepDistribution | str = TimestepDistribution.UNIFORM
    min_noising_strength: float = 0.0
    max_noising_strength: float = 1.0
    noising_weight: float = 0.0
    noising_bias: float = 0.0
    timestep_shift: float = 1.0
    dynamic_timestep_shifting: bool = False

    # ---- Model part configs ----
    unet: TrainModelPartConfig = field(default_factory=lambda: TrainModelPartConfig(
        train=True, stop_training_after=0, learning_rate=None,
    ))
    prior: TrainModelPartConfig = field(default_factory=lambda: TrainModelPartConfig(
        model_name="", train=True, stop_training_after=0, learning_rate=None,
    ))
    transformer: TrainModelPartConfig = field(default_factory=lambda: TrainModelPartConfig(
        model_name="", train=True, stop_training_after=0, learning_rate=None,
    ))
    text_encoder: TrainModelPartConfig = field(default_factory=lambda: TrainModelPartConfig(
        train=True, stop_training_after=30,
        stop_training_after_unit=TimeUnit.EPOCH, learning_rate=None,
    ))
    text_encoder_layer_skip: int = 0
    text_encoder_2: TrainModelPartConfig = field(default_factory=lambda: TrainModelPartConfig(
        train=True, stop_training_after=30,
        stop_training_after_unit=TimeUnit.EPOCH, learning_rate=None,
    ))
    text_encoder_2_layer_skip: int = 0
    text_encoder_2_sequence_length: int = 77
    text_encoder_3: TrainModelPartConfig = field(default_factory=lambda: TrainModelPartConfig(
        train=True, stop_training_after=30,
        stop_training_after_unit=TimeUnit.EPOCH, learning_rate=None,
    ))
    text_encoder_3_layer_skip: int = 0
    text_encoder_4: TrainModelPartConfig = field(default_factory=lambda: TrainModelPartConfig(
        train=True, stop_training_after=30,
        stop_training_after_unit=TimeUnit.EPOCH, learning_rate=None,
    ))
    text_encoder_4_layer_skip: int = 0
    vae: TrainModelPartConfig = field(default_factory=lambda: TrainModelPartConfig(
        model_name="",
    ))
    effnet_encoder: TrainModelPartConfig = field(default_factory=lambda: TrainModelPartConfig(
        model_name="",
    ))
    decoder: TrainModelPartConfig = field(default_factory=lambda: TrainModelPartConfig(
        model_name="",
    ))
    decoder_text_encoder: TrainModelPartConfig = field(
        default_factory=TrainModelPartConfig,
    )
    decoder_vqgan: TrainModelPartConfig = field(
        default_factory=TrainModelPartConfig,
    )

    # ---- Masked training ----
    masked_training: bool = False
    unmasked_probability: float = 0.1
    unmasked_weight: float = 0.1
    normalize_masked_area_loss: bool = False
    masked_prior_preservation_weight: float = 0.0
    custom_conditioning_image: bool = False

    # ---- Embedding settings ----
    embedding_learning_rate: float | None = None
    preserve_embedding_norm: bool = False
    embedding_weight_dtype: DataType | str = DataType.FLOAT_32

    # ---- LoRA settings ----
    peft_type: PeftType | str = PeftType.LORA
    lora_model_name: str = ""
    lora_rank: int = 16
    lora_alpha: float = 1.0
    lora_decompose: bool = False
    lora_decompose_norm_epsilon: bool = True
    lora_decompose_output_axis: bool = False
    lora_weight_dtype: DataType | str = DataType.FLOAT_32
    bundle_additional_embeddings: bool = True

    # ---- LoKR settings ----
    lokr_dim: int = 16
    lokr_alpha: float = 16.0
    lokr_factor: int = -1
    lokr_decompose_factor: int = -1
    lokr_decompose_both: bool = False
    lokr_use_tucker: bool = False
    lokr_weight_decompose: bool = False
    lokr_dora_on_output: bool = True
    lokr_rs_lora: bool = False
    lokr_full_matrix: bool = False

    # ---- OFT settings ----
    oft_block_size: int = 32
    oft_coft: bool = False
    coft_eps: float = 1e-4
    oft_block_share: bool = False

    # ---- Optimizer ----
    optimizer: TrainOptimizerConfig = field(default_factory=TrainOptimizerConfig)
    optimizer_defaults: dict[str, Any] = field(default_factory=dict)

    # ---- Sample settings ----
    sample_definition_file_name: str = "training_samples/samples.json"
    samples: list[Any] | None = None
    sample_after: float = 10.0
    sample_after_unit: TimeUnit | str = TimeUnit.MINUTE
    sample_skip_first: int = 0
    sample_image_format: ImageFormat | str = ImageFormat.JPG
    samples_to_tensorboard: bool = True
    non_ema_sampling: bool = True

    # ---- Backup settings ----
    backup_after: float = 30.0
    backup_after_unit: TimeUnit | str = TimeUnit.MINUTE
    rolling_backup: bool = False
    rolling_backup_count: int = 3
    backup_before_save: bool = True
    save_every: int = 0
    save_every_unit: TimeUnit | str = TimeUnit.NEVER
    save_skip_first: int = 0
    save_filename_prefix: str = ""

    def __post_init__(self) -> None:
        self.model_type = _coerce_model_type(self.model_type)
        self.training_method = _coerce_training_method(self.training_method)
        self.output_dir = str(self.output_dir)

        # Coerce enum fields
        _enum_fields = {
            "output_dtype": DataType,
            "output_model_format": ModelFormat,
            "gradient_checkpointing": GradientCheckpointingMethod,
            "learning_rate_scheduler": LearningRateScheduler,
            "train_dtype": DataType,
            "fallback_train_dtype": DataType,
            "loss_weight_fn": LossWeight,
            "loss_scaler": LossScaler,
            "learning_rate_scaler": LearningRateScaler,
            "timestep_distribution": TimestepDistribution,
            "ema": EMAMode,
            "validate_after_unit": TimeUnit,
            "sample_after_unit": TimeUnit,
            "sample_image_format": ImageFormat,
            "backup_after_unit": TimeUnit,
            "save_every_unit": TimeUnit,
            "peft_type": PeftType,
            "lora_weight_dtype": DataType,
            "embedding_weight_dtype": DataType,
        }
        for attr_name, enum_cls in _enum_fields.items():
            val = getattr(self, attr_name, None)
            if val is not None and not isinstance(val, enum_cls):
                setattr(self, attr_name, _coerce_enum(val, enum_cls))

        # Coerce sub-config dicts to dataclasses
        _part_fields = [
            "unet", "prior", "transformer",
            "text_encoder", "text_encoder_2", "text_encoder_3", "text_encoder_4",
            "vae", "effnet_encoder", "decoder", "decoder_text_encoder", "decoder_vqgan",
        ]
        for pf in _part_fields:
            val = getattr(self, pf, None)
            if isinstance(val, dict):
                setattr(self, pf, TrainModelPartConfig(**val))

        if isinstance(self.optimizer, dict):
            self.optimizer = TrainOptimizerConfig(**self.optimizer)

        # Coerce concept dicts
        if self.concepts:
            coerced = []
            for c in self.concepts:
                if isinstance(c, dict):
                    coerced.append(ConceptConfig.from_dict(c))
                else:
                    coerced.append(c)
            self.concepts = coerced


TrainerConfig = TrainConfig


@dataclass
class NoiseConfig:
    """Noise configuration options for SDXL-style offset noise."""

    offset_noise_weight: float = 0.0
    generalized_offset_noise: bool = False


def load_config(path: str | Path) -> TrainConfig:
    """Load a config file (JSON/YAML) into a TrainConfig.

    Automatically applies config migrations if the file's ``__version``
    is older than the current schema version.
    """
    from serenity.core.config_migration import migrate_config, needs_migration

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    data: dict[str, Any]
    if path.suffix.lower() in {".json"}:
        data = json.loads(path.read_text())
    elif path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("PyYAML required to load YAML config") from exc
        data = yaml.safe_load(path.read_text())
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")

    # Apply config migrations if needed
    if needs_migration(data):
        data = migrate_config(data)

    # Filter to only fields the dataclass accepts
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(TrainConfig)}
    filtered = {k: v for k, v in data.items() if k in valid_fields}

    return TrainConfig(**filtered)


def default_model_dir() -> Path:
    """Return the default model directory from env or sensible default."""
    return Path(os.environ.get("SERENITY_MODELS_DIR", os.path.expanduser("~/EriDiffusion/Models")))


def default_output_dir() -> Path:
    """Return the default output directory from env or sensible default."""
    return Path(os.environ.get("SERENITY_OUTPUT_DIR", "./output"))


__all__ = [
    "TrainingMethod",
    "TrainConfig",
    "TrainerConfig",
    "TrainModelPartConfig",
    "TrainOptimizerConfig",
    "ModelType",
    "NoiseConfig",
    "load_config",
    "default_model_dir",
    "default_output_dir",
]
