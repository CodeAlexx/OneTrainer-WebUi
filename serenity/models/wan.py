"""WAN 2.1/2.2 model adapter (UMT5 text encoder, WanTransformer3DModel, AutoencoderKLWan)."""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from typing import Any

import torch

from serenity.core.interfaces import ModelType
from serenity.models.base import BaseModelImpl

# WAN 2.1/2.2 14B has 40 transformer blocks (~700 MB each in BF16).
# Diffusers uses attn1 (self-attention), attn2 (cross-attention), ffn.net (feed-forward).
WAN_TRANSFORMER_LORA_TARGETS: tuple[str, ...] = (
    "attn1.to_q",
    "attn1.to_k",
    "attn1.to_v",
    "attn1.to_out.0",
    "attn2.to_q",
    "attn2.to_k",
    "attn2.to_v",
    "attn2.to_out.0",
    "ffn.net.0.proj",
    "ffn.net.2",
)

WAN_LAYER_PRESETS: dict[str, list[str]] = {
    "attn-mlp": ["attn1", "attn2", "ffn.net"],
    "attn-only": ["attn1", "attn2"],
    "blocks": ["blocks"],
    "full": [],
}


def _resolve_wan_vae_path(model_path: str, config_vae_path: str | None) -> str | None:
    """Find WAN VAE directory (from_pretrained needs a folder with config.json)."""
    if config_vae_path:
        candidate = Path(config_vae_path).expanduser()
        if candidate.exists():
            return str(candidate)

    # Check HuggingFace cache for ai-toolkit/wan2.1-vae
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    for repo_dir in sorted(hf_cache.glob("models--ai-toolkit--wan2.1*")):
        refs_main = repo_dir / "refs" / "main"
        if refs_main.exists():
            revision = refs_main.read_text().strip()
            snapshot = repo_dir / "snapshots" / revision
            if snapshot.exists():
                return str(snapshot)
        snapshots = sorted((repo_dir / "snapshots").glob("*")) if (repo_dir / "snapshots").exists() else []
        if snapshots:
            return str(snapshots[-1])

    # Check sibling directories relative to model_path
    parent = Path(model_path).parent
    for name in ("vae", "wan_vae", "wan2.1-vae"):
        candidate = parent / name
        if candidate.exists() and (candidate / "config.json").exists():
            return str(candidate)

    return None


def _resolve_wan_text_encoder_path(model_path: str, config_te_path: str | None) -> str | None:
    """Find UMT5 text encoder weights (single .safetensors file)."""
    if config_te_path:
        candidate = Path(config_te_path).expanduser()
        if candidate.exists():
            return str(candidate)

    # Common location in SwarmUI
    for name in ("umt5_xxl_fp16.safetensors", "umt5_xxl.safetensors"):
        candidate = Path(model_path).parent.parent / "clip" / name
        if candidate.exists():
            return str(candidate)
        # Also check one level up
        candidate = Path(model_path).parent / name
        if candidate.exists():
            return str(candidate)

    return None


class WanModel(BaseModelImpl):
    """Native WAN 2.1/2.2 adapter for Serenity training paths.

    Architecture: UMT5-XXL text encoder + WanTransformer3DModel (40 blocks, 14B) +
    AutoencoderKLWan (3D VAE, 16ch latent, 8x spatial / 4x temporal downsampling).
    Uses flow-matching objective.

    For first-pass training, only the low-noise transformer is loaded (not the
    dual high/low noise setup). This keeps the adapter simple while still testing
    the full 40-block, 14B model.
    """

    family = "wan"
    resolution_multiple = 16  # 8x spatial VAE * 2x2 patch
    train_module_attr = "transformer"
    flow_objective = True

    def __init__(self, *, wan_stage: str | None = None, is_vace: bool = False) -> None:
        super().__init__(model_type=ModelType.WAN)
        self.wan_stage = wan_stage
        self.is_vace = is_vace

    def load_pipeline(
        self,
        model_path: str,
        dtype: torch.dtype,
        train_device: torch.device,
        **kwargs: Any,
    ):
        del train_device
        from diffusers import AutoencoderKLWan, WanPipeline, WanTransformer3DModel
        from diffusers import FlowMatchEulerDiscreteScheduler
        from transformers import AutoTokenizer, UMT5EncoderModel

        model_block = kwargs.get("model_block") or {}

        # --- Load transformer from single file ---
        model_file = Path(model_path).expanduser()
        if not model_file.exists():
            raise FileNotFoundError(f"WAN transformer weights not found: {model_file}")

        # Detect VACE architecture by checking for vace_blocks keys.
        is_vace = self.is_vace
        if not is_vace:
            from safetensors import safe_open
            with safe_open(str(model_file), framework="pt") as f:
                is_vace = any(k.startswith("vace_blocks.") for k in f.keys())
        self.is_vace = is_vace

        if is_vace:
            from diffusers import WanVACETransformer3DModel
            print(f"[wan] loading VACE transformer from {model_file}")
            transformer = WanVACETransformer3DModel.from_single_file(
                str(model_file),
                torch_dtype=dtype,
            )
        else:
            print(f"[wan] loading transformer from {model_file}")
            transformer = WanTransformer3DModel.from_single_file(
                str(model_file),
                torch_dtype=dtype,
            )
        transformer.to("cpu")
        # from_single_file() calls accelerate.dispatch_model() which installs
        # AlignDevicesHook on every module. These hooks intercept .to() and
        # parameter access, preventing stagehand from managing block placement.
        # Strip all dispatch hooks and the device_map attribute.
        try:
            from accelerate.hooks import remove_hook_from_module
            for module in transformer.modules():
                remove_hook_from_module(module)
        except ImportError:
            pass
        if hasattr(transformer, "hf_device_map"):
            del transformer.hf_device_map

        # --- Load VAE ---
        vae_path = _resolve_wan_vae_path(
            model_path,
            kwargs.get("vae_path") or model_block.get("vae_path"),
        )
        if vae_path is None:
            raise FileNotFoundError(
                "WAN VAE not found. Download with: huggingface-cli download ai-toolkit/wan2.1-vae\n"
                "Or set model.vae_path in config."
            )
        print(f"[wan] loading VAE from {vae_path}")
        vae = AutoencoderKLWan.from_pretrained(
            vae_path,
            torch_dtype=dtype,
            local_files_only=True,
        )
        vae.to("cpu")

        # --- Load text encoder (UMT5-XXL) ---
        te_path = _resolve_wan_text_encoder_path(
            model_path,
            kwargs.get("text_encoder_path") or model_block.get("text_encoder_path"),
        )

        # UMT5 can be loaded from a single safetensors or a HF directory
        text_encoder = None
        tokenizer = None
        if te_path:
            te_file = Path(te_path)
            if te_file.is_file() and te_file.suffix == ".safetensors":
                # Load config from HF cache (just config.json, not weights), then load local weights
                print(f"[wan] loading UMT5 text encoder from {te_file}")
                from transformers import AutoConfig
                from safetensors.torch import load_file

                config_obj = AutoConfig.from_pretrained("google/umt5-xxl")
                text_encoder = UMT5EncoderModel(config_obj).to(dtype=dtype)
                state_dict = load_file(str(te_file))
                text_encoder.load_state_dict(state_dict, strict=False)
                text_encoder.to("cpu")
                tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")
            elif te_file.is_dir():
                print(f"[wan] loading UMT5 text encoder from {te_file}")
                text_encoder = UMT5EncoderModel.from_pretrained(
                    str(te_file),
                    torch_dtype=dtype,
                    local_files_only=True,
                )
                text_encoder.to("cpu")
                tokenizer = AutoTokenizer.from_pretrained(str(te_file))
        else:
            # Fall back to downloading google/umt5-xxl
            print("[wan] loading UMT5-XXL text encoder from HuggingFace")
            text_encoder = UMT5EncoderModel.from_pretrained(
                "google/umt5-xxl",
                torch_dtype=dtype,
            )
            text_encoder.to("cpu")
            tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")

        # --- Scheduler ---
        flow_shift = float(kwargs.get("flow_shift") or model_block.get("flow_shift", 5.0))
        scheduler = FlowMatchEulerDiscreteScheduler(shift=flow_shift)

        # --- Assemble pipeline ---
        if is_vace:
            from diffusers import WanVACEPipeline
            pipeline = WanVACEPipeline(
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                vae=vae,
                transformer=transformer,
                scheduler=scheduler,
            )
        else:
            pipeline = WanPipeline(
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                vae=vae,
                transformer=transformer,
                scheduler=scheduler,
            )
        pipeline.to("cpu")
        return pipeline

    def get_train_module(self, pipeline: Any) -> torch.nn.Module:
        return pipeline.transformer

    def encode_latents(self, pipeline: Any, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode pixels to latent space using WAN 3D VAE.

        WAN VAE expects 5D input (B, C, T, H, W). For image training we add a
        temporal dim. Latents are normalized using per-channel mean/std from the
        VAE config.
        """
        if pixel_values.ndim == 4:
            pixel_values = pixel_values.unsqueeze(2)  # (B, C, H, W) -> (B, C, 1, H, W)

        latents = pipeline.vae.encode(pixel_values).latent_dist.sample()

        # WAN-specific normalization: (latents - mean) / std
        vae_config = pipeline.vae.config
        latents_mean = getattr(vae_config, "latents_mean", None)
        latents_std = getattr(vae_config, "latents_std", None)

        if latents_mean is not None and latents_std is not None:
            mean = torch.tensor(latents_mean, device=latents.device, dtype=latents.dtype)
            std = torch.tensor(latents_std, device=latents.device, dtype=latents.dtype)
            # Reshape for broadcasting: (1, C, 1, 1, 1) to match (B, C, T, H, W)
            shape = [1] * latents.dim()
            shape[1] = len(latents_mean)  # channel dim
            mean = mean.view(*shape)
            std = std.view(*shape)
            latents = (latents - mean) / std

        return latents

    def encode_prompt_features(
        self,
        pipeline: Any,
        prompt: str,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Encode text using UMT5-XXL text encoder."""
        text_encoder = pipeline.text_encoder
        tokenizer = pipeline.tokenizer

        if text_encoder is None or tokenizer is None:
            return (
                torch.zeros((1, 512, 4096), device=device, dtype=torch.float32),
                None,
                None,
            )

        if text_encoder.device != device:
            text_encoder.to(device)

        tok_out = tokenizer(
            [prompt],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        tokens = tok_out.input_ids.to(device)
        attention_mask = tok_out.attention_mask.to(device)

        outputs = text_encoder(
            input_ids=tokens,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        prompt_embeds = outputs.last_hidden_state

        return prompt_embeds, None, attention_mask

    def cache_prompt_device(self, pipeline: Any, train_device: torch.device) -> torch.device:
        del pipeline
        return train_device

    def move_text_encoders_to_device(self, pipeline: Any, device: torch.device) -> None:
        text_encoder = getattr(pipeline, "text_encoder", None)
        if text_encoder is not None:
            text_encoder.to(device)

    def offload_text_encoders(self, pipeline: Any) -> None:
        text_encoder = getattr(pipeline, "text_encoder", None)
        if text_encoder is not None:
            with suppress(Exception):
                text_encoder.to("cpu")


__all__ = [
    "WanModel",
    "WAN_TRANSFORMER_LORA_TARGETS",
    "WAN_LAYER_PRESETS",
]
