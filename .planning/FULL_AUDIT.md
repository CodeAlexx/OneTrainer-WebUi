# Serenity Full Codebase Audit

- Scope: `/home/alex/serenity/serenity/`
- Files scanned: `479`
- Issues reported: `497`
- Generated: `2026-02-09T04:48:47.950243+00:00`

## Phase 5C: Diffusers Import Audit (2026-02-08)

### Summary
44 diffusers imports across 16 files. ALL are actively used and cannot be removed without breaking functionality.

### KEEP — Model Architecture (nn.Module subclasses)
- `AutoencoderKL` (sd3, flux1, flux2_klein, zimage, inference/engine, inference/models/loader)
- `AutoencoderKLFlux2` (flux2_klein)
- `AutoencoderKLQwenImage` (qwen)
- `FluxTransformer2DModel` (flux1, flux2_klein, inference/models/flux, inference/models/chroma)
- `Flux2Transformer2DModel` (flux2_transformer — wrapped in Serenity compatibility class)
- `SD3Transformer2DModel` (sd3, inference/models/sd3)
- `ZImageTransformer2DModel` (zimage)
- `QwenImageTransformer2DModel` (qwen)
- `UNet2DConditionModel` (inference/models/sd15, inference/models/sdxl)
- `WanTransformer3DModel` (inference/models/wan)
- `BasicTransformerBlock` (training/checkpointing — gradient checkpointing layer ID)
- `GGUFLinear`, `dequantize_gguf_tensor` (models/quantization — GGUF support)

### KEEP — Pipeline Classes (used for model loading via from_pretrained)
These load pretrained models from HuggingFace. Removing requires major refactor to load components individually.
- `StableDiffusionPipeline` (sd15), `StableDiffusionXLPipeline` (sdxl)
- `StableDiffusion3Pipeline` (sd3), `FluxPipeline` (flux1)
- `ChromaPipeline` (chroma), `HunyuanVideoPipeline` (hunyuan_video)
- `ZImagePipeline` (zimage), `DiffusionPipeline` (flux2_klein — fallback loader)
- Dynamic `getattr(diffusers, class_name)` in flux1, flux2, qwen, ltx2, sampling/sampler

### KEEP — Schedulers (used for inference denoising + config loading)
- `FlowMatchEulerDiscreteScheduler` (flux2_klein — denoising loop, sd3/zimage/qwen — config loading)

### KEEP — Quantization Configs (both versions needed for different components)
- `BitsAndBytesConfig` from diffusers (qwen — for transformer quantization)
- `QuantoConfig` from diffusers (qwen — for FP8 quantization)

### FLAGGED — Optional Dependencies (do NOT remove)
- `bitsandbytes`: models/quantization.py, training/optimizers.py, models/qwen.py, inference/quantization/bnb.py
- `xformers`: inference/attention/xformers_attn.py, inference/attention/backends.py
- `fastapi`: ui/web/server.py, ui/web/api.py, ui/web/websocket.py

### Conclusion
No diffusers imports are removable. All pipeline/scheduler imports are integral to model loading.
Future refactor opportunity: replace Pipeline.from_pretrained() with direct component loading.

---

## Findings

FILE: serenity/training/lora_manager.py
LINE: 182
CATEGORY: [1]
SEVERITY: HIGH
ISSUE: External dependency `peft` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/training/lora_manager.py
LINE: 228
CATEGORY: [1]
SEVERITY: HIGH
ISSUE: External dependency `peft` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/training/lora_manager.py
LINE: 236
CATEGORY: [1]
SEVERITY: HIGH
ISSUE: External dependency `peft` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/training/lycoris_manager.py
LINE: 18
CATEGORY: [1]
SEVERITY: HIGH
ISSUE: External dependency `lycoris` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/.eri-rpg/runs/6213c88d7205.json
LINE: 90
CATEGORY: [2]
SEVERITY: HIGH
ISSUE: Hardcoded absolute path
DETAIL: Found literal path `/home/alex/OneTrainer/eritrainer`; move to config/env and path resolver.

FILE: serenity/.eri-rpg/runs/b7130c1f65a0.json
LINE: 120
CATEGORY: [2]
SEVERITY: HIGH
ISSUE: Hardcoded absolute path
DETAIL: Found literal path `/home/alex/OneTrainer/eritrainer`; move to config/env and path resolver.

FILE: serenity/models/flux1.py
LINE: 132
CATEGORY: [2]
SEVERITY: HIGH
ISSUE: Hardcoded absolute path
DETAIL: Found literal path `/home/alex/eriui/comfyui/ComfyUI/models/clip/t5xxl_fp16.safetensors`; move to config/env and path resolver.

FILE: serenity/models/flux1.py
LINE: 133
CATEGORY: [2]
SEVERITY: HIGH
ISSUE: Hardcoded absolute path
DETAIL: Found literal path `/home/alex/eriui/comfyui/ComfyUI/models/clip/t5xxl_enconly.safetensors`; move to config/env and path resolver.

FILE: serenity/ui/tabs/inference.py
LINE: 107
CATEGORY: [2]
SEVERITY: HIGH
ISSUE: Hardcoded absolute path
DETAIL: Found literal path `/home/alex/EriDiffusion/Models`; move to config/env and path resolver.

FILE: serenity/ui/web/api.py
LINE: 30
CATEGORY: [2]
SEVERITY: HIGH
ISSUE: Hardcoded absolute path
DETAIL: Found literal path `/home/alex/serenity/output`; move to config/env and path resolver.

FILE: serenity/ui/web/api.py
LINE: 70
CATEGORY: [2]
SEVERITY: HIGH
ISSUE: Hardcoded absolute path
DETAIL: Found literal path `/home/alex/serenity/output`; move to config/env and path resolver.

FILE: serenity/ui/web/server.py
LINE: 23
CATEGORY: [2]
SEVERITY: HIGH
ISSUE: Hardcoded absolute path
DETAIL: Found literal path `/home/alex/EriDiffusion/Models`; move to config/env and path resolver.

FILE: serenity/ui/web/server.py
LINE: 24
CATEGORY: [2]
SEVERITY: HIGH
ISSUE: Hardcoded absolute path
DETAIL: Found literal path `/home/alex/serenity/output`; move to config/env and path resolver.

FILE: serenity/cli/native_diffusion.py
LINE: 1
CATEGORY: [5]
SEVERITY: HIGH
ISSUE: Very large source file (god module)
DETAIL: 2471 lines indicates mixed responsibilities and poor modularity.

FILE: serenity/cli/native_flux2.py
LINE: 1
CATEGORY: [5]
SEVERITY: HIGH
ISSUE: Very large source file (god module)
DETAIL: 1230 lines indicates mixed responsibilities and poor modularity.

FILE: serenity/models/flux2_klein.py
LINE: 1
CATEGORY: [5]
SEVERITY: HIGH
ISSUE: Very large source file (god module)
DETAIL: 1528 lines indicates mixed responsibilities and poor modularity.

FILE: serenity/sampling/sampler.py
LINE: 1
CATEGORY: [5]
SEVERITY: HIGH
ISSUE: Very large source file (god module)
DETAIL: 1205 lines indicates mixed responsibilities and poor modularity.

FILE: serenity/checkpoint/conversion.py
LINE: 101
CATEGORY: [6]
SEVERITY: HIGH
ISSUE: Exception swallowed with pass
DETAIL: Broad exception catches and discards errors.

FILE: serenity/checkpoint/conversion.py
LINE: 101
CATEGORY: [6]
SEVERITY: HIGH
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception with pass silently drops failure.

FILE: serenity/inference/attention/backends.py
LINE: 82
CATEGORY: [6]
SEVERITY: HIGH
ISSUE: Exception swallowed with pass
DETAIL: Broad exception catches and discards errors.

FILE: serenity/inference/attention/backends.py
LINE: 82
CATEGORY: [6]
SEVERITY: HIGH
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception with pass silently drops failure.

FILE: serenity/inference/attention/sdp.py
LINE: 50
CATEGORY: [6]
SEVERITY: HIGH
ISSUE: Exception swallowed with pass
DETAIL: Broad exception catches and discards errors.

FILE: serenity/inference/attention/sdp.py
LINE: 50
CATEGORY: [6]
SEVERITY: HIGH
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception with pass silently drops failure.

FILE: serenity/inference/lora/merge.py
LINE: 48
CATEGORY: [6]
SEVERITY: HIGH
ISSUE: Exception swallowed with pass
DETAIL: Broad exception catches and discards errors.

FILE: serenity/inference/lora/merge.py
LINE: 48
CATEGORY: [6]
SEVERITY: HIGH
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception with pass silently drops failure.

FILE: serenity/inference/memory/pinned.py
LINE: 44
CATEGORY: [6]
SEVERITY: HIGH
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception with pass silently drops failure.

FILE: serenity/inference/memory/streams.py
LINE: 253
CATEGORY: [6]
SEVERITY: HIGH
ISSUE: Exception swallowed with pass
DETAIL: Broad exception catches and discards errors.

FILE: serenity/inference/memory/streams.py
LINE: 253
CATEGORY: [6]
SEVERITY: HIGH
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception with pass silently drops failure.

FILE: serenity/models/flux2_klein.py
LINE: 1525
CATEGORY: [6]
SEVERITY: HIGH
ISSUE: Exception swallowed with pass
DETAIL: Broad exception catches and discards errors.

FILE: serenity/models/flux2_klein.py
LINE: 1525
CATEGORY: [6]
SEVERITY: HIGH
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception with pass silently drops failure.

FILE: serenity/models/zimage.py
LINE: 132
CATEGORY: [6]
SEVERITY: HIGH
ISSUE: Exception swallowed with pass
DETAIL: Broad exception catches and discards errors.

FILE: serenity/models/zimage.py
LINE: 132
CATEGORY: [6]
SEVERITY: HIGH
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception with pass silently drops failure.

FILE: serenity/training/lora_manager.py
LINE: 166
CATEGORY: [6]
SEVERITY: HIGH
ISSUE: Exception swallowed with pass
DETAIL: Broad exception catches and discards errors.

FILE: serenity/training/lora_manager.py
LINE: 166
CATEGORY: [6]
SEVERITY: HIGH
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception with pass silently drops failure.

FILE: serenity/ui/app.py
LINE: 392
CATEGORY: [6]
SEVERITY: HIGH
ISSUE: Exception swallowed with pass
DETAIL: Broad exception catches and discards errors.

FILE: serenity/ui/app.py
LINE: 392
CATEGORY: [6]
SEVERITY: HIGH
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception with pass silently drops failure.

FILE: serenity/ui/state.py
LINE: 104
CATEGORY: [6]
SEVERITY: HIGH
ISSUE: Exception swallowed with pass
DETAIL: Broad exception catches and discards errors.

FILE: serenity/ui/state.py
LINE: 104
CATEGORY: [6]
SEVERITY: HIGH
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception with pass silently drops failure.

FILE: serenity/ui/tabs/concepts.py
LINE: 65
CATEGORY: [6]
SEVERITY: HIGH
ISSUE: Exception swallowed with pass
DETAIL: Broad exception catches and discards errors.

FILE: serenity/ui/tabs/concepts.py
LINE: 65
CATEGORY: [6]
SEVERITY: HIGH
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception with pass silently drops failure.

FILE: serenity/ui/tabs/inference.py
LINE: 167
CATEGORY: [6]
SEVERITY: HIGH
ISSUE: Exception swallowed with pass
DETAIL: Broad exception catches and discards errors.

FILE: serenity/ui/tabs/inference.py
LINE: 167
CATEGORY: [6]
SEVERITY: HIGH
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception with pass silently drops failure.

FILE: serenity/ui/tabs/inference.py
LINE: 453
CATEGORY: [6]
SEVERITY: HIGH
ISSUE: Exception swallowed with pass
DETAIL: Broad exception catches and discards errors.

FILE: serenity/ui/tabs/inference.py
LINE: 453
CATEGORY: [6]
SEVERITY: HIGH
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception with pass silently drops failure.

FILE: serenity/ui/tabs/inference.py
LINE: 465
CATEGORY: [6]
SEVERITY: HIGH
ISSUE: Exception swallowed with pass
DETAIL: Broad exception catches and discards errors.

FILE: serenity/ui/tabs/inference.py
LINE: 465
CATEGORY: [6]
SEVERITY: HIGH
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception with pass silently drops failure.

FILE: serenity/ui/tabs/inference.py
LINE: 607
CATEGORY: [6]
SEVERITY: HIGH
ISSUE: Exception swallowed with pass
DETAIL: Broad exception catches and discards errors.

FILE: serenity/ui/tabs/inference.py
LINE: 607
CATEGORY: [6]
SEVERITY: HIGH
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception with pass silently drops failure.

FILE: serenity/ui/tabs/inference.py
LINE: 684
CATEGORY: [6]
SEVERITY: HIGH
ISSUE: Exception swallowed with pass
DETAIL: Broad exception catches and discards errors.

FILE: serenity/ui/tabs/inference.py
LINE: 684
CATEGORY: [6]
SEVERITY: HIGH
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception with pass silently drops failure.

FILE: serenity/ui/tabs/inference.py
LINE: 720
CATEGORY: [6]
SEVERITY: HIGH
ISSUE: Exception swallowed with pass
DETAIL: Broad exception catches and discards errors.

FILE: serenity/ui/tabs/inference.py
LINE: 720
CATEGORY: [6]
SEVERITY: HIGH
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception with pass silently drops failure.

FILE: serenity/ui/tabs/lora.py
LINE: 236
CATEGORY: [6]
SEVERITY: HIGH
ISSUE: Exception swallowed with pass
DETAIL: Broad exception catches and discards errors.

FILE: serenity/ui/tabs/lora.py
LINE: 236
CATEGORY: [6]
SEVERITY: HIGH
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception with pass silently drops failure.

FILE: serenity/ui/tabs/sampling.py
LINE: 127
CATEGORY: [6]
SEVERITY: HIGH
ISSUE: Exception swallowed with pass
DETAIL: Broad exception catches and discards errors.

FILE: serenity/ui/tabs/sampling.py
LINE: 127
CATEGORY: [6]
SEVERITY: HIGH
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception with pass silently drops failure.

FILE: serenity/ui/theme.py
LINE: 72
CATEGORY: [6]
SEVERITY: HIGH
ISSUE: Exception swallowed with pass
DETAIL: Broad exception catches and discards errors.

FILE: serenity/ui/theme.py
LINE: 72
CATEGORY: [6]
SEVERITY: HIGH
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception with pass silently drops failure.

FILE: serenity/ui/theme.py
LINE: 86
CATEGORY: [6]
SEVERITY: HIGH
ISSUE: Exception swallowed with pass
DETAIL: Broad exception catches and discards errors.

FILE: serenity/ui/theme.py
LINE: 86
CATEGORY: [6]
SEVERITY: HIGH
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception with pass silently drops failure.

FILE: serenity/core/trainer.py
LINE: 557-603
CATEGORY: [7]
SEVERITY: HIGH
ISSUE: NaN/Inf loss checked only after backward pass
DETAIL: The loop calls backward() before NaN validation. Invalid gradients can already contaminate params/optimizer state before skip/abort logic. Validate finite loss before backward and zero gradients on invalid steps.

FILE: serenity/inference/engine.py
LINE: 641-643
CATEGORY: [7]
SEVERITY: HIGH
ISSUE: Flow sigma schedule computes log(0)
DETAIL: torch.linspace(1.0, 0.0, 1000) includes zero, and sigmas.log() stores -inf in _log_sigmas. This can destabilize sigma-to-timestep mapping and downstream denoising math.

FILE: serenity/training/flux2/base.py
LINE: 463-469
CATEGORY: [7]
SEVERITY: HIGH
ISSUE: Velocity weighting uses raw discrete timestep as sigma
DETAIL: sigma is set to integer timesteps (0..999), then used in sqrt/snr formulas meant for normalized noise scale. This produces invalid/negative SNR weighting. Convert timesteps to normalized sigma first.

FILE: serenity/training/flux2/edit_trainer.py
LINE: 407-411
CATEGORY: [7]
SEVERITY: HIGH
ISSUE: Concat-conditioning output slicing uses incorrect channel heuristic
DETAIL: Prediction channels are split via packed_pred.shape[-1] // 2, assuming a 1:1 concat layout. This is incorrect for variable reference counts/channels and can truncate or misalign outputs.

FILE: serenity/training/flux2/image_trainer.py
LINE: 200-213
CATEGORY: [7]
SEVERITY: HIGH
ISSUE: Caption processing references undefined trainer state
DETAIL: _process_captions() reads self.training but Flux2ImageTrainer is not nn.Module and never defines training. This path can raise AttributeError when captions are processed on-the-fly.

FILE: serenity/adapters/__init__.py
LINE: 81-109
CATEGORY: [8]
SEVERITY: HIGH
ISSUE: Non-vanilla adapter types route through LyCORIS wrapper manager
DETAIL: BaseAdapter._ensure_manager() constructs LyCORISManager for DoRA/LoKr/LoHa/OFT/IA3/BOFT/GLoRA/DyLoRA, so these are not native Serenity implementations and parity is effectively delegated to an external library.

FILE: serenity/cli/native_diffusion.py
LINE: 1118-1355
CATEGORY: [8]
SEVERITY: HIGH
ISSUE: Masked training config is not applied in native diffusion loss path
DETAIL: masked_training/unmasked_weight/normalize_masked_area_loss exist in config, but _compute_loss does not ingest any spatial mask or call masked loss helpers.

FILE: serenity/cli/native_flux2.py
LINE: 1130-1228
CATEGORY: [8]
SEVERITY: HIGH
ISSUE: Native FLUX2 loop lacks optimizer/scheduler state save-resume parity
DETAIL: Loop saves adapter/full transformer checkpoints but does not persist and restore optimizer/lr-scheduler/global-step state, so resume cannot reproduce training continuation semantics.

FILE: serenity/training/lora_manager.py
LINE: 182-195
CATEGORY: [8]
SEVERITY: HIGH
ISSUE: Standard LoRA training is PEFT-backed instead of native torch
DETAIL: Vanilla LoRA path depends on peft.LoraConfig and PEFT adapter internals. Serenity parity target was native implementation; this adds external dependency and indirection for a trivial adapter path.

FILE: serenity/.eri-rpg/context/work_add_gradient_clipping_to_training_loop.md
LINE: 2295
CATEGORY: [10]
SEVERITY: HIGH
ISSUE: Use of eval()
DETAIL: eval() is code-execution surface.

FILE: serenity/cli/native_diffusion.py
LINE: 2234
CATEGORY: [10]
SEVERITY: HIGH
ISSUE: Use of eval()
DETAIL: eval() is code-execution surface.

FILE: serenity/core/interfaces.py
LINE: 235
CATEGORY: [10]
SEVERITY: HIGH
ISSUE: Use of eval()
DETAIL: eval() is code-execution surface.

FILE: serenity/inference/engine.py
LINE: 594
CATEGORY: [10]
SEVERITY: HIGH
ISSUE: Use of eval()
DETAIL: eval() is code-execution surface.

FILE: serenity/inference/models/chroma.py
LINE: 100
CATEGORY: [10]
SEVERITY: HIGH
ISSUE: Use of eval()
DETAIL: eval() is code-execution surface.

FILE: serenity/inference/models/flux.py
LINE: 171
CATEGORY: [10]
SEVERITY: HIGH
ISSUE: Use of eval()
DETAIL: eval() is code-execution surface.

FILE: serenity/inference/models/loader.py
LINE: 158
CATEGORY: [10]
SEVERITY: HIGH
ISSUE: Use of eval()
DETAIL: eval() is code-execution surface.

FILE: serenity/inference/models/sd15.py
LINE: 101
CATEGORY: [10]
SEVERITY: HIGH
ISSUE: Use of eval()
DETAIL: eval() is code-execution surface.

FILE: serenity/inference/models/sd3.py
LINE: 113
CATEGORY: [10]
SEVERITY: HIGH
ISSUE: Use of eval()
DETAIL: eval() is code-execution surface.

FILE: serenity/inference/models/sdxl.py
LINE: 161
CATEGORY: [10]
SEVERITY: HIGH
ISSUE: Use of eval()
DETAIL: eval() is code-execution surface.

FILE: serenity/inference/models/wan.py
LINE: 98
CATEGORY: [10]
SEVERITY: HIGH
ISSUE: Use of eval()
DETAIL: eval() is code-execution surface.

FILE: serenity/inference/text/clip.py
LINE: 149
CATEGORY: [10]
SEVERITY: HIGH
ISSUE: Use of eval()
DETAIL: eval() is code-execution surface.

FILE: serenity/inference/text/gemma.py
LINE: 67
CATEGORY: [10]
SEVERITY: HIGH
ISSUE: Use of eval()
DETAIL: eval() is code-execution surface.

FILE: serenity/inference/text/qwen_enc.py
LINE: 67
CATEGORY: [10]
SEVERITY: HIGH
ISSUE: Use of eval()
DETAIL: eval() is code-execution surface.

FILE: serenity/inference/text/t5.py
LINE: 67
CATEGORY: [10]
SEVERITY: HIGH
ISSUE: Use of eval()
DETAIL: eval() is code-execution surface.

FILE: serenity/inference/vae/decoder.py
LINE: 78
CATEGORY: [10]
SEVERITY: HIGH
ISSUE: Use of eval()
DETAIL: eval() is code-execution surface.

FILE: serenity/inference/vae/encoder.py
LINE: 67
CATEGORY: [10]
SEVERITY: HIGH
ISSUE: Use of eval()
DETAIL: eval() is code-execution surface.

FILE: serenity/models/base.py
LINE: 23
CATEGORY: [10]
SEVERITY: HIGH
ISSUE: Use of eval()
DETAIL: eval() is code-execution surface.

FILE: serenity/models/setup.py
LINE: 522
CATEGORY: [10]
SEVERITY: HIGH
ISSUE: Use of eval()
DETAIL: eval() is code-execution surface.

FILE: serenity/tests/inference/test_memory.py
LINE: 362
CATEGORY: [10]
SEVERITY: HIGH
ISSUE: Use of eval()
DETAIL: eval() is code-execution surface.

FILE: serenity/tests/test_flux2_klein_forward.py
LINE: 170
CATEGORY: [10]
SEVERITY: HIGH
ISSUE: Use of eval()
DETAIL: eval() is code-execution surface.

FILE: serenity/training/flux2/base.py
LINE: 750
CATEGORY: [10]
SEVERITY: HIGH
ISSUE: Use of eval()
DETAIL: eval() is code-execution surface.

FILE: serenity/training/text_encoder.py
LINE: 108
CATEGORY: [10]
SEVERITY: HIGH
ISSUE: Use of eval()
DETAIL: eval() is code-execution surface.

FILE: serenity/training/vae_training.py
LINE: 54
CATEGORY: [10]
SEVERITY: HIGH
ISSUE: Use of eval()
DETAIL: eval() is code-execution surface.

FILE: serenity/inference/attention/xformers_attn.py
LINE: 30
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `xformers` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/inference/engine.py
LINE: 581
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `diffusers` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/inference/models/chroma.py
LINE: 71
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `diffusers` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/inference/models/flux.py
LINE: 133
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `diffusers` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/inference/models/flux.py
LINE: 331
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `diffusers` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/inference/models/flux.py
LINE: 393
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `diffusers` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/inference/models/loader.py
LINE: 234
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `diffusers` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/inference/models/sd15.py
LINE: 77
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `diffusers` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/inference/models/sd3.py
LINE: 69
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `diffusers` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/inference/models/sdxl.py
LINE: 137
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `diffusers` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/inference/models/sdxl.py
LINE: 261
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `diffusers` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/inference/models/wan.py
LINE: 55
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `diffusers` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/inference/quantization/bnb.py
LINE: 19
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `bitsandbytes` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/inference/quantization/bnb.py
LINE: 20
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `bitsandbytes` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/models/chroma.py
LINE: 66
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `diffusers` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/models/flux1.py
LINE: 35
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `diffusers` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/models/flux1.py
LINE: 202
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `diffusers` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/models/flux2.py
LINE: 30
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `diffusers` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/models/flux2_klein.py
LINE: 1038
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `diffusers` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/models/flux2_klein.py
LINE: 1129
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `diffusers` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/models/flux2_klein.py
LINE: 1151
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `diffusers` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/models/flux2_klein.py
LINE: 1162
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `diffusers` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/models/flux2_klein.py
LINE: 1173
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `diffusers` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/models/flux2_klein.py
LINE: 1183
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `diffusers` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/models/flux2_klein.py
LINE: 1255
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `diffusers` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/models/flux2_klein.py
LINE: 1281
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `diffusers` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/models/flux2_klein.py
LINE: 1407
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `diffusers` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/models/flux2_transformer.py
LINE: 10
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `diffusers` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/models/flux2_transformer.py
LINE: 13
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `diffusers` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/models/hunyuan_video.py
LINE: 86
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `diffusers` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/models/ltx2.py
LINE: 345
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `diffusers` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/models/quantization.py
LINE: 39
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `bitsandbytes` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/models/quantization.py
LINE: 50
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `diffusers` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/models/qwen.py
LINE: 71
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `diffusers` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/models/qwen.py
LINE: 176
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `diffusers` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/models/qwen.py
LINE: 223
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `bitsandbytes` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/models/qwen.py
LINE: 227
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `diffusers` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/models/qwen.py
LINE: 243
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `diffusers` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/models/qwen.py
LINE: 266
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `diffusers` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/models/sd15.py
LINE: 33
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `diffusers` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/models/sd3.py
LINE: 121
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `diffusers` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/models/sd3.py
LINE: 172
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `diffusers` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/models/sdxl.py
LINE: 33
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `diffusers` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/models/zimage.py
LINE: 86
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `diffusers` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/models/zimage.py
LINE: 158
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `diffusers` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/sampling/sampler.py
LINE: 183
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `diffusers` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/tests/test_flux2_klein_integration.py
LINE: 60
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `diffusers` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/tests/test_flux2_klein_integration.py
LINE: 106
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `diffusers` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/training/checkpointing.py
LINE: 428
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `diffusers` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/training/optimizers.py
LINE: 119
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `bitsandbytes` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/ui/web/api.py
LINE: 13
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `fastapi` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/ui/web/server.py
LINE: 13
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `fastapi` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/ui/web/server.py
LINE: 14
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `fastapi` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/ui/web/server.py
LINE: 15
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `fastapi` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/ui/web/server.py
LINE: 117
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `fastapi` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/ui/web/websocket.py
LINE: 8
CATEGORY: [1]
SEVERITY: MEDIUM
ISSUE: External dependency `fastapi` in core code
DETAIL: Dependency can likely be replaced with native implementation for this scope.

FILE: serenity/tests/test_command_routing.py
LINE: 19
CATEGORY: [2]
SEVERITY: MEDIUM
ISSUE: Hardcoded absolute path
DETAIL: Found literal path `/tmp/mock-model`; move to config/env and path resolver.

FILE: serenity/tests/test_command_routing.py
LINE: 90
CATEGORY: [2]
SEVERITY: MEDIUM
ISSUE: Hardcoded absolute path
DETAIL: Found literal path `/tmp/out`; move to config/env and path resolver.

FILE: serenity/tests/test_command_routing.py
LINE: 103
CATEGORY: [2]
SEVERITY: MEDIUM
ISSUE: Hardcoded absolute path
DETAIL: Found literal path `/tmp/out`; move to config/env and path resolver.

FILE: serenity/tests/test_flux2_native_training_modes.py
LINE: 88
CATEGORY: [2]
SEVERITY: MEDIUM
ISSUE: Hardcoded absolute path
DETAIL: Found literal path `/tmp/mock`; move to config/env and path resolver.

FILE: serenity/tests/test_flux2_native_training_modes.py
LINE: 106
CATEGORY: [2]
SEVERITY: MEDIUM
ISSUE: Hardcoded absolute path
DETAIL: Found literal path `/tmp/mock`; move to config/env and path resolver.

FILE: serenity/tests/test_native_diffusion_training.py
LINE: 364
CATEGORY: [2]
SEVERITY: MEDIUM
ISSUE: Hardcoded absolute path
DETAIL: Found literal path `/tmp/mock`; move to config/env and path resolver.

FILE: serenity/tests/test_native_diffusion_training.py
LINE: 399
CATEGORY: [2]
SEVERITY: MEDIUM
ISSUE: Hardcoded absolute path
DETAIL: Found literal path `/tmp/mock`; move to config/env and path resolver.

FILE: serenity/tests/test_native_diffusion_training.py
LINE: 436
CATEGORY: [2]
SEVERITY: MEDIUM
ISSUE: Hardcoded absolute path
DETAIL: Found literal path `/tmp/mock`; move to config/env and path resolver.

FILE: serenity/models/base.py
LINE: 14
CATEGORY: [3]
SEVERITY: MEDIUM
ISSUE: Stub class `BaseModelImpl` with no-op methods
DETAIL: All methods are pass/return self/return empty; likely dead abstraction.

FILE: serenity/cli/native_diffusion.py
LINE: 520
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `_is_condlabel_image`
DETAIL: Near-identical implementation appears in multiple files: serenity/cli/native_flux2.py:_is_condlabel_image:636

FILE: serenity/cli/native_diffusion.py
LINE: 525
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `_strip_condlabel_suffix`
DETAIL: Near-identical implementation appears in multiple files: serenity/cli/native_flux2.py:_strip_condlabel_suffix:641

FILE: serenity/cli/native_diffusion.py
LINE: 625
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `_load_image_tensor`
DETAIL: Near-identical implementation appears in multiple files: serenity/cli/native_flux2.py:_load_image_tensor:660

FILE: serenity/cli/native_flux2.py
LINE: 153
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `_as_bool`
DETAIL: Near-identical implementation appears in multiple files: serenity/sampling/sampler.py:_as_bool:190

FILE: serenity/cli/native_flux2.py
LINE: 486
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `_optional_float`
DETAIL: Near-identical implementation appears in multiple files: serenity/sampling/sampler.py:_optional_float:212

FILE: serenity/cli/native_flux2.py
LINE: 636
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `_is_condlabel_image`
DETAIL: Near-identical implementation appears in multiple files: serenity/cli/native_diffusion.py:_is_condlabel_image:520

FILE: serenity/cli/native_flux2.py
LINE: 641
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `_strip_condlabel_suffix`
DETAIL: Near-identical implementation appears in multiple files: serenity/cli/native_diffusion.py:_strip_condlabel_suffix:525

FILE: serenity/cli/native_flux2.py
LINE: 660
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `_load_image_tensor`
DETAIL: Near-identical implementation appears in multiple files: serenity/cli/native_diffusion.py:_load_image_tensor:625

FILE: serenity/core/enums.py
LINE: 21
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `offload`
DETAIL: Near-identical implementation appears in multiple files: serenity/training/flux2/base.py:offload:48

FILE: serenity/inference/models/base.py
LINE: 77
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `architecture`
DETAIL: Near-identical implementation appears in multiple files: serenity/inference/models/sd15.py:architecture:55

FILE: serenity/inference/models/base.py
LINE: 91
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `get_text_encoder_types`
DETAIL: Near-identical implementation appears in multiple files: serenity/inference/models/sd15.py:get_text_encoder_types:104

FILE: serenity/inference/models/base.py
LINE: 100
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `get_default_resolution`
DETAIL: Near-identical implementation appears in multiple files: serenity/inference/models/sd15.py:get_default_resolution:113

FILE: serenity/inference/models/chroma.py
LINE: 103
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `get_text_encoder_types`
DETAIL: Near-identical implementation appears in multiple files: serenity/inference/models/wan.py:get_text_encoder_types:101

FILE: serenity/inference/models/chroma.py
LINE: 112
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `get_default_resolution`
DETAIL: Near-identical implementation appears in multiple files: serenity/inference/models/flux.py:get_default_resolution:183, serenity/inference/models/lumina.py:get_default_resolution:77, serenity/inference/models/qwen.py:get_default_resolution:78, serenity/inference/models/sd3.py:get_default_resolution:125

FILE: serenity/inference/models/flux.py
LINE: 174
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `get_text_encoder_types`
DETAIL: Near-identical implementation appears in multiple files: serenity/tests/inference/test_model_loader.py:get_text_encoder_types:160

FILE: serenity/inference/models/flux.py
LINE: 183
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `get_default_resolution`
DETAIL: Near-identical implementation appears in multiple files: serenity/inference/models/chroma.py:get_default_resolution:112, serenity/inference/models/lumina.py:get_default_resolution:77, serenity/inference/models/qwen.py:get_default_resolution:78, serenity/inference/models/sd3.py:get_default_resolution:125

FILE: serenity/inference/models/lumina.py
LINE: 68
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `get_text_encoder_types`
DETAIL: Near-identical implementation appears in multiple files: serenity/inference/models/zimage.py:get_text_encoder_types:71

FILE: serenity/inference/models/lumina.py
LINE: 77
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `get_default_resolution`
DETAIL: Near-identical implementation appears in multiple files: serenity/inference/models/chroma.py:get_default_resolution:112, serenity/inference/models/flux.py:get_default_resolution:183, serenity/inference/models/qwen.py:get_default_resolution:78, serenity/inference/models/sd3.py:get_default_resolution:125

FILE: serenity/inference/models/qwen.py
LINE: 78
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `get_default_resolution`
DETAIL: Near-identical implementation appears in multiple files: serenity/inference/models/chroma.py:get_default_resolution:112, serenity/inference/models/flux.py:get_default_resolution:183, serenity/inference/models/lumina.py:get_default_resolution:77, serenity/inference/models/sd3.py:get_default_resolution:125

FILE: serenity/inference/models/sd15.py
LINE: 55
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `architecture`
DETAIL: Near-identical implementation appears in multiple files: serenity/inference/models/base.py:architecture:77

FILE: serenity/inference/models/sd15.py
LINE: 104
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `get_text_encoder_types`
DETAIL: Near-identical implementation appears in multiple files: serenity/inference/models/base.py:get_text_encoder_types:91

FILE: serenity/inference/models/sd15.py
LINE: 113
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `get_default_resolution`
DETAIL: Near-identical implementation appears in multiple files: serenity/inference/models/base.py:get_default_resolution:100

FILE: serenity/inference/models/sd3.py
LINE: 125
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `get_default_resolution`
DETAIL: Near-identical implementation appears in multiple files: serenity/inference/models/chroma.py:get_default_resolution:112, serenity/inference/models/flux.py:get_default_resolution:183, serenity/inference/models/lumina.py:get_default_resolution:77, serenity/inference/models/qwen.py:get_default_resolution:78

FILE: serenity/inference/models/sdxl.py
LINE: 173
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `get_default_resolution`
DETAIL: Near-identical implementation appears in multiple files: serenity/inference/models/chroma.py:get_default_resolution:112, serenity/inference/models/flux.py:get_default_resolution:183, serenity/inference/models/lumina.py:get_default_resolution:77, serenity/inference/models/qwen.py:get_default_resolution:78

FILE: serenity/inference/models/sdxl.py
LINE: 296
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `get_default_resolution`
DETAIL: Near-identical implementation appears in multiple files: serenity/inference/models/chroma.py:get_default_resolution:112, serenity/inference/models/flux.py:get_default_resolution:183, serenity/inference/models/lumina.py:get_default_resolution:77, serenity/inference/models/qwen.py:get_default_resolution:78

FILE: serenity/inference/models/wan.py
LINE: 101
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `get_text_encoder_types`
DETAIL: Near-identical implementation appears in multiple files: serenity/inference/models/chroma.py:get_text_encoder_types:103

FILE: serenity/inference/models/zimage.py
LINE: 71
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `get_text_encoder_types`
DETAIL: Near-identical implementation appears in multiple files: serenity/inference/models/lumina.py:get_text_encoder_types:68

FILE: serenity/inference/models/zimage.py
LINE: 80
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `get_default_resolution`
DETAIL: Near-identical implementation appears in multiple files: serenity/inference/models/chroma.py:get_default_resolution:112, serenity/inference/models/flux.py:get_default_resolution:183, serenity/inference/models/lumina.py:get_default_resolution:77, serenity/inference/models/qwen.py:get_default_resolution:78

FILE: serenity/inference/sampling/prediction.py
LINE: 95
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `calculate_denoised`
DETAIL: Near-identical implementation appears in multiple files: serenity/inference/sampling/prediction.py:calculate_denoised:169

FILE: serenity/inference/sampling/prediction.py
LINE: 142
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `sigma_to_timestep`
DETAIL: Near-identical implementation appears in multiple files: serenity/inference/sampling/prediction.py:sigma_to_timestep:316, serenity/inference/sampling/prediction.py:sigma_to_timestep:328

FILE: serenity/inference/sampling/prediction.py
LINE: 169
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `calculate_denoised`
DETAIL: Near-identical implementation appears in multiple files: serenity/inference/sampling/prediction.py:calculate_denoised:95

FILE: serenity/inference/sampling/prediction.py
LINE: 183
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `inverse_noise_scaling`
DETAIL: Near-identical implementation appears in multiple files: serenity/inference/sampling/prediction.py:inverse_noise_scaling:267

FILE: serenity/inference/sampling/prediction.py
LINE: 267
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `inverse_noise_scaling`
DETAIL: Near-identical implementation appears in multiple files: serenity/inference/sampling/prediction.py:inverse_noise_scaling:183

FILE: serenity/inference/sampling/prediction.py
LINE: 316
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `sigma_to_timestep`
DETAIL: Near-identical implementation appears in multiple files: serenity/inference/sampling/prediction.py:sigma_to_timestep:142, serenity/inference/sampling/prediction.py:sigma_to_timestep:328

FILE: serenity/inference/sampling/prediction.py
LINE: 328
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `sigma_to_timestep`
DETAIL: Near-identical implementation appears in multiple files: serenity/inference/sampling/prediction.py:sigma_to_timestep:142, serenity/inference/sampling/prediction.py:sigma_to_timestep:316

FILE: serenity/inference/text/clip.py
LINE: 151
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `unload`
DETAIL: Near-identical implementation appears in multiple files: serenity/inference/text/gemma.py:unload:69, serenity/inference/text/qwen_enc.py:unload:69, serenity/inference/text/t5.py:unload:69

FILE: serenity/inference/text/clip.py
LINE: 161
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `is_loaded`
DETAIL: Near-identical implementation appears in multiple files: serenity/inference/text/gemma.py:is_loaded:79, serenity/inference/text/qwen_enc.py:is_loaded:79, serenity/inference/text/t5.py:is_loaded:79

FILE: serenity/inference/text/clip.py
LINE: 279
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `_tokenize_bare`
DETAIL: Near-identical implementation appears in multiple files: serenity/inference/text/t5.py:_tokenize_bare:172

FILE: serenity/inference/text/gemma.py
LINE: 31
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `__init__`
DETAIL: Near-identical implementation appears in multiple files: serenity/inference/text/qwen_enc.py:__init__:31, serenity/inference/text/t5.py:__init__:31

FILE: serenity/inference/text/gemma.py
LINE: 69
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `unload`
DETAIL: Near-identical implementation appears in multiple files: serenity/inference/text/clip.py:unload:151, serenity/inference/text/qwen_enc.py:unload:69, serenity/inference/text/t5.py:unload:69

FILE: serenity/inference/text/gemma.py
LINE: 79
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `is_loaded`
DETAIL: Near-identical implementation appears in multiple files: serenity/inference/text/clip.py:is_loaded:161, serenity/inference/text/qwen_enc.py:is_loaded:79, serenity/inference/text/t5.py:is_loaded:79

FILE: serenity/inference/text/gemma.py
LINE: 172
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `_tokenize_bare`
DETAIL: Near-identical implementation appears in multiple files: serenity/inference/text/qwen_enc.py:_tokenize_bare:172

FILE: serenity/inference/text/qwen_enc.py
LINE: 31
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `__init__`
DETAIL: Near-identical implementation appears in multiple files: serenity/inference/text/gemma.py:__init__:31, serenity/inference/text/t5.py:__init__:31

FILE: serenity/inference/text/qwen_enc.py
LINE: 69
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `unload`
DETAIL: Near-identical implementation appears in multiple files: serenity/inference/text/clip.py:unload:151, serenity/inference/text/gemma.py:unload:69, serenity/inference/text/t5.py:unload:69

FILE: serenity/inference/text/qwen_enc.py
LINE: 79
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `is_loaded`
DETAIL: Near-identical implementation appears in multiple files: serenity/inference/text/clip.py:is_loaded:161, serenity/inference/text/gemma.py:is_loaded:79, serenity/inference/text/t5.py:is_loaded:79

FILE: serenity/inference/text/qwen_enc.py
LINE: 172
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `_tokenize_bare`
DETAIL: Near-identical implementation appears in multiple files: serenity/inference/text/gemma.py:_tokenize_bare:172

FILE: serenity/inference/text/t5.py
LINE: 31
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `__init__`
DETAIL: Near-identical implementation appears in multiple files: serenity/inference/text/gemma.py:__init__:31, serenity/inference/text/qwen_enc.py:__init__:31

FILE: serenity/inference/text/t5.py
LINE: 69
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `unload`
DETAIL: Near-identical implementation appears in multiple files: serenity/inference/text/clip.py:unload:151, serenity/inference/text/gemma.py:unload:69, serenity/inference/text/qwen_enc.py:unload:69

FILE: serenity/inference/text/t5.py
LINE: 79
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `is_loaded`
DETAIL: Near-identical implementation appears in multiple files: serenity/inference/text/clip.py:is_loaded:161, serenity/inference/text/gemma.py:is_loaded:79, serenity/inference/text/qwen_enc.py:is_loaded:79

FILE: serenity/inference/text/t5.py
LINE: 172
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `_tokenize_bare`
DETAIL: Near-identical implementation appears in multiple files: serenity/inference/text/clip.py:_tokenize_bare:279

FILE: serenity/inference/vae/decoder.py
LINE: 41
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `__init__`
DETAIL: Near-identical implementation appears in multiple files: serenity/inference/vae/encoder.py:__init__:33

FILE: serenity/inference/vae/decoder.py
LINE: 80
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `unload`
DETAIL: Near-identical implementation appears in multiple files: serenity/inference/vae/encoder.py:unload:69

FILE: serenity/inference/vae/decoder.py
LINE: 87
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `is_loaded`
DETAIL: Near-identical implementation appears in multiple files: serenity/inference/vae/encoder.py:is_loaded:76

FILE: serenity/inference/vae/encoder.py
LINE: 33
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `__init__`
DETAIL: Near-identical implementation appears in multiple files: serenity/inference/vae/decoder.py:__init__:41

FILE: serenity/inference/vae/encoder.py
LINE: 69
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `unload`
DETAIL: Near-identical implementation appears in multiple files: serenity/inference/vae/decoder.py:unload:80

FILE: serenity/inference/vae/encoder.py
LINE: 76
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `is_loaded`
DETAIL: Near-identical implementation appears in multiple files: serenity/inference/vae/decoder.py:is_loaded:87

FILE: serenity/models/chroma.py
LINE: 76
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `get_train_module`
DETAIL: Near-identical implementation appears in multiple files: serenity/models/flux1.py:get_train_module:290, serenity/models/flux2.py:get_train_module:76, serenity/models/hunyuan_video.py:get_train_module:96, serenity/models/ltx2.py:get_train_module:498

FILE: serenity/models/chroma.py
LINE: 79
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `encode_latents`
DETAIL: Near-identical implementation appears in multiple files: serenity/models/flux1.py:encode_latents:293, serenity/models/sd3.py:encode_latents:242, serenity/models/zimage.py:encode_latents:207

FILE: serenity/models/chroma.py
LINE: 131
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `cache_prompt_device`
DETAIL: Near-identical implementation appears in multiple files: serenity/models/flux1.py:cache_prompt_device:345, serenity/models/hunyuan_video.py:cache_prompt_device:181, serenity/models/sd3.py:cache_prompt_device:276, serenity/models/zimage.py:cache_prompt_device:254

FILE: serenity/models/chroma.py
LINE: 137
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `move_text_encoders_to_device`
DETAIL: Near-identical implementation appears in multiple files: serenity/models/zimage.py:move_text_encoders_to_device:261

FILE: serenity/models/chroma.py
LINE: 145
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `offload_text_encoders`
DETAIL: Near-identical implementation appears in multiple files: serenity/models/sd15.py:offload_text_encoders:86, serenity/models/zimage.py:offload_text_encoders:270

FILE: serenity/models/flux1.py
LINE: 290
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `get_train_module`
DETAIL: Near-identical implementation appears in multiple files: serenity/models/chroma.py:get_train_module:76, serenity/models/flux2.py:get_train_module:76, serenity/models/hunyuan_video.py:get_train_module:96, serenity/models/ltx2.py:get_train_module:498

FILE: serenity/models/flux1.py
LINE: 293
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `encode_latents`
DETAIL: Near-identical implementation appears in multiple files: serenity/models/chroma.py:encode_latents:79, serenity/models/sd3.py:encode_latents:242, serenity/models/zimage.py:encode_latents:207

FILE: serenity/models/flux1.py
LINE: 345
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `cache_prompt_device`
DETAIL: Near-identical implementation appears in multiple files: serenity/models/chroma.py:cache_prompt_device:131, serenity/models/hunyuan_video.py:cache_prompt_device:181, serenity/models/sd3.py:cache_prompt_device:276, serenity/models/zimage.py:cache_prompt_device:254

FILE: serenity/models/flux1.py
LINE: 352
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `move_text_encoders_to_device`
DETAIL: Near-identical implementation appears in multiple files: serenity/models/hunyuan_video.py:move_text_encoders_to_device:187

FILE: serenity/models/flux1.py
LINE: 362
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `offload_text_encoders`
DETAIL: Near-identical implementation appears in multiple files: serenity/models/hunyuan_video.py:offload_text_encoders:196, serenity/models/sdxl.py:offload_text_encoders:78

FILE: serenity/models/flux2.py
LINE: 76
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `get_train_module`
DETAIL: Near-identical implementation appears in multiple files: serenity/models/chroma.py:get_train_module:76, serenity/models/flux1.py:get_train_module:290, serenity/models/hunyuan_video.py:get_train_module:96, serenity/models/ltx2.py:get_train_module:498

FILE: serenity/models/flux2.py
LINE: 167
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `cache_prompt_device`
DETAIL: Near-identical implementation appears in multiple files: serenity/models/ltx2.py:cache_prompt_device:566, serenity/models/sd15.py:cache_prompt_device:76, serenity/models/sdxl.py:cache_prompt_device:67

FILE: serenity/models/flux2.py
LINE: 171
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `move_text_encoders_to_device`
DETAIL: Near-identical implementation appears in multiple files: serenity/models/ltx2.py:move_text_encoders_to_device:570

FILE: serenity/models/flux2.py
LINE: 177
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `offload_text_encoders`
DETAIL: Near-identical implementation appears in multiple files: serenity/models/ltx2.py:offload_text_encoders:576

FILE: serenity/models/hunyuan_video.py
LINE: 96
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `get_train_module`
DETAIL: Near-identical implementation appears in multiple files: serenity/models/chroma.py:get_train_module:76, serenity/models/flux1.py:get_train_module:290, serenity/models/flux2.py:get_train_module:76, serenity/models/ltx2.py:get_train_module:498

FILE: serenity/models/hunyuan_video.py
LINE: 181
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `cache_prompt_device`
DETAIL: Near-identical implementation appears in multiple files: serenity/models/chroma.py:cache_prompt_device:131, serenity/models/flux1.py:cache_prompt_device:345, serenity/models/sd3.py:cache_prompt_device:276, serenity/models/zimage.py:cache_prompt_device:254

FILE: serenity/models/hunyuan_video.py
LINE: 187
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `move_text_encoders_to_device`
DETAIL: Near-identical implementation appears in multiple files: serenity/models/flux1.py:move_text_encoders_to_device:352

FILE: serenity/models/hunyuan_video.py
LINE: 196
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `offload_text_encoders`
DETAIL: Near-identical implementation appears in multiple files: serenity/models/flux1.py:offload_text_encoders:362, serenity/models/sdxl.py:offload_text_encoders:78

FILE: serenity/models/ltx2.py
LINE: 337
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `_component_has_weights`
DETAIL: Near-identical implementation appears in multiple files: serenity/models/qwen.py:_component_has_weights:62, serenity/models/sd3.py:_component_has_weights:67, serenity/models/zimage.py:_component_has_weights:78

FILE: serenity/models/ltx2.py
LINE: 498
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `get_train_module`
DETAIL: Near-identical implementation appears in multiple files: serenity/models/chroma.py:get_train_module:76, serenity/models/flux1.py:get_train_module:290, serenity/models/flux2.py:get_train_module:76, serenity/models/hunyuan_video.py:get_train_module:96

FILE: serenity/models/ltx2.py
LINE: 566
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `cache_prompt_device`
DETAIL: Near-identical implementation appears in multiple files: serenity/models/flux2.py:cache_prompt_device:167, serenity/models/sd15.py:cache_prompt_device:76, serenity/models/sdxl.py:cache_prompt_device:67

FILE: serenity/models/ltx2.py
LINE: 570
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `move_text_encoders_to_device`
DETAIL: Near-identical implementation appears in multiple files: serenity/models/flux2.py:move_text_encoders_to_device:171

FILE: serenity/models/ltx2.py
LINE: 576
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `offload_text_encoders`
DETAIL: Near-identical implementation appears in multiple files: serenity/models/flux2.py:offload_text_encoders:177

FILE: serenity/models/qwen.py
LINE: 62
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `_component_has_weights`
DETAIL: Near-identical implementation appears in multiple files: serenity/models/ltx2.py:_component_has_weights:337, serenity/models/sd3.py:_component_has_weights:67, serenity/models/zimage.py:_component_has_weights:78

FILE: serenity/models/qwen.py
LINE: 399
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `get_train_module`
DETAIL: Near-identical implementation appears in multiple files: serenity/models/chroma.py:get_train_module:76, serenity/models/flux1.py:get_train_module:290, serenity/models/flux2.py:get_train_module:76, serenity/models/hunyuan_video.py:get_train_module:96

FILE: serenity/models/sd15.py
LINE: 52
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `get_train_module`
DETAIL: Near-identical implementation appears in multiple files: serenity/models/sdxl.py:get_train_module:43

FILE: serenity/models/sd15.py
LINE: 55
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `encode_latents`
DETAIL: Near-identical implementation appears in multiple files: serenity/models/sdxl.py:encode_latents:46

FILE: serenity/models/sd15.py
LINE: 76
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `cache_prompt_device`
DETAIL: Near-identical implementation appears in multiple files: serenity/models/flux2.py:cache_prompt_device:167, serenity/models/ltx2.py:cache_prompt_device:566, serenity/models/sdxl.py:cache_prompt_device:67

FILE: serenity/models/sd15.py
LINE: 86
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `offload_text_encoders`
DETAIL: Near-identical implementation appears in multiple files: serenity/models/chroma.py:offload_text_encoders:145, serenity/models/zimage.py:offload_text_encoders:270

FILE: serenity/models/sd3.py
LINE: 67
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `_component_has_weights`
DETAIL: Near-identical implementation appears in multiple files: serenity/models/ltx2.py:_component_has_weights:337, serenity/models/qwen.py:_component_has_weights:62, serenity/models/zimage.py:_component_has_weights:78

FILE: serenity/models/sd3.py
LINE: 239
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `get_train_module`
DETAIL: Near-identical implementation appears in multiple files: serenity/models/chroma.py:get_train_module:76, serenity/models/flux1.py:get_train_module:290, serenity/models/flux2.py:get_train_module:76, serenity/models/hunyuan_video.py:get_train_module:96

FILE: serenity/models/sd3.py
LINE: 242
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `encode_latents`
DETAIL: Near-identical implementation appears in multiple files: serenity/models/chroma.py:encode_latents:79, serenity/models/flux1.py:encode_latents:293, serenity/models/zimage.py:encode_latents:207

FILE: serenity/models/sd3.py
LINE: 276
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `cache_prompt_device`
DETAIL: Near-identical implementation appears in multiple files: serenity/models/chroma.py:cache_prompt_device:131, serenity/models/flux1.py:cache_prompt_device:345, serenity/models/hunyuan_video.py:cache_prompt_device:181, serenity/models/zimage.py:cache_prompt_device:254

FILE: serenity/models/sdxl.py
LINE: 43
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `get_train_module`
DETAIL: Near-identical implementation appears in multiple files: serenity/models/sd15.py:get_train_module:52

FILE: serenity/models/sdxl.py
LINE: 46
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `encode_latents`
DETAIL: Near-identical implementation appears in multiple files: serenity/models/sd15.py:encode_latents:55

FILE: serenity/models/sdxl.py
LINE: 67
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `cache_prompt_device`
DETAIL: Near-identical implementation appears in multiple files: serenity/models/flux2.py:cache_prompt_device:167, serenity/models/ltx2.py:cache_prompt_device:566, serenity/models/sd15.py:cache_prompt_device:76

FILE: serenity/models/sdxl.py
LINE: 78
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `offload_text_encoders`
DETAIL: Near-identical implementation appears in multiple files: serenity/models/flux1.py:offload_text_encoders:362, serenity/models/hunyuan_video.py:offload_text_encoders:196

FILE: serenity/models/setup.py
LINE: 400
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `create_parameters`
DETAIL: Near-identical implementation appears in multiple files: serenity/models/setup.py:create_parameters:433

FILE: serenity/models/setup.py
LINE: 433
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `create_parameters`
DETAIL: Near-identical implementation appears in multiple files: serenity/models/setup.py:create_parameters:400

FILE: serenity/models/zimage.py
LINE: 78
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `_component_has_weights`
DETAIL: Near-identical implementation appears in multiple files: serenity/models/ltx2.py:_component_has_weights:337, serenity/models/qwen.py:_component_has_weights:62, serenity/models/sd3.py:_component_has_weights:67

FILE: serenity/models/zimage.py
LINE: 204
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `get_train_module`
DETAIL: Near-identical implementation appears in multiple files: serenity/models/chroma.py:get_train_module:76, serenity/models/flux1.py:get_train_module:290, serenity/models/flux2.py:get_train_module:76, serenity/models/hunyuan_video.py:get_train_module:96

FILE: serenity/models/zimage.py
LINE: 207
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `encode_latents`
DETAIL: Near-identical implementation appears in multiple files: serenity/models/chroma.py:encode_latents:79, serenity/models/flux1.py:encode_latents:293, serenity/models/sd3.py:encode_latents:242

FILE: serenity/models/zimage.py
LINE: 254
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `cache_prompt_device`
DETAIL: Near-identical implementation appears in multiple files: serenity/models/chroma.py:cache_prompt_device:131, serenity/models/flux1.py:cache_prompt_device:345, serenity/models/hunyuan_video.py:cache_prompt_device:181, serenity/models/sd3.py:cache_prompt_device:276

FILE: serenity/models/zimage.py
LINE: 261
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `move_text_encoders_to_device`
DETAIL: Near-identical implementation appears in multiple files: serenity/models/chroma.py:move_text_encoders_to_device:137

FILE: serenity/models/zimage.py
LINE: 270
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `offload_text_encoders`
DETAIL: Near-identical implementation appears in multiple files: serenity/models/chroma.py:offload_text_encoders:145, serenity/models/sd15.py:offload_text_encoders:86

FILE: serenity/sampling/sampler.py
LINE: 190
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `_as_bool`
DETAIL: Near-identical implementation appears in multiple files: serenity/cli/native_flux2.py:_as_bool:153

FILE: serenity/sampling/sampler.py
LINE: 212
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `_optional_float`
DETAIL: Near-identical implementation appears in multiple files: serenity/cli/native_flux2.py:_optional_float:486

FILE: serenity/sampling/sampler.py
LINE: 930
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `__init__`
DETAIL: Near-identical implementation appears in multiple files: serenity/sampling/sampler.py:__init__:987

FILE: serenity/sampling/sampler.py
LINE: 987
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `__init__`
DETAIL: Near-identical implementation appears in multiple files: serenity/sampling/sampler.py:__init__:930

FILE: serenity/tests/inference/test_model_adapters_flux.py
LINE: 39
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `test_prediction_type`
DETAIL: Near-identical implementation appears in multiple files: serenity/tests/inference/test_model_adapters_flux.py:test_prediction_type:132, serenity/tests/inference/test_model_adapters_flux.py:test_prediction_type:217

FILE: serenity/tests/inference/test_model_adapters_flux.py
LINE: 42
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `test_vae_scaling_factor`
DETAIL: Near-identical implementation appears in multiple files: serenity/tests/inference/test_model_adapters_flux.py:test_vae_scaling_factor:135, serenity/tests/inference/test_model_adapters_flux.py:test_vae_scaling_factor:220

FILE: serenity/tests/inference/test_model_adapters_flux.py
LINE: 45
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `test_default_resolution`
DETAIL: Near-identical implementation appears in multiple files: serenity/tests/inference/test_model_adapters_flux.py:test_default_resolution:138, serenity/tests/inference/test_model_adapters_flux.py:test_default_resolution:223, serenity/tests/inference/test_model_adapters_sd.py:test_default_resolution:94, serenity/tests/inference/test_model_adapters_sd.py:test_default_resolution:190

FILE: serenity/tests/inference/test_model_adapters_flux.py
LINE: 48
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `test_is_base_model_adapter_subclass`
DETAIL: Near-identical implementation appears in multiple files: serenity/tests/inference/test_model_adapters_flux.py:test_is_base_model_adapter_subclass:226

FILE: serenity/tests/inference/test_model_adapters_flux.py
LINE: 132
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `test_prediction_type`
DETAIL: Near-identical implementation appears in multiple files: serenity/tests/inference/test_model_adapters_flux.py:test_prediction_type:39, serenity/tests/inference/test_model_adapters_flux.py:test_prediction_type:217

FILE: serenity/tests/inference/test_model_adapters_flux.py
LINE: 135
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `test_vae_scaling_factor`
DETAIL: Near-identical implementation appears in multiple files: serenity/tests/inference/test_model_adapters_flux.py:test_vae_scaling_factor:42, serenity/tests/inference/test_model_adapters_flux.py:test_vae_scaling_factor:220

FILE: serenity/tests/inference/test_model_adapters_flux.py
LINE: 138
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `test_default_resolution`
DETAIL: Near-identical implementation appears in multiple files: serenity/tests/inference/test_model_adapters_flux.py:test_default_resolution:45, serenity/tests/inference/test_model_adapters_flux.py:test_default_resolution:223, serenity/tests/inference/test_model_adapters_sd.py:test_default_resolution:94, serenity/tests/inference/test_model_adapters_sd.py:test_default_resolution:190

FILE: serenity/tests/inference/test_model_adapters_flux.py
LINE: 217
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `test_prediction_type`
DETAIL: Near-identical implementation appears in multiple files: serenity/tests/inference/test_model_adapters_flux.py:test_prediction_type:39, serenity/tests/inference/test_model_adapters_flux.py:test_prediction_type:132

FILE: serenity/tests/inference/test_model_adapters_flux.py
LINE: 220
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `test_vae_scaling_factor`
DETAIL: Near-identical implementation appears in multiple files: serenity/tests/inference/test_model_adapters_flux.py:test_vae_scaling_factor:42, serenity/tests/inference/test_model_adapters_flux.py:test_vae_scaling_factor:135

FILE: serenity/tests/inference/test_model_adapters_flux.py
LINE: 223
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `test_default_resolution`
DETAIL: Near-identical implementation appears in multiple files: serenity/tests/inference/test_model_adapters_flux.py:test_default_resolution:45, serenity/tests/inference/test_model_adapters_flux.py:test_default_resolution:138, serenity/tests/inference/test_model_adapters_sd.py:test_default_resolution:94, serenity/tests/inference/test_model_adapters_sd.py:test_default_resolution:190

FILE: serenity/tests/inference/test_model_adapters_flux.py
LINE: 226
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `test_is_base_model_adapter_subclass`
DETAIL: Near-identical implementation appears in multiple files: serenity/tests/inference/test_model_adapters_flux.py:test_is_base_model_adapter_subclass:48

FILE: serenity/tests/inference/test_model_adapters_sd.py
LINE: 35
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `test_prediction_type`
DETAIL: Near-identical implementation appears in multiple files: serenity/tests/inference/test_model_adapters_sd.py:test_prediction_type:88, serenity/tests/inference/test_model_adapters_sd.py:test_prediction_type:184

FILE: serenity/tests/inference/test_model_adapters_sd.py
LINE: 88
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `test_prediction_type`
DETAIL: Near-identical implementation appears in multiple files: serenity/tests/inference/test_model_adapters_sd.py:test_prediction_type:35, serenity/tests/inference/test_model_adapters_sd.py:test_prediction_type:184

FILE: serenity/tests/inference/test_model_adapters_sd.py
LINE: 91
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `test_vae_scaling_factor`
DETAIL: Near-identical implementation appears in multiple files: serenity/tests/inference/test_model_adapters_sd.py:test_vae_scaling_factor:187

FILE: serenity/tests/inference/test_model_adapters_sd.py
LINE: 94
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `test_default_resolution`
DETAIL: Near-identical implementation appears in multiple files: serenity/tests/inference/test_model_adapters_flux.py:test_default_resolution:45, serenity/tests/inference/test_model_adapters_flux.py:test_default_resolution:138, serenity/tests/inference/test_model_adapters_flux.py:test_default_resolution:223, serenity/tests/inference/test_model_adapters_sd.py:test_default_resolution:190

FILE: serenity/tests/inference/test_model_adapters_sd.py
LINE: 184
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `test_prediction_type`
DETAIL: Near-identical implementation appears in multiple files: serenity/tests/inference/test_model_adapters_sd.py:test_prediction_type:35, serenity/tests/inference/test_model_adapters_sd.py:test_prediction_type:88

FILE: serenity/tests/inference/test_model_adapters_sd.py
LINE: 187
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `test_vae_scaling_factor`
DETAIL: Near-identical implementation appears in multiple files: serenity/tests/inference/test_model_adapters_sd.py:test_vae_scaling_factor:91

FILE: serenity/tests/inference/test_model_adapters_sd.py
LINE: 190
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `test_default_resolution`
DETAIL: Near-identical implementation appears in multiple files: serenity/tests/inference/test_model_adapters_flux.py:test_default_resolution:45, serenity/tests/inference/test_model_adapters_flux.py:test_default_resolution:138, serenity/tests/inference/test_model_adapters_flux.py:test_default_resolution:223, serenity/tests/inference/test_model_adapters_sd.py:test_default_resolution:94

FILE: serenity/tests/inference/test_model_adapters_sd.py
LINE: 252
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `test_default_resolution`
DETAIL: Near-identical implementation appears in multiple files: serenity/tests/inference/test_model_adapters_flux.py:test_default_resolution:45, serenity/tests/inference/test_model_adapters_flux.py:test_default_resolution:138, serenity/tests/inference/test_model_adapters_flux.py:test_default_resolution:223, serenity/tests/inference/test_model_adapters_sd.py:test_default_resolution:94

FILE: serenity/tests/inference/test_model_loader.py
LINE: 160
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `get_text_encoder_types`
DETAIL: Near-identical implementation appears in multiple files: serenity/inference/models/flux.py:get_text_encoder_types:174

FILE: serenity/tests/inference/test_model_loader.py
LINE: 169
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `get_default_resolution`
DETAIL: Near-identical implementation appears in multiple files: serenity/inference/models/chroma.py:get_default_resolution:112, serenity/inference/models/flux.py:get_default_resolution:183, serenity/inference/models/lumina.py:get_default_resolution:77, serenity/inference/models/qwen.py:get_default_resolution:78

FILE: serenity/tests/inference/test_sampling_pipeline.py
LINE: 541
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `callback`
DETAIL: Near-identical implementation appears in multiple files: serenity/tests/inference/test_sampling_pipeline.py:callback:636, serenity/tests/inference/test_sampling_pipeline.py:callback:726

FILE: serenity/tests/inference/test_sampling_pipeline.py
LINE: 552
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `model_fn`
DETAIL: Near-identical implementation appears in multiple files: serenity/tests/inference/test_sampling_pipeline.py:model_fn:565, serenity/tests/inference/test_sampling_pipeline.py:model_fn:659, serenity/tests/inference/test_sampling_pipeline.py:model_fn:756, serenity/tests/inference/test_sampling_pipeline.py:model_fn:766

FILE: serenity/tests/inference/test_sampling_pipeline.py
LINE: 565
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `model_fn`
DETAIL: Near-identical implementation appears in multiple files: serenity/tests/inference/test_sampling_pipeline.py:model_fn:552, serenity/tests/inference/test_sampling_pipeline.py:model_fn:659, serenity/tests/inference/test_sampling_pipeline.py:model_fn:756, serenity/tests/inference/test_sampling_pipeline.py:model_fn:766

FILE: serenity/tests/inference/test_sampling_pipeline.py
LINE: 636
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `callback`
DETAIL: Near-identical implementation appears in multiple files: serenity/tests/inference/test_sampling_pipeline.py:callback:541, serenity/tests/inference/test_sampling_pipeline.py:callback:726

FILE: serenity/tests/inference/test_sampling_pipeline.py
LINE: 644
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `model_fn`
DETAIL: Near-identical implementation appears in multiple files: serenity/tests/inference/test_sampling_pipeline.py:model_fn:706

FILE: serenity/tests/inference/test_sampling_pipeline.py
LINE: 659
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `model_fn`
DETAIL: Near-identical implementation appears in multiple files: serenity/tests/inference/test_sampling_pipeline.py:model_fn:552, serenity/tests/inference/test_sampling_pipeline.py:model_fn:565, serenity/tests/inference/test_sampling_pipeline.py:model_fn:756, serenity/tests/inference/test_sampling_pipeline.py:model_fn:766

FILE: serenity/tests/inference/test_sampling_pipeline.py
LINE: 706
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `model_fn`
DETAIL: Near-identical implementation appears in multiple files: serenity/tests/inference/test_sampling_pipeline.py:model_fn:644

FILE: serenity/tests/inference/test_sampling_pipeline.py
LINE: 726
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `callback`
DETAIL: Near-identical implementation appears in multiple files: serenity/tests/inference/test_sampling_pipeline.py:callback:541, serenity/tests/inference/test_sampling_pipeline.py:callback:636

FILE: serenity/tests/inference/test_sampling_pipeline.py
LINE: 756
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `model_fn`
DETAIL: Near-identical implementation appears in multiple files: serenity/tests/inference/test_sampling_pipeline.py:model_fn:552, serenity/tests/inference/test_sampling_pipeline.py:model_fn:565, serenity/tests/inference/test_sampling_pipeline.py:model_fn:659, serenity/tests/inference/test_sampling_pipeline.py:model_fn:766

FILE: serenity/tests/inference/test_sampling_pipeline.py
LINE: 766
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `model_fn`
DETAIL: Near-identical implementation appears in multiple files: serenity/tests/inference/test_sampling_pipeline.py:model_fn:552, serenity/tests/inference/test_sampling_pipeline.py:model_fn:565, serenity/tests/inference/test_sampling_pipeline.py:model_fn:659, serenity/tests/inference/test_sampling_pipeline.py:model_fn:756

FILE: serenity/tests/inference/test_sampling_pipeline.py
LINE: 776
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `model_fn`
DETAIL: Near-identical implementation appears in multiple files: serenity/tests/inference/test_sampling_pipeline.py:model_fn:552, serenity/tests/inference/test_sampling_pipeline.py:model_fn:565, serenity/tests/inference/test_sampling_pipeline.py:model_fn:659, serenity/tests/inference/test_sampling_pipeline.py:model_fn:756

FILE: serenity/tests/test_sampling_samplers.py
LINE: 160
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `_load_class`
DETAIL: Near-identical implementation appears in multiple files: serenity/tests/test_sampling_samplers.py:_load_class:186

FILE: serenity/tests/test_sampling_samplers.py
LINE: 186
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `_load_class`
DETAIL: Near-identical implementation appears in multiple files: serenity/tests/test_sampling_samplers.py:_load_class:160

FILE: serenity/training/flux2/base.py
LINE: 48
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `offload`
DETAIL: Near-identical implementation appears in multiple files: serenity/core/enums.py:offload:21

FILE: serenity/training/lora_manager.py
LINE: 47
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `_normalize_model_type`
DETAIL: Near-identical implementation appears in multiple files: serenity/training/lycoris_manager.py:_normalize_model_type:121

FILE: serenity/training/lora_manager.py
LINE: 52
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `_coerce_dtype`
DETAIL: Near-identical implementation appears in multiple files: serenity/training/lycoris_manager.py:_coerce_dtype:126

FILE: serenity/training/lycoris_manager.py
LINE: 121
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `_normalize_model_type`
DETAIL: Near-identical implementation appears in multiple files: serenity/training/lora_manager.py:_normalize_model_type:47

FILE: serenity/training/lycoris_manager.py
LINE: 126
CATEGORY: [4]
SEVERITY: MEDIUM
ISSUE: Duplicated function body `_coerce_dtype`
DETAIL: Near-identical implementation appears in multiple files: serenity/training/lora_manager.py:_coerce_dtype:52

FILE: serenity/cli/commands.py
LINE: 1
CATEGORY: [5]
SEVERITY: MEDIUM
ISSUE: Oversized source file
DETAIL: 1029 lines; likely too many responsibilities in one module.

FILE: serenity/inference/engine.py
LINE: 1
CATEGORY: [5]
SEVERITY: MEDIUM
ISSUE: Oversized source file
DETAIL: 992 lines; likely too many responsibilities in one module.

FILE: serenity/tests/inference/test_sampling_pipeline.py
LINE: 1
CATEGORY: [5]
SEVERITY: MEDIUM
ISSUE: Oversized source file
DETAIL: 937 lines; likely too many responsibilities in one module.

FILE: serenity/tests/inference/test_text_encoding.py
LINE: 1
CATEGORY: [5]
SEVERITY: MEDIUM
ISSUE: Oversized source file
DETAIL: 1071 lines; likely too many responsibilities in one module.

FILE: serenity/training/flux2/base.py
LINE: 1
CATEGORY: [5]
SEVERITY: MEDIUM
ISSUE: Oversized source file
DETAIL: 768 lines; likely too many responsibilities in one module.

FILE: serenity/training/optimizers.py
LINE: 1
CATEGORY: [5]
SEVERITY: MEDIUM
ISSUE: Oversized source file
DETAIL: 820 lines; likely too many responsibilities in one module.

FILE: serenity/ui/tabs/inference.py
LINE: 1
CATEGORY: [5]
SEVERITY: MEDIUM
ISSUE: Oversized source file
DETAIL: 783 lines; likely too many responsibilities in one module.

FILE: serenity/checkpoint/conversion.py
LINE: 88
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/checkpoint/resume.py
LINE: 208
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/checkpoint/resume.py
LINE: 220
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/checkpoint/resume.py
LINE: 236
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/checkpoint/resume.py
LINE: 246
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/cli/commands.py
LINE: 205
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: suppress(Exception) hides runtime errors
DETAIL: Broad suppression can silently mask incorrect behavior.

FILE: serenity/cli/native_diffusion.py
LINE: 236
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: suppress(Exception) hides runtime errors
DETAIL: Broad suppression can silently mask incorrect behavior.

FILE: serenity/cli/native_diffusion.py
LINE: 259
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: suppress(Exception) hides runtime errors
DETAIL: Broad suppression can silently mask incorrect behavior.

FILE: serenity/cli/native_diffusion.py
LINE: 268
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: suppress(Exception) hides runtime errors
DETAIL: Broad suppression can silently mask incorrect behavior.

FILE: serenity/cli/native_diffusion.py
LINE: 1379
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/cli/native_diffusion.py
LINE: 1404
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: suppress(Exception) hides runtime errors
DETAIL: Broad suppression can silently mask incorrect behavior.

FILE: serenity/cli/native_diffusion.py
LINE: 1454
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: suppress(Exception) hides runtime errors
DETAIL: Broad suppression can silently mask incorrect behavior.

FILE: serenity/cli/native_diffusion.py
LINE: 1471
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: suppress(Exception) hides runtime errors
DETAIL: Broad suppression can silently mask incorrect behavior.

FILE: serenity/cli/native_diffusion.py
LINE: 1476
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: suppress(Exception) hides runtime errors
DETAIL: Broad suppression can silently mask incorrect behavior.

FILE: serenity/cli/native_diffusion.py
LINE: 1481
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: suppress(Exception) hides runtime errors
DETAIL: Broad suppression can silently mask incorrect behavior.

FILE: serenity/cli/native_diffusion.py
LINE: 1485
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: suppress(Exception) hides runtime errors
DETAIL: Broad suppression can silently mask incorrect behavior.

FILE: serenity/cli/native_diffusion.py
LINE: 1489
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: suppress(Exception) hides runtime errors
DETAIL: Broad suppression can silently mask incorrect behavior.

FILE: serenity/cli/native_diffusion.py
LINE: 1536
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/cli/native_diffusion.py
LINE: 1668
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: suppress(Exception) hides runtime errors
DETAIL: Broad suppression can silently mask incorrect behavior.

FILE: serenity/cli/native_diffusion.py
LINE: 1691
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/cli/native_diffusion.py
LINE: 1795
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: suppress(Exception) hides runtime errors
DETAIL: Broad suppression can silently mask incorrect behavior.

FILE: serenity/cli/native_diffusion.py
LINE: 1812
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/cli/native_diffusion.py
LINE: 2206
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: suppress(Exception) hides runtime errors
DETAIL: Broad suppression can silently mask incorrect behavior.

FILE: serenity/cli/native_diffusion.py
LINE: 2228
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: suppress(Exception) hides runtime errors
DETAIL: Broad suppression can silently mask incorrect behavior.

FILE: serenity/cli/native_diffusion.py
LINE: 2467
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: suppress(Exception) hides runtime errors
DETAIL: Broad suppression can silently mask incorrect behavior.

FILE: serenity/cli/native_flux2.py
LINE: 323
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/cli/native_flux2.py
LINE: 801
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/cli/native_flux2.py
LINE: 894
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/core/trainer.py
LINE: 214
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/core/trainer.py
LINE: 220
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/core/trainer.py
LINE: 268
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/inference/attention/flash.py
LINE: 34
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/inference/attention/flash.py
LINE: 74
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/inference/attention/flash.py
LINE: 142
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/inference/attention/sage.py
LINE: 55
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/inference/attention/sage.py
LINE: 201
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/inference/attention/xformers_attn.py
LINE: 34
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/inference/attention/xformers_attn.py
LINE: 113
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/inference/engine.py
LINE: 606
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/inference/memory/streams.py
LINE: 294
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/inference/memory/streams.py
LINE: 369
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/inference/memory/vram.py
LINE: 233
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/inference/models/loader.py
LINE: 247
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/inference/quantization/ops.py
LINE: 242
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/models/chroma.py
LINE: 149
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: suppress(Exception) hides runtime errors
DETAIL: Broad suppression can silently mask incorrect behavior.

FILE: serenity/models/flux1.py
LINE: 49
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/models/flux1.py
LINE: 241
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/models/flux1.py
LINE: 252
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/models/flux1.py
LINE: 367
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: suppress(Exception) hides runtime errors
DETAIL: Broad suppression can silently mask incorrect behavior.

FILE: serenity/models/flux2.py
LINE: 44
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/models/flux2.py
LINE: 181
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: suppress(Exception) hides runtime errors
DETAIL: Broad suppression can silently mask incorrect behavior.

FILE: serenity/models/flux2_klein.py
LINE: 1101
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/models/flux2_klein.py
LINE: 1157
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/models/flux2_klein.py
LINE: 1179
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/models/flux2_klein.py
LINE: 1190
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/models/flux2_klein.py
LINE: 1205
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/models/flux2_klein.py
LINE: 1217
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/models/flux2_transformer.py
LINE: 11
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/models/flux2_transformer.py
LINE: 14
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/models/hunyuan_video.py
LINE: 201
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: suppress(Exception) hides runtime errors
DETAIL: Broad suppression can silently mask incorrect behavior.

FILE: serenity/models/ltx2.py
LINE: 73
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/models/ltx2.py
LINE: 359
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/models/ltx2.py
LINE: 458
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: suppress(Exception) hides runtime errors
DETAIL: Broad suppression can silently mask incorrect behavior.

FILE: serenity/models/ltx2.py
LINE: 580
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: suppress(Exception) hides runtime errors
DETAIL: Broad suppression can silently mask incorrect behavior.

FILE: serenity/models/qwen.py
LINE: 86
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/models/qwen.py
LINE: 224
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/models/qwen.py
LINE: 245
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/models/qwen.py
LINE: 460
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: suppress(Exception) hides runtime errors
DETAIL: Broad suppression can silently mask incorrect behavior.

FILE: serenity/models/sd15.py
LINE: 90
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: suppress(Exception) hides runtime errors
DETAIL: Broad suppression can silently mask incorrect behavior.

FILE: serenity/models/sd3.py
LINE: 75
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: suppress(Exception) hides runtime errors
DETAIL: Broad suppression can silently mask incorrect behavior.

FILE: serenity/models/sd3.py
LINE: 106
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/models/sd3.py
LINE: 298
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: suppress(Exception) hides runtime errors
DETAIL: Broad suppression can silently mask incorrect behavior.

FILE: serenity/models/sdxl.py
LINE: 83
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: suppress(Exception) hides runtime errors
DETAIL: Broad suppression can silently mask incorrect behavior.

FILE: serenity/models/zimage.py
LINE: 274
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: suppress(Exception) hides runtime errors
DETAIL: Broad suppression can silently mask incorrect behavior.

FILE: serenity/pipeline/cache.py
LINE: 91
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/pipeline/cache.py
LINE: 147
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/sampling/sampler.py
LINE: 112
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/sampling/sampler.py
LINE: 358
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: suppress(Exception) hides runtime errors
DETAIL: Broad suppression can silently mask incorrect behavior.

FILE: serenity/sampling/sampler.py
LINE: 411
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: suppress(Exception) hides runtime errors
DETAIL: Broad suppression can silently mask incorrect behavior.

FILE: serenity/sampling/sampler.py
LINE: 434
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/sampling/sampler.py
LINE: 449
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/sampling/sampler.py
LINE: 470
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/sampling/sampler.py
LINE: 496
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: suppress(Exception) hides runtime errors
DETAIL: Broad suppression can silently mask incorrect behavior.

FILE: serenity/sampling/sampler.py
LINE: 499
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/sampling/sampler.py
LINE: 576
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/sampling/sampler.py
LINE: 614
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/sampling/sampler.py
LINE: 644
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/sampling/sampler.py
LINE: 709
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: suppress(Exception) hides runtime errors
DETAIL: Broad suppression can silently mask incorrect behavior.

FILE: serenity/sampling/sampler.py
LINE: 975
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/sampling/sampler.py
LINE: 1020
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/tests/test_flux2_klein_forward.py
LINE: 135
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/tests/test_flux2_klein_forward.py
LINE: 252
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/tests/test_flux2_klein_forward.py
LINE: 361
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/tests/test_flux2_klein_forward.py
LINE: 384
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/tests/test_flux2_klein_forward.py
LINE: 391
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/tests/test_flux2_klein_forward.py
LINE: 397
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/tests/test_flux2_klein_integration.py
LINE: 98
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/tests/test_flux2_klein_integration.py
LINE: 135
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/tests/test_flux2_klein_integration.py
LINE: 179
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/tests/test_flux2_klein_integration.py
LINE: 184
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/tests/test_memory_contracts.py
LINE: 381
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/tests/test_memory_contracts.py
LINE: 390
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/tests/test_memory_contracts.py
LINE: 399
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/tests/test_memory_contracts.py
LINE: 407
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/tests/test_memory_contracts.py
LINE: 415
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/tests/test_memory_contracts.py
LINE: 425
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/tests/test_pipeline_integration.py
LINE: 588
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/tests/test_pipeline_integration.py
LINE: 597
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/tests/test_pipeline_integration.py
LINE: 605
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/tests/test_pipeline_integration.py
LINE: 613
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/tests/test_pipeline_integration.py
LINE: 621
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/tests/test_pipeline_integration.py
LINE: 632
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/tests/test_pipeline_integration.py
LINE: 640
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/tests/test_pipeline_integration.py
LINE: 649
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/tests/test_pipeline_integration.py
LINE: 658
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/tests/test_pipeline_integration.py
LINE: 666
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/training/embedding.py
LINE: 93
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/training/lora_manager.py
LINE: 163
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/training/lycoris_manager.py
LINE: 19
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/ui/app.py
LINE: 319
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/ui/app.py
LINE: 357
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/ui/state.py
LINE: 76
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/ui/state.py
LINE: 88
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/ui/tabs/inference.py
LINE: 448
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/ui/tabs/inference.py
LINE: 699
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/ui/tabs/inference.py
LINE: 763
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/ui/web/api.py
LINE: 258
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/ui/web/server.py
LINE: 83
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/ui/web/server.py
LINE: 111
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/ui/web/websocket.py
LINE: 44
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (Exception)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/utils/paths.py
LINE: 173
CATEGORY: [6]
SEVERITY: MEDIUM
ISSUE: Broad exception catch (BaseException)
DETAIL: Broad exception handling may hide specific failure causes.

FILE: serenity/core/trainer.py
LINE: 485-500
CATEGORY: [7]
SEVERITY: MEDIUM
ISSUE: Training loop does not clear gradients before first micro-step
DETAIL: No optimizer.zero_grad() is issued at loop start. If caller enters with stale grads, the first accumulation boundary includes prior-step gradients. Explicitly clear gradients before entering the epoch loop.

FILE: serenity/training/flux2/base.py
LINE: 431-444
CATEGORY: [7]
SEVERITY: MEDIUM
ISSUE: Resolution-dependent shift is silently overwritten for non-uniform densities
DETAIL: sample_timesteps() computes shifted timesteps, then fully replaces t for logit_normal/sigmoid branches. Shift no longer applies in these modes. Combine density shaping with shifted distribution instead of overriding.

FILE: serenity/training/flux2/edit_trainer.py
LINE: 483-485
CATEGORY: [7]
SEVERITY: MEDIUM
ISSUE: Edit trainer omits latent size when sampling timesteps
DETAIL: sample_timesteps(batch_size) is called without latent_height/latent_width, so resolution-dependent timestep shift always uses defaults instead of current batch resolution.

FILE: serenity/cli/native_diffusion.py
LINE: 625-627
CATEGORY: [8]
SEVERITY: MEDIUM
ISSUE: Aspect-ratio bucketing option is not honored in native image loading path
DETAIL: Images are always resized to square (resolution x resolution). This bypasses real bucketed AR training behavior despite config exposing aspect_ratio_bucketing.

FILE: serenity/cli/native_diffusion.py
LINE: 625-630
CATEGORY: [8]
SEVERITY: MEDIUM
ISSUE: Multi-resolution training behaves as fixed single-resolution resize
DETAIL: Data path uses one scalar resolution and hard resize. No active per-sample multi-resolution schedule/bucket assignment is used in native diffusion loop.

FILE: serenity/cli/native_diffusion.py
LINE: 1118-1355
CATEGORY: [8]
SEVERITY: MEDIUM
ISSUE: Min-SNR and other configured loss weighting are not wired into training loss
DETAIL: Native diffusion loss path returns plain F.mse_loss across families and never consumes configured loss_weight_fn/min-SNR weighting from training.losses helpers.

FILE: serenity/cli/native_diffusion.py
LINE: 2331-2335
CATEGORY: [8]
SEVERITY: MEDIUM
ISSUE: EMA disabled for adapter-mode training
DETAIL: When adapter training is active, code logs warning and skips EMA construction. That leaves adapter runs without EMA despite config support.

FILE: serenity/cli/native_diffusion.py
LINE: 1445
CATEGORY: [10]
SEVERITY: MEDIUM
ISSUE: torch.load without weights_only=True
DETAIL: Deserializing pickled objects can execute arbitrary code on untrusted files.

FILE: serenity/.eri-rpg/context/checkpointing.md
LINE: 338
CATEGORY: [3]
SEVERITY: LOW
ISSUE: Unfinished marker in code
DETAIL: TODO/FIXME/HACK marker indicates unfinished behavior.

FILE: serenity/.eri-rpg/context/work_add_gradient_clipping_to_training_loop.md
LINE: 2560
CATEGORY: [3]
SEVERITY: LOW
ISSUE: Large commented-out code block
DETAIL: Found >=6 contiguous commented lines; likely dead/abandoned code.

FILE: serenity/__main__.py
LINE: 1
CATEGORY: [3]
SEVERITY: LOW
ISSUE: Near-empty source file
DETAIL: Only 7 non-comment/non-empty lines.

FILE: serenity/adapters/base.py
LINE: 1
CATEGORY: [3]
SEVERITY: LOW
ISSUE: Near-empty source file
DETAIL: Only 7 non-comment/non-empty lines.

FILE: serenity/cli/__init__.py
LINE: 1
CATEGORY: [3]
SEVERITY: LOW
ISSUE: Near-empty source file
DETAIL: Only 3 non-comment/non-empty lines.

FILE: serenity/inference/cache/__init__.py
LINE: 1
CATEGORY: [3]
SEVERITY: LOW
ISSUE: Near-empty source file
DETAIL: Only 7 non-comment/non-empty lines.

FILE: serenity/inference/models/chroma.py
LINE: 20
CATEGORY: [3]
SEVERITY: LOW
ISSUE: Large commented-out code block
DETAIL: Found >=6 contiguous commented lines; likely dead/abandoned code.

FILE: serenity/inference/models/convert.py
LINE: 339
CATEGORY: [3]
SEVERITY: LOW
ISSUE: Large commented-out code block
DETAIL: Found >=6 contiguous commented lines; likely dead/abandoned code.

FILE: serenity/inference/models/detection.py
LINE: 352
CATEGORY: [3]
SEVERITY: LOW
ISSUE: Large commented-out code block
DETAIL: Found >=6 contiguous commented lines; likely dead/abandoned code.

FILE: serenity/inference/vae/__init__.py
LINE: 1
CATEGORY: [3]
SEVERITY: LOW
ISSUE: Near-empty source file
DETAIL: Only 8 non-comment/non-empty lines.

FILE: serenity/models/flux2_klein.py
LINE: 1416
CATEGORY: [3]
SEVERITY: LOW
ISSUE: Large commented-out code block
DETAIL: Found >=6 contiguous commented lines; likely dead/abandoned code.

FILE: serenity/models/flux_klein.py
LINE: 1
CATEGORY: [3]
SEVERITY: LOW
ISSUE: Near-empty source file
DETAIL: Only 8 non-comment/non-empty lines.

FILE: serenity/pipeline/concept.py
LINE: 1
CATEGORY: [3]
SEVERITY: LOW
ISSUE: Near-empty source file
DETAIL: Only 9 non-comment/non-empty lines.

FILE: serenity/tests/__init__.py
LINE: 1
CATEGORY: [3]
SEVERITY: LOW
ISSUE: Near-empty source file
DETAIL: Only 0 non-comment/non-empty lines.

FILE: serenity/tests/inference/__init__.py
LINE: 1
CATEGORY: [3]
SEVERITY: LOW
ISSUE: Near-empty source file
DETAIL: Only 0 non-comment/non-empty lines.

FILE: serenity/tests/test_boundary_violations.py
LINE: 142
CATEGORY: [3]
SEVERITY: LOW
ISSUE: Unfinished marker in code
DETAIL: TODO/FIXME/HACK marker indicates unfinished behavior.

FILE: serenity/training/loss.py
LINE: 1
CATEGORY: [3]
SEVERITY: LOW
ISSUE: Near-empty source file
DETAIL: Only 6 non-comment/non-empty lines.

FILE: serenity/training/trainer.py
LINE: 1
CATEGORY: [3]
SEVERITY: LOW
ISSUE: Near-empty source file
DETAIL: Only 3 non-comment/non-empty lines.

FILE: serenity/ui/__init__.py
LINE: 1
CATEGORY: [3]
SEVERITY: LOW
ISSUE: Near-empty source file
DETAIL: Only 2 non-comment/non-empty lines.

FILE: serenity/ui/__main__.py
LINE: 1
CATEGORY: [3]
SEVERITY: LOW
ISSUE: Near-empty source file
DETAIL: Only 8 non-comment/non-empty lines.

FILE: serenity/ui/main.py
LINE: 1
CATEGORY: [3]
SEVERITY: LOW
ISSUE: Near-empty source file
DETAIL: Only 8 non-comment/non-empty lines.

FILE: serenity/ui/web/__init__.py
LINE: 1
CATEGORY: [3]
SEVERITY: LOW
ISSUE: Near-empty source file
DETAIL: Only 4 non-comment/non-empty lines.

FILE: serenity/ui/web/__main__.py
LINE: 1
CATEGORY: [3]
SEVERITY: LOW
ISSUE: Near-empty source file
DETAIL: Only 5 non-comment/non-empty lines.

FILE: serenity/checkpoint/conversion.py
LINE: 1
CATEGORY: [9]
SEVERITY: LOW
ISSUE: No direct test references found
DETAIL: Module appears unreferenced by current test suite text search.

FILE: serenity/checkpoint/saver.py
LINE: 1
CATEGORY: [9]
SEVERITY: LOW
ISSUE: No direct test references found
DETAIL: Module appears unreferenced by current test suite text search.

FILE: serenity/core/callbacks.py
LINE: 1
CATEGORY: [9]
SEVERITY: LOW
ISSUE: No direct test references found
DETAIL: Module appears unreferenced by current test suite text search.

FILE: serenity/core/concept_config.py
LINE: 1
CATEGORY: [9]
SEVERITY: LOW
ISSUE: No direct test references found
DETAIL: Module appears unreferenced by current test suite text search.

FILE: serenity/core/config_migration.py
LINE: 1
CATEGORY: [9]
SEVERITY: LOW
ISSUE: No direct test references found
DETAIL: Module appears unreferenced by current test suite text search.

FILE: serenity/core/sample_config.py
LINE: 1
CATEGORY: [9]
SEVERITY: LOW
ISSUE: No direct test references found
DETAIL: Module appears unreferenced by current test suite text search.

FILE: serenity/core/weight_dtypes.py
LINE: 1
CATEGORY: [9]
SEVERITY: LOW
ISSUE: No direct test references found
DETAIL: Module appears unreferenced by current test suite text search.

FILE: serenity/inference/attention/token_merging.py
LINE: 1
CATEGORY: [9]
SEVERITY: LOW
ISSUE: No direct test references found
DETAIL: Module appears unreferenced by current test suite text search.

FILE: serenity/inference/utils/interrupt.py
LINE: 1
CATEGORY: [9]
SEVERITY: LOW
ISSUE: No direct test references found
DETAIL: Module appears unreferenced by current test suite text search.

FILE: serenity/models/flux_klein.py
LINE: 1
CATEGORY: [9]
SEVERITY: LOW
ISSUE: No direct test references found
DETAIL: Module appears unreferenced by current test suite text search.

FILE: serenity/pipeline/augmentations.py
LINE: 1
CATEGORY: [9]
SEVERITY: LOW
ISSUE: No direct test references found
DETAIL: Module appears unreferenced by current test suite text search.

FILE: serenity/pipeline/buckets.py
LINE: 1
CATEGORY: [9]
SEVERITY: LOW
ISSUE: No direct test references found
DETAIL: Module appears unreferenced by current test suite text search.

FILE: serenity/pipeline/masks.py
LINE: 1
CATEGORY: [9]
SEVERITY: LOW
ISSUE: No direct test references found
DETAIL: Module appears unreferenced by current test suite text search.

FILE: serenity/pipeline/staged_loader.py
LINE: 1
CATEGORY: [9]
SEVERITY: LOW
ISSUE: No direct test references found
DETAIL: Module appears unreferenced by current test suite text search.

FILE: serenity/training/flux2/edit_trainer.py
LINE: 1
CATEGORY: [9]
SEVERITY: LOW
ISSUE: No direct test references found
DETAIL: Module appears unreferenced by current test suite text search.

FILE: serenity/training/grad_scaler.py
LINE: 1
CATEGORY: [9]
SEVERITY: LOW
ISSUE: No direct test references found
DETAIL: Module appears unreferenced by current test suite text search.

FILE: serenity/training/optimizers.py
LINE: 1
CATEGORY: [9]
SEVERITY: LOW
ISSUE: No direct test references found
DETAIL: Module appears unreferenced by current test suite text search.

FILE: serenity/training/stochastic_rounding.py
LINE: 1
CATEGORY: [9]
SEVERITY: LOW
ISSUE: No direct test references found
DETAIL: Module appears unreferenced by current test suite text search.

FILE: serenity/training/tensorboard.py
LINE: 1
CATEGORY: [9]
SEVERITY: LOW
ISSUE: No direct test references found
DETAIL: Module appears unreferenced by current test suite text search.

FILE: serenity/training/token_pruning.py
LINE: 1
CATEGORY: [9]
SEVERITY: LOW
ISSUE: No direct test references found
DETAIL: Module appears unreferenced by current test suite text search.

FILE: serenity/training/vae_training.py
LINE: 1
CATEGORY: [9]
SEVERITY: LOW
ISSUE: No direct test references found
DETAIL: Module appears unreferenced by current test suite text search.

FILE: serenity/ui/theme.py
LINE: 1
CATEGORY: [9]
SEVERITY: LOW
ISSUE: No direct test references found
DETAIL: Module appears unreferenced by current test suite text search.

FILE: serenity/ui/web/api.py
LINE: 1
CATEGORY: [9]
SEVERITY: LOW
ISSUE: No direct test references found
DETAIL: Module appears unreferenced by current test suite text search.

FILE: serenity/ui/web/server.py
LINE: 1
CATEGORY: [9]
SEVERITY: LOW
ISSUE: No direct test references found
DETAIL: Module appears unreferenced by current test suite text search.

FILE: serenity/ui/web/websocket.py
LINE: 1
CATEGORY: [9]
SEVERITY: LOW
ISSUE: No direct test references found
DETAIL: Module appears unreferenced by current test suite text search.

FILE: serenity/ui/widgets.py
LINE: 1
CATEGORY: [9]
SEVERITY: LOW
ISSUE: No direct test references found
DETAIL: Module appears unreferenced by current test suite text search.

## Totals By Category

- Category 1: 60
- Category 2: 17
- Category 3: 24
- Category 4: 144
- Category 5: 11
- Category 6: 174
- Category 7: 8
- Category 8: 8
- Category 9: 26
- Category 10: 25

## Totals By Severity

- CRITICAL: 0
- HIGH: 93
- MEDIUM: 355
- LOW: 49

## Top 10 Worst Files

- serenity/cli/native_diffusion.py: 30
- serenity/sampling/sampler.py: 20
- serenity/models/flux2_klein.py: 19
- serenity/ui/tabs/inference.py: 17
- serenity/models/flux1.py: 13
- serenity/models/qwen.py: 12
- serenity/tests/inference/test_sampling_pipeline.py: 12
- serenity/models/zimage.py: 11
- serenity/tests/inference/test_model_adapters_flux.py: 11
- serenity/cli/native_flux2.py: 10

## Systemic Patterns

- Adapter stack is fragmented: vanilla LoRA uses PEFT while most advanced adapters proxy through LyCORIS, creating backend inconsistency and duplicated manager logic.
- Training/inference codepaths contain many silent-fallback patterns (broad exception suppression, dummy conditioning/denoise/decoder fallbacks), which hide misconfiguration and generate plausible-but-wrong outputs.
- Config surface area exceeds implementation: multiple training flags are exposed in UI/config but not consumed in active native loops (loss weighting, masking, bucketing/multi-resolution parity).
- Large monolithic CLI modules (`native_diffusion.py`, `native_flux2.py`) accumulate mixed responsibilities (data loading, model orchestration, optimization, checkpointing, sampling), increasing regression risk.
- Hardcoded machine-local paths remain in model/UI defaults, blocking portability and reproducibility across hosts.
- Checkpointing/resume semantics are inconsistent across training backends (diffusion has state resume path; flux2 loop mostly checkpoint-only without optimizer/scheduler continuity).
