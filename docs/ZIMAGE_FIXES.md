# Z-Image Training Fixes

## Date: 2026-01-20 (updated 2026-01-21)

## Summary

Fixed critical bugs in OneTrainer's Z-Image implementation that caused poor training and sampling results. Also fixed EriTrainer VAE decoding.

## Bugs Fixed

### 1. Sigma Calculation Off-by-One Error

**File**: `modules/modelSetup/mixin/ModelSetupFlowMatchingMixin.py`

**Problem**: The sigma calculation used array indexing with an off-by-one error:
```python
# OLD (BROKEN)
all_timesteps = torch.arange(start=1, end=num_timesteps + 1)  # [1, 2, ..., 1000]
self.__sigma = all_timesteps / num_timesteps  # [0.001, 0.002, ..., 1.0]
sigmas = self.__sigma[timestep]  # Wrong: timestep=0 gives 0.001, should be 0.0
```

**Fix**: Calculate sigma directly from timestep:
```python
# NEW (FIXED)
sigmas = timestep.float() / num_timesteps  # Direct calculation: sigma = t / T
```

**Impact**: This affects ALL flow matching models (Z-Image, Flux, SD3, etc.). The flow matching interpolation formula `noisy = sigma * noise + (1 - sigma) * latent` was using incorrect sigma values.

### 2. Z-Image Using Flux Timestep Shift Parameters

**File**: `modules/model/ZImageModel.py`

**Problem**: Z-Image's `calculate_timestep_shift()` was using Flux-specific defaults:
```python
# OLD (BROKEN)
# Comment: "these values are not defined in the scheduler config of Z-Image.
# They are therefore taken from the default of FlowMatchEulerDiscreteScheduler - which are Flux settings"
base_shift = 0.5   # Flux default
max_shift = 1.15   # Flux default
```

**Fix**: Return no shift since Z-Image doesn't define these parameters:
```python
# NEW (FIXED)
return 1.0  # No dynamic timestep shifting for Z-Image
```

**Impact**: Dynamic timestep shifting with wrong parameters caused incorrect noise schedules during training.

## Verification

After these fixes:
- Flow matching noise interpolation uses correct sigma values
- Z-Image training uses appropriate noise schedule (no Flux shift)
- Training loss should converge properly
- Sample quality should improve significantly

### 3. Sampler Missing Mu Parameter (Timestep Shifting)

**File**: `modules/modelSampler/ZImageSampler.py`

**Problem**: The sampler was not calculating or passing mu for timestep shifting:
```python
# OLD (BROKEN)
noise_scheduler.set_timesteps(diffusion_steps, device=self.train_device)  # Missing mu!
```

**Fix**: Calculate mu based on image sequence length:
```python
# NEW (FIXED)
latent_height = height // vae_scale_factor
latent_width = width // vae_scale_factor
image_seq_len = latent_height * latent_width // 4  # patch_size=2
mu = calculate_shift(image_seq_len)
noise_scheduler.set_timesteps(diffusion_steps, device=self.train_device, mu=mu)
```

**Impact**: Without mu, the scheduler generates incorrect timestep distributions for sampling.

### 4. Wrong CFG Formula

**File**: `modules/modelSampler/ZImageSampler.py`

**Problem**: Using standard CFG formula instead of Z-Image's non-standard formula:
```python
# OLD (BROKEN - standard CFG)
noise_pred = noise_pred_negative + cfg_scale * (noise_pred_positive - noise_pred_negative)
```

**Fix**: Use Z-Image's formula:
```python
# NEW (FIXED - Z-Image CFG)
noise_pred = noise_pred_positive + cfg_scale * (noise_pred_positive - noise_pred_negative)
```

**Impact**: This significantly affects sample quality. The standard formula is `neg + scale*(pos-neg)`, but Z-Image uses `pos + scale*(pos-neg)` which applies guidance more aggressively.

## Verification

After these fixes:
- Flow matching noise interpolation uses correct sigma values
- Z-Image training uses appropriate noise schedule (no Flux shift)
- Training loss should converge properly
- Sample quality should improve significantly
- **Sampling uses correct timestep distribution with mu**
- **CFG guidance applied correctly during inference**

## Related Files

- `modules/modelSetup/BaseZImageSetup.py` - Uses the fixed `_add_noise_discrete()`
- `modules/modelSetup/mixin/ModelSetupDiffusionLossMixin.py` - Loss calculation (unchanged, working as designed)
- `modules/modelSetup/mixin/ModelSetupNoiseMixin.py` - Timestep distribution (unchanged, working correctly)
- `modules/modelSampler/ZImageSampler.py` - Fixed mu parameter, CFG formula, timestep batching, and latent handling

### 5. Timestep Not Batched for CFG (2026-01-21)

**File**: `modules/modelSampler/ZImageSampler.py`

**Problem**: Timestep tensor was only 1 element even when CFG required batch_size=2:
```python
# OLD (BROKEN)
timestep_model_input = timestep.unsqueeze(0)  # Always [1] element
# ... passed to transformer expecting [batch_size] elements
```

**Fix**: Normalize timestep first, then repeat for CFG batch:
```python
# NEW (FIXED)
timestep_normalized = (1000 - timestep) / 1000
if cfg_scale > 1.0:
    timestep_model_input = timestep_normalized.unsqueeze(0).repeat(2)  # [2] for CFG
else:
    timestep_model_input = timestep_normalized.unsqueeze(0)
```

**Impact**: CFG was receiving mismatched batch sizes (latents=2, timesteps=1), causing incorrect predictions.

### 6. Latent Repeat vs Cat for CFG (2026-01-21)

**File**: `modules/modelSampler/ZImageSampler.py`

**Problem**: Used `torch.cat([latent] * 2)` instead of `latent.repeat(2, 1, 1, 1)`:
```python
# OLD (BROKEN)
latent_model_input = latent_image.unsqueeze(2).to(dtype=...)
latent_model_input = torch.cat([latent_model_input] * batch_size)
```

**Fix**: Match diffusers pattern with repeat:
```python
# NEW (FIXED)
latent_model_input = latent_image.to(dtype=...)
if cfg_scale > 1.0:
    latent_model_input = latent_model_input.repeat(2, 1, 1, 1)
latent_model_input = latent_model_input.unsqueeze(2)  # Add frame dim after repeat
```

**Impact**: The order of operations was wrong - unsqueeze then cat vs repeat then unsqueeze.

### 7. CFG Output Handling (2026-01-21)

**File**: `modules/modelSampler/ZImageSampler.py`

**Problem**: Used `chunk(2)` on stacked tensor instead of slicing output list:
```python
# OLD (BROKEN)
noise_pred = - torch.stack(output_list, dim=0).squeeze(dim=2)
noise_pred_positive, noise_pred_negative = noise_pred.chunk(2)
```

**Fix**: Match diffusers pattern - slice list, then process:
```python
# NEW (FIXED)
if cfg_scale > 1.0:
    pos_out = output_list[:1]
    neg_out = output_list[1:]
    pos = pos_out[0].float()
    neg = neg_out[0].float()
    noise_pred = pos + cfg_scale * (pos - neg)
    noise_pred = noise_pred.unsqueeze(0)
```

**Impact**: More robust handling of variable-length outputs.

### 8. EriTrainer VAE Missing shift_factor (2026-01-21)

**File**: `eritrainer/sampling/sampler.py`

**Problem**: VAE decoding only used scaling_factor, ignoring shift_factor:
```python
# OLD (BROKEN)
vae_scale = getattr(self._vae.config, "scaling_factor", 0.13025)
latents_scaled = latents / vae_scale
```

**Fix**: Include shift_factor in the formula:
```python
# NEW (FIXED)
vae_scale = getattr(self._vae.config, "scaling_factor", 0.13025)
vae_shift = getattr(self._vae.config, "shift_factor", 0.0)
latents_scaled = (latents / vae_scale) + vae_shift
```

**Impact**: Z-Image VAE requires both scaling and shifting for correct decoding. Missing shift caused color/contrast issues.

---

## Klein (FLUX.2) Fixes

### 9. EriTrainer Klein Missing Chat Template (2026-01-21)

**File**: `eritrainer/models/flux2_klein.py`

**Problem**: Text encoding was missing Qwen3 chat template:
```python
# OLD (BROKEN)
text_inputs = self.tokenizer(
    prompts,  # Raw prompts without chat formatting
    padding="max_length",
    ...
)
```

**Fix**: Apply chat template before tokenization:
```python
# NEW (FIXED)
formatted_prompts = []
for p in prompts:
    messages = [{"role": "user", "content": p}]
    formatted = self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # Klein uses False, Z-Image uses True
    )
    formatted_prompts.append(formatted)

text_inputs = self.tokenizer(
    formatted_prompts,
    ...
)
```

**Impact**: Without chat template, text embeddings are incorrect and model produces poor results.

**Note**: Klein uses `enable_thinking=False` while Z-Image uses `enable_thinking=True`.

### 10. OneTrainer Flux2Sampler VAE dtype (2026-01-21)

**File**: `modules/modelSampler/Flux2Sampler.py`

**Problem**: No dtype conversion before VAE decode:
```python
# OLD (BROKEN)
latents = self.model.unpatchify_latents(latents)
image = vae.decode(latents, return_dict=False)[0]
```

**Fix**: Convert to VAE dtype before decode:
```python
# NEW (FIXED)
latents = self.model.unpatchify_latents(latents)
latents = latents.to(vae.dtype)
image = vae.decode(latents, return_dict=False)[0]
```

**Impact**: Potential quality issues when latent dtype doesn't match VAE dtype.
