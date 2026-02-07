# Serenity LTX2 Alignment Notes

## Scope
Align Serenity LTX2 training and caching with OneTrainer and upstream LTX2
behavior while avoiding MGDS.

## Key Differences Identified
- VAE caching path: Serenity used a diffusers-style VAE loader; upstream LTX2
  expects ltx_core VideoEncoder/VideoDecoder for video latents.
- Loss space: Serenity compared patchified predictions against unpatchified
  targets, which mismatched shapes and training objectives.
- Noise/timestep sampling: Serenity used uniform sampling; OneTrainer uses
  shifted logit-normal sampling based on sequence length and supports offset and
  perturbation noise.
- Offload/checkpointing: Serenity used staged offload and legacy activation
  offloading; OneTrainer uses per-block checkpointing with a LayerOffloadConductor.

## Fixes Applied
- `serenity/models/ltx2.py`: load VAE-only video encoder/decoder from ltx_core
  for video caching compatibility.
- `serenity/components/model_setup.py`: LTX2 flow matching now unpatchifies
  predicted flow, uses shifted logit-normal timesteps, and matches OneTrainer
  noise options (offset/perturbation).
- `serenity/training/ltx2_checkpointing.py`: OneTrainer-style per-block
  checkpointing with LayerOffloadConductor (no MGDS).
- `serenity/core/trainer.py`: when LayerOffloadConductor is active, skip the
  legacy activation offloading wrapper to avoid double offload paths.

## Other Changes Included
- `modules/modelSetup/BaseZImageSetup.py`: remove debug image saves for flow and
  predictions.
- `web_ui/backend/services/caption_service.py` and `web_ui/backend/api/caption.py`:
  track total batch size in caption progress and default caption dtype to bf16.

## Testing Notes
Run a short training step to validate memory/offload behavior:
`venv/bin/python -m serenity train --config serenity_ltx2_test.yaml --steps 1`
