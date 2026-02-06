#!/usr/bin/env python3
"""
Simplified LTX2 Layer Offload Test

Tests layer offloading with the 19B transformer without loading the full text encoder.
Uses mock text embeddings to focus on the layer offloading mechanism.
"""

import torch
import torch.nn.functional as F
import gc
from pathlib import Path

TRANSFORMER_PATH = "/home/alex/eriui/comfyui/ComfyUI/models/diffusion_models/ltx-2-19b-dev-fp8.safetensors"

# Gemma 3 hidden dim for caption_channels
GEMMA3_HIDDEN_DIM = 3840


def main():
    print("=" * 60)
    print("LTX2 Layer Offload Test (Simplified)")
    print("=" * 60)

    # Check transformer path
    if not Path(TRANSFORMER_PATH).exists():
        print(f"ERROR: Transformer not found: {TRANSFORMER_PATH}")
        return 1

    # Clear GPU memory
    gc.collect()
    torch.cuda.empty_cache()

    # Load transformer using from_single_file (handles ComfyUI format automatically)
    print("\n1. Loading transformer (no text encoder)...")
    from diffusers import LTXVideoTransformer3DModel, FlowMatchEulerDiscreteScheduler

    print("   Loading via from_single_file...")
    transformer = LTXVideoTransformer3DModel.from_single_file(
        TRANSFORMER_PATH,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        caption_channels=GEMMA3_HIDDEN_DIM,
    )
    print(f"   Loaded transformer with {len(list(transformer.transformer_blocks))} blocks")

    gc.collect()

    # Import layer offload manager
    print("\n2. Setting up layer offloading...")
    from eritrainer.utils.layer_offload import LayerOffloadManager

    layers = list(transformer.transformer_blocks)
    print(f"   Found {len(layers)} transformer blocks")

    # Create manager with 85% offload (15% on GPU = ~7 blocks)
    manager = LayerOffloadManager(
        layers=layers,
        layer_offload_fraction=0.85,
        train_device="cuda:0",
        temp_device="cpu",
        use_checkpoint=True,
    )

    # Activate offloading
    manager.activate()

    # Move non-block components to GPU
    device = torch.device("cuda:0")
    if hasattr(transformer, 'proj_in'):
        transformer.proj_in.to(device)
    if hasattr(transformer, 'proj_out'):
        transformer.proj_out.to(device)
    if hasattr(transformer, 'time_embed'):
        transformer.time_embed.to(device)
    if hasattr(transformer, 'caption_projection'):
        transformer.caption_projection.to(device)
    if hasattr(transformer, 'scale_shift_table') and transformer.scale_shift_table is not None:
        transformer.scale_shift_table.data = transformer.scale_shift_table.data.to(device)

    gc.collect()
    torch.cuda.empty_cache()
    mem = torch.cuda.memory_allocated() / 1e9
    print(f"   GPU memory after setup: {mem:.2f} GB")

    # Create test batch
    print("\n3. Creating test batch...")
    batch_size = 1
    num_frames = 9  # 8*1 + 1
    height, width = 16, 16  # Latent size

    # Mock latents
    latent_video = torch.randn(
        batch_size, 128, num_frames, height, width,
        device=device, dtype=torch.bfloat16
    )

    # Mock text embeddings (Gemma 3 hidden dim = 3840)
    text_embeddings = torch.randn(
        batch_size, 77, 3840,
        device=device, dtype=torch.bfloat16
    )
    text_mask = torch.ones(batch_size, 77, device=device, dtype=torch.long)

    print(f"   Latent shape: {latent_video.shape}")
    print(f"   Text shape: {text_embeddings.shape}")

    # Training loop
    print("\n4. Running training loop...")
    transformer.train()

    # Enable gradients
    for p in transformer.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(transformer.parameters(), lr=1e-5)
    scheduler = FlowMatchEulerDiscreteScheduler()

    num_steps = 3  # Reduced for faster testing
    losses = []
    print(f"   Running {num_steps} training steps (reduced for speed)...")

    for step in range(num_steps):
        optimizer.zero_grad()

        # Start forward tracking
        manager.start_forward()

        # Sample random timestep (index on CPU, then move result to device)
        t_cpu = torch.randint(0, 1000, (batch_size,))
        sigmas = scheduler.sigmas[t_cpu].to(device=device, dtype=torch.bfloat16)

        # Add noise
        noise = torch.randn_like(latent_video)
        noisy_latent = latent_video + sigmas.view(-1, 1, 1, 1, 1) * noise

        # Patchify input - reshape [B, C, T, H, W] -> [B, num_patches, C]
        b, c, t_frames, h, w = noisy_latent.shape
        hidden_states = noisy_latent.permute(0, 2, 3, 4, 1).reshape(b, -1, c)

        # Scale timesteps
        scaled_timesteps = (1.0 - sigmas) * 1000
        scaled_timesteps = scaled_timesteps.long()

        # Forward pass through transformer
        output = transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=text_embeddings,
            timestep=scaled_timesteps,
            encoder_attention_mask=text_mask,
            num_frames=t_frames,
            height=h,
            width=w,
            return_dict=False,
        )

        # Get prediction
        if isinstance(output, tuple):
            predicted = output[0]
        else:
            predicted = output.sample

        # Reshape back - [B, num_patches, C] -> [B, C, T, H, W]
        predicted = predicted.reshape(b, t_frames, h, w, c).permute(0, 4, 1, 2, 3)

        # Flow matching target: noise - latent
        target = noise - latent_video

        # Loss
        loss = F.mse_loss(predicted.float(), target.float())

        # Backward
        loss.backward()

        # End step
        manager.end_step()

        # Optimizer step
        optimizer.step()

        losses.append(loss.item())
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"   Step {step + 1}/{num_steps}: loss = {loss.item():.6f}, peak GPU = {peak_mem:.2f} GB")

    # Results
    print("\n5. Results:")
    print(f"   First loss: {losses[0]:.6f}")
    print(f"   Last loss: {losses[-1]:.6f}")
    print(f"   Average loss: {sum(losses)/len(losses):.6f}")

    if losses[-1] < losses[0]:
        print("   Loss is decreasing (training is working)")
    else:
        print("   Loss did not decrease (may need more steps)")

    # Memory stats
    stats = manager.get_memory_stats()
    print(f"\n6. Layer offload stats:")
    print(f"   Total layers: {stats['total_layers']}")
    print(f"   Loaded: {stats['loaded_layers']} ({stats['loaded_bytes']/1e9:.2f} GB)")
    print(f"   Offloaded: {stats['offloaded_layers']} ({stats['offloaded_bytes']/1e9:.2f} GB)")

    print("\n" + "=" * 60)
    print("TEST PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
