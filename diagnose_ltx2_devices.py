#!/usr/bin/env python3
"""Diagnose device placement for LTX2 transformer modules."""

import torch
from pathlib import Path

TRANSFORMER_PATH = "/home/alex/eriui/comfyui/ComfyUI/models/diffusion_models/ltx-2-19b-dev-fp8.safetensors"
TEXT_ENCODER_PATH = "/home/alex/models/gemma-3-12b-it"


def main():
    print("=" * 60)
    print("LTX2 Transformer Device Diagnostic")
    print("=" * 60)

    # Import and load model
    from eritrainer.models.ltx2 import LTX2Model

    print("\n1. Loading model...")
    model = LTX2Model(
        model_path=TRANSFORMER_PATH,
        text_encoder_path=TEXT_ENCODER_PATH,
        dtype=torch.bfloat16,
    )

    transformer = model._transformer
    print(f"   Transformer type: {type(transformer).__name__}")

    # List top-level modules
    print("\n2. Top-level modules in transformer:")
    for name, module in transformer.named_children():
        # Get first param device
        params = list(module.parameters())
        buffers = list(module.buffers())
        if params:
            device = params[0].device
            param_count = len(params)
            print(f"   {name}: {type(module).__name__} ({param_count} params, device={device})")
        elif buffers:
            device = buffers[0].device
            print(f"   {name}: {type(module).__name__} (buffers only, device={device})")
        else:
            print(f"   {name}: {type(module).__name__} (no params/buffers)")

    # Find blocks
    print("\n3. Looking for transformer blocks...")
    blocks = model._get_transformer_blocks()
    print(f"   Found {len(blocks)} blocks")
    if blocks:
        first_block = blocks[0]
        print(f"   First block type: {type(first_block).__name__}")
        print(f"   First block submodules:")
        for name, mod in first_block.named_children():
            params = list(mod.parameters())
            if params:
                print(f"      {name}: {type(mod).__name__} ({len(params)} params)")

    # Now enable layer offloading and check again
    # Use 85% to allow ~7 blocks on GPU at a time (15% of 48 = 7.2 blocks)
    print("\n4. Enabling layer offloading (85%)...")
    model.enable_layer_offload(
        layer_offload_fraction=0.85,
        train_device="cuda:0",
        temp_device="cpu",
    )

    # Check all modules again
    print("\n5. Module devices after offloading:")
    for name, param in transformer.named_parameters():
        parts = name.split(".")
        if len(parts) <= 3:  # Top-level or shallow
            print(f"   {name}: {param.device}")
        elif parts[0] == "transformer_blocks" and parts[1] in ["0", "1"]:
            print(f"   {name}: {param.device}")

    # Check buffers
    print("\n6. Buffer devices after offloading:")
    for name, buffer in transformer.named_buffers():
        print(f"   {name}: {buffer.device}")

    # Try a minimal forward pass
    print("\n7. Testing minimal forward...")
    device = torch.device("cuda:0")
    batch_size = 1
    latent_channels = 128
    hidden_dim = transformer.config.hidden_size if hasattr(transformer.config, 'hidden_size') else 3072
    print(f"   hidden_dim from config: {hidden_dim}")

    # Create minimal input
    # LTX2 expects patchified input [B, num_patches, C]
    num_patches = 9 * 16 * 16  # T * H * W
    hidden_states = torch.randn(
        batch_size, num_patches, latent_channels,
        device=device, dtype=torch.bfloat16
    )
    print(f"   Input hidden_states: {hidden_states.shape}, device={hidden_states.device}")

    # Check proj_in specifically
    if hasattr(transformer, 'proj_in'):
        proj_in = transformer.proj_in
        print(f"\n8. proj_in details:")
        print(f"   Type: {type(proj_in)}")
        for name, param in proj_in.named_parameters():
            print(f"   {name}: {param.shape}, device={param.device}")
        for name, buffer in proj_in.named_buffers():
            print(f"   buffer {name}: device={buffer.device}")

        # Try just proj_in forward
        try:
            print("\n   Trying proj_in forward...")
            out = proj_in(hidden_states)
            print(f"   proj_in output: {out.shape}, device={out.device}")
        except Exception as e:
            print(f"   proj_in FAILED: {e}")

    # Check for patch_embed or similar
    for attr in ['patch_embed', 'pos_embed', 'time_embed', 'adaln_single']:
        if hasattr(transformer, attr):
            mod = getattr(transformer, attr)
            params = list(mod.parameters())
            if params:
                print(f"\n{attr}: {len(params)} params, first device={params[0].device}")

    # Free memory
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\n   GPU memory before test: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated")

    # Now test a forward through the first few blocks
    print("\n9. Testing forward through blocks with offloading...")

    # Get the blocks
    blocks = model._get_transformer_blocks()
    if len(blocks) > 0:
        # The forward wrapper was installed during enable_layer_offload
        # Let's manually call start_forward first
        model._layer_offload_manager.start_forward()

        # Create minimal test input for a single block
        # LTX transformer blocks expect hidden_states in shape [B, seq_len, hidden_dim]
        hidden_dim = transformer.config.hidden_size if hasattr(transformer.config, 'hidden_size') else 4096
        seq_len = 100
        test_input = torch.randn(1, seq_len, hidden_dim, device=device, dtype=torch.bfloat16)

        # Try calling first block (patched forward)
        try:
            print(f"   Calling block 0 with input shape {test_input.shape}...")
            # LTX blocks need more params - let's check the signature
            import inspect
            sig = inspect.signature(blocks[0].forward if hasattr(blocks[0], '_original_forward') else blocks[0].forward)
            print(f"   Block forward signature: {sig}")

            # Can't easily test individual blocks without proper inputs
            # Let's try the full forward instead
            print("\n   Testing full model.forward() with mock batch...")
            # Use smaller latents to save memory
            # frames=9, height=8, width=8 (minimum valid size)

            batch = {
                "latent_video": torch.randn(1, 128, 1, 8, 8, device=device, dtype=torch.bfloat16),
                "text_encoder_hidden_state": torch.randn(1, 16, 3840, device=device, dtype=torch.bfloat16),
                "text_encoder_mask": torch.ones(1, 16, device=device, dtype=torch.long),
            }
            print(f"   Batch: latent_video={batch['latent_video'].shape}")

            # Test with gradients - checkpointing is now built-in to layer offloading
            output = model.forward(batch)
            print(f"   Forward SUCCESS! Output keys: {output.keys()}")
            print(f"   Predicted shape: {output['predicted'].shape}")
            print(f"   Target shape: {output['target'].shape}")

            # Test backward pass
            print("\n   Testing backward pass...")
            import torch.nn.functional as F
            loss = F.mse_loss(output['predicted'].float(), output['target'].float())
            print(f"   Loss: {loss.item():.6f}")
            loss.backward()
            print(f"   Backward SUCCESS!")

            # Check GPU memory after backward
            mem_after = torch.cuda.memory_allocated() / 1e9
            print(f"   GPU memory after backward: {mem_after:.2f} GB")

        except Exception as e:
            print(f"   FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Diagnostic complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
