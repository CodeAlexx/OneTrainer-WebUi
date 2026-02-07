"""
FLUX.2 Klein Forward/Backward Pass Test

Tests the actual forward and backward pass with the correct dimensions
using EriTrainer's custom Flux2Transformer2DModel which is compatible with
FLUX.2 Klein weight naming.

Klein 4B config:
- num_attention_heads: 24
- attention_head_dim: 128
- inner_dim: 3072 (24 * 128)
- joint_attention_dim: 7680 (3 * 2560 from Qwen3)
- num_layers: 5 (double stream)
- num_single_layers: 20 (single stream)
- in_channels: 128 (after patchification)
- guidance_embeds: false

Reference: SimpleTuner flux2/transformer.py
"""

import gc
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add serenity to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Config from actual Klein 4B transformer config.json
KLEIN_4B_CONFIG = {
    "num_attention_heads": 24,
    "attention_head_dim": 128,
    "joint_attention_dim": 7680,  # 3 * 2560 (stacked Qwen3 layers)
    "num_layers": 5,  # double stream blocks
    "num_single_layers": 20,  # single stream blocks
    "in_channels": 128,  # after patchification
    "inner_dim": 24 * 128,  # = 3072
    "guidance_embeds": False,
}

KLEIN_9B_CONFIG = {
    "num_attention_heads": 32,
    "attention_head_dim": 128,
    "joint_attention_dim": 12288,  # 3 * 4096 (stacked Qwen3 layers)
    "num_layers": 8,  # double stream blocks
    "num_single_layers": 24,  # single stream blocks
    "in_channels": 128,  # after patchification
    "inner_dim": 32 * 128,  # = 4096
    "guidance_embeds": False,
}


def get_klein_4b_path() -> Path:
    """Get path to Klein 4B transformer folder."""
    hf_cache = Path.home() / ".cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-base-4B"
    if not hf_cache.exists():
        return None

    refs_main = hf_cache / "refs" / "main"
    if refs_main.exists():
        with open(refs_main) as f:
            revision = f.read().strip()
        snapshot_path = hf_cache / "snapshots" / revision
        return snapshot_path / "transformer"

    return None


# =============================================================================
# Test: Model Loading with Correct Config
# =============================================================================

class TestKleinModelLoading:
    """Test loading Klein transformer with correct dimensions."""

    def test_load_transformer_with_custom_loader(self):
        """Load transformer using EriTrainer's Flux2Transformer2DModel."""
        transformer_path = get_klein_4b_path()
        if transformer_path is None or not transformer_path.exists():
            print("  SKIPPED: Model not downloaded")
            return

        try:
            from serenity.models.flux2_transformer import Flux2Transformer2DModel

            print(f"  Loading from: {transformer_path}")

            # Load using custom loader - directly to GPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            transformer = Flux2Transformer2DModel.from_pretrained_klein(
                str(transformer_path),
                torch_dtype=torch.bfloat16,
                device=device,
            )

            # Verify config matches expected
            config = transformer.config
            print(f"  Loaded config:")
            print(f"    num_attention_heads: {config.num_attention_heads}")
            print(f"    attention_head_dim: {config.attention_head_dim}")
            print(f"    joint_attention_dim: {config.joint_attention_dim}")
            print(f"    num_layers: {config.num_layers}")
            print(f"    num_single_layers: {config.num_single_layers}")
            print(f"    in_channels: {config.in_channels}")
            print(f"    guidance_embeds: {config.guidance_embeds}")

            inner_dim = config.num_attention_heads * config.attention_head_dim
            print(f"    inner_dim (computed): {inner_dim}")

            # Count parameters
            total_params = sum(p.numel() for p in transformer.parameters())
            print(f"    total_params: {total_params / 1e9:.2f}B")

            # Verify against expected Klein 4B config
            assert config.num_attention_heads == KLEIN_4B_CONFIG["num_attention_heads"], \
                f"Expected {KLEIN_4B_CONFIG['num_attention_heads']} heads, got {config.num_attention_heads}"
            assert config.joint_attention_dim == KLEIN_4B_CONFIG["joint_attention_dim"], \
                f"Expected joint_attention_dim {KLEIN_4B_CONFIG['joint_attention_dim']}, got {config.joint_attention_dim}"
            assert inner_dim == KLEIN_4B_CONFIG["inner_dim"], \
                f"Expected inner_dim {KLEIN_4B_CONFIG['inner_dim']}, got {inner_dim}"
            assert config.guidance_embeds == KLEIN_4B_CONFIG["guidance_embeds"], \
                f"Expected guidance_embeds=False, got {config.guidance_embeds}"

            print("  Config matches Klein 4B expected values")

            del transformer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("  PASSED")

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            raise


# =============================================================================
# Test: Forward Pass with Correct Dimensions
# =============================================================================

class TestKleinForwardPass:
    """Test forward pass with correct input dimensions."""

    def test_forward_pass_gpu(self):
        """Test forward pass on GPU with minimal batch."""
        transformer_path = get_klein_4b_path()
        if transformer_path is None or not transformer_path.exists():
            print("  SKIPPED: Model not downloaded")
            return

        if not torch.cuda.is_available():
            print("  SKIPPED: No CUDA available (need GPU for Klein forward pass)")
            return

        try:
            from serenity.models.flux2_transformer import Flux2Transformer2DModel

            print(f"  Loading transformer...")

            transformer = Flux2Transformer2DModel.from_pretrained_klein(
                str(transformer_path),
                torch_dtype=torch.bfloat16,
                device="cuda",
            )
            transformer.eval()

            # Create test inputs with correct dimensions
            batch_size = 1
            seq_len_img = 64  # 8x8 latent spatial
            seq_len_txt = 128

            # From config
            in_channels = 128  # after patchification
            joint_attention_dim = 7680  # stacked Qwen3 layers

            print(f"  Creating test inputs:")
            print(f"    hidden_states: ({batch_size}, {seq_len_img}, {in_channels})")
            print(f"    encoder_hidden_states: ({batch_size}, {seq_len_txt}, {joint_attention_dim})")

            device = torch.device("cuda")

            # Image latents (packed): (B, S, C) where C=128
            hidden_states = torch.randn(
                batch_size, seq_len_img, in_channels,
                dtype=torch.bfloat16, device=device
            )

            # Text embeddings: (B, L, D) where D=7680
            encoder_hidden_states = torch.randn(
                batch_size, seq_len_txt, joint_attention_dim,
                dtype=torch.bfloat16, device=device
            )

            # Timestep (normalized 0-1)
            timestep = torch.tensor([0.5], dtype=torch.bfloat16, device=device)

            # Position IDs for image tokens (S, 4) - t, h, w, l
            h, w = 8, 8  # sqrt(64) = 8
            img_ids = torch.zeros(seq_len_img, 4, dtype=torch.long, device=device)
            for i in range(seq_len_img):
                img_ids[i, 0] = 0  # t
                img_ids[i, 1] = i // w  # h
                img_ids[i, 2] = i % w   # w
                img_ids[i, 3] = 0  # l

            # Position IDs for text tokens (L, 4)
            txt_ids = torch.zeros(seq_len_txt, 4, dtype=torch.long, device=device)
            for i in range(seq_len_txt):
                txt_ids[i, 0] = 0  # t
                txt_ids[i, 1] = 0  # h (dummy)
                txt_ids[i, 2] = 0  # w (dummy)
                txt_ids[i, 3] = i  # l (sequence position)

            # Klein has NO guidance, so don't pass it
            print("  Running forward pass (no guidance for Klein)...")

            with torch.no_grad():
                output = transformer(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep,
                    img_ids=img_ids,
                    txt_ids=txt_ids,
                    guidance=None,  # Klein has no guidance embeddings
                    return_dict=True,
                )

            model_output = output.sample
            print(f"  Output shape: {model_output.shape}")
            print(f"  Expected: ({batch_size}, {seq_len_img}, {in_channels})")

            assert model_output.shape == (batch_size, seq_len_img, in_channels), \
                f"Output shape mismatch: {model_output.shape}"

            # Check for NaN/Inf
            assert not torch.isnan(model_output).any(), "Output contains NaN"
            assert not torch.isinf(model_output).any(), "Output contains Inf"

            print(f"  Output stats: min={model_output.min():.4f}, max={model_output.max():.4f}, mean={model_output.mean():.4f}")

            del transformer, hidden_states, encoder_hidden_states, output
            gc.collect()
            torch.cuda.empty_cache()

            print("  PASSED")

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            raise

    def test_backward_pass(self):
        """Test backward pass (gradient computation)."""
        transformer_path = get_klein_4b_path()
        if transformer_path is None or not transformer_path.exists():
            print("  SKIPPED: Model not downloaded")
            return

        if not torch.cuda.is_available():
            print("  SKIPPED: No CUDA available")
            return

        try:
            from serenity.models.flux2_transformer import Flux2Transformer2DModel

            print(f"  Loading transformer for gradient test...")

            transformer = Flux2Transformer2DModel.from_pretrained_klein(
                str(transformer_path),
                torch_dtype=torch.bfloat16,
                device="cuda",
            )
            transformer.train()  # Training mode for gradients

            device = torch.device("cuda")

            # Small test dimensions
            batch_size = 1
            seq_len_img = 16  # 4x4 latent
            seq_len_txt = 32
            in_channels = 128
            joint_attention_dim = 7680

            # Create inputs
            hidden_states = torch.randn(
                batch_size, seq_len_img, in_channels,
                dtype=torch.bfloat16, device=device,
                requires_grad=True
            )
            encoder_hidden_states = torch.randn(
                batch_size, seq_len_txt, joint_attention_dim,
                dtype=torch.bfloat16, device=device
            )
            timestep = torch.tensor([0.5], dtype=torch.bfloat16, device=device)

            # Position IDs
            h, w = 4, 4
            img_ids = torch.zeros(seq_len_img, 4, dtype=torch.long, device=device)
            for i in range(seq_len_img):
                img_ids[i, 1] = i // w
                img_ids[i, 2] = i % w

            txt_ids = torch.zeros(seq_len_txt, 4, dtype=torch.long, device=device)
            for i in range(seq_len_txt):
                txt_ids[i, 3] = i

            print("  Running forward pass for backward test...")

            output = transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=None,
                return_dict=True,
            )

            print("  Computing loss and backward...")

            # Simple MSE loss with dummy target
            target = torch.randn_like(output.sample)
            loss = nn.functional.mse_loss(output.sample, target)

            loss.backward()

            print(f"  Loss: {loss.item():.6f}")

            # Check gradients exist
            grad_count = 0
            for name, param in transformer.named_parameters():
                if param.grad is not None:
                    grad_count += 1

            print(f"  Parameters with gradients: {grad_count}")
            assert grad_count > 0, "No gradients computed"

            # Check for NaN gradients
            nan_grads = 0
            for name, param in transformer.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    nan_grads += 1

            if nan_grads > 0:
                print(f"  WARNING: {nan_grads} parameters have NaN gradients")
            else:
                print("  No NaN gradients detected")

            del transformer, hidden_states, encoder_hidden_states, output
            gc.collect()
            torch.cuda.empty_cache()

            print("  PASSED")

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            raise


# =============================================================================
# Run Tests
# =============================================================================

def run_all_tests():
    """Run all Klein forward/backward tests."""
    print("=== FLUX.2 Klein Forward/Backward Tests ===\n")
    print("Klein 4B Config Reference:")
    for k, v in KLEIN_4B_CONFIG.items():
        print(f"  {k}: {v}")
    print()

    print("Test: Model Loading with Custom Loader")
    t = TestKleinModelLoading()
    try:
        t.test_load_transformer_with_custom_loader()
    except Exception as e:
        print(f"  FAILED: {e}")

    print("\nTest: Forward Pass")
    t = TestKleinForwardPass()
    try:
        t.test_forward_pass_gpu()
    except Exception as e:
        print(f"  FAILED: {e}")

    print("\nTest: Backward Pass")
    try:
        t.test_backward_pass()
    except Exception as e:
        print(f"  FAILED: {e}")

    print("\n=== Tests Complete ===")


if __name__ == "__main__":
    run_all_tests()
