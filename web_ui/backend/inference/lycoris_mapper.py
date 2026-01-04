"""
LyCORIS Name Mapping Utility

Maps LyCORIS weight key names between OneTrainer and diffusers conventions.
Handles the conversion of module names from OneTrainer's format to diffusers' format.
"""

import re
from typing import Dict, Any, Tuple, Optional
from pathlib import Path


# OneTrainer to Diffusers module name mappings for common architectures
FLUX_MAPPINGS = {
    # OneTrainer LyCORIS format -> Diffusers transformer format
    "lycoris_layers_": "transformer.transformer_blocks.",
    "adaLN_modulation_0": "norm1",
    "attention_to_k": "attn.to_k",
    "attention_to_q": "attn.to_q",
    "attention_to_v": "attn.to_v",
    "attention_to_out_0": "attn.to_out.0",
    "feed_forward_w1": "ff.net.0.proj",
    "feed_forward_w2": "ff.net.2",
    "feed_forward_w3": "ff_context.net.0.proj",
}

SDXL_MAPPINGS = {
    # OneTrainer format -> Diffusers UNet format
    "lora_unet_": "unet.",
    "down_blocks_": "down_blocks.",
    "up_blocks_": "up_blocks.",
    "mid_block_": "mid_block.",
    "resnets_": "resnets.",
    "attentions_": "attentions.",
    "downsamplers_": "downsamplers.",
    "upsamplers_": "upsamplers.",
    "conv": "conv",
    "time_emb_proj": "time_emb_proj",
    "to_k": "to_k",
    "to_q": "to_q",
    "to_v": "to_v",
    "to_out": "to_out",
    "proj_in": "proj_in",
    "proj_out": "proj_out",
}

SD3_MAPPINGS = {
    # SD3/SD3.5 transformer mappings
    "lycoris_transformer_": "transformer.",
    "transformer_blocks_": "transformer_blocks.",
    "attn_": "attn.",
    "norm_": "norm",
    "ff_": "ff.",
}


def detect_architecture(state_dict: Dict[str, Any]) -> str:
    """Detect the model architecture from state dict keys."""
    keys = list(state_dict.keys())
    first_keys = keys[:20]  # Check first 20 keys

    # FLUX detection
    flux_patterns = ["lycoris_layers_", "adaLN_modulation", "feed_forward_w"]
    if any(any(p in k for p in flux_patterns) for k in first_keys):
        return "flux"

    # SD3/SD3.5 detection
    sd3_patterns = ["lycoris_transformer_", "transformer_blocks_"]
    if any(any(p in k for p in sd3_patterns) for k in first_keys):
        return "sd3"

    # SDXL/SD1.5 UNet detection
    unet_patterns = ["lora_unet_", "unet_"]
    if any(any(p in k for p in unet_patterns) for k in first_keys):
        return "sdxl"

    return "unknown"


def map_onetrainer_to_diffusers(key: str, architecture: str = "auto") -> str:
    """
    Map a OneTrainer LyCORIS key to diffusers format.

    Args:
        key: The original key from OneTrainer LyCORIS
        architecture: Target architecture (flux, sdxl, sd3, auto)

    Returns:
        The mapped key for diffusers
    """
    # Select mapping based on architecture
    if architecture == "flux":
        mappings = FLUX_MAPPINGS
    elif architecture == "sdxl":
        mappings = SDXL_MAPPINGS
    elif architecture == "sd3":
        mappings = SD3_MAPPINGS
    else:
        # Try to auto-detect from key
        if "lycoris_layers_" in key:
            mappings = FLUX_MAPPINGS
        elif "lycoris_transformer_" in key:
            mappings = SD3_MAPPINGS
        else:
            mappings = SDXL_MAPPINGS

    result = key
    for old, new in mappings.items():
        result = result.replace(old, new)

    return result


def convert_lycoris_state_dict(
    state_dict: Dict[str, Any],
    target_module_names: Dict[str, str] = None,
    architecture: str = "auto"
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    Convert a LyCORIS state dict from OneTrainer format to diffusers format.

    Args:
        state_dict: Original state dict from LyCORIS file
        target_module_names: Optional mapping of diffusers module names
        architecture: Target architecture

    Returns:
        Tuple of (converted_state_dict, key_mapping)
    """
    if architecture == "auto":
        architecture = detect_architecture(state_dict)
        print(f"Detected architecture: {architecture}")

    converted = {}
    key_mapping = {}

    for key, value in state_dict.items():
        new_key = map_onetrainer_to_diffusers(key, architecture)

        # If target module names provided, try to match
        if target_module_names:
            matched = False
            for module_name, target_name in target_module_names.items():
                if module_name in new_key:
                    new_key = new_key.replace(module_name, target_name)
                    matched = True
                    break

        converted[new_key] = value
        if new_key != key:
            key_mapping[key] = new_key

    return converted, key_mapping


def create_lycoris_with_mapping(
    lycoris_path: str,
    base_model,
    multiplier: float = 1.0,
    architecture: str = "auto"
):
    """
    Create a LyCORIS network with automatic key mapping.

    Args:
        lycoris_path: Path to the LyCORIS safetensors file
        base_model: The base model (UNet or Transformer)
        multiplier: Weight multiplier
        architecture: Target architecture

    Returns:
        The LyCORIS network ready to apply
    """
    import sys
    from pathlib import Path as PyPath
    from safetensors.torch import load_file

    # Add LyCORIS to path
    lycoris_lib_path = PyPath("/home/alex/diffusion-pipe-lyco/LyCORIS")
    if lycoris_lib_path.exists() and str(lycoris_lib_path) not in sys.path:
        sys.path.insert(0, str(lycoris_lib_path))

    from lycoris import create_lycoris_from_weights

    # Load and convert state dict
    print(f"Loading LyCORIS from: {lycoris_path}")
    state_dict = load_file(lycoris_path, device="cpu")

    # Detect architecture
    if architecture == "auto":
        architecture = detect_architecture(state_dict)
        print(f"Detected architecture: {architecture}")

    # Get target module names from base model
    target_modules = {}
    for name, module in base_model.named_modules():
        # Store simplified name -> full name mapping
        simple_name = name.replace(".", "_")
        target_modules[simple_name] = name

    # Convert state dict
    converted_dict, key_mapping = convert_lycoris_state_dict(
        state_dict,
        target_module_names=target_modules,
        architecture=architecture
    )

    if key_mapping:
        print(f"Mapped {len(key_mapping)} keys")
        # Show a few examples
        for i, (old, new) in enumerate(list(key_mapping.items())[:3]):
            print(f"  {old} -> {new}")
        if len(key_mapping) > 3:
            print(f"  ... and {len(key_mapping) - 3} more")

    # Create LyCORIS network with converted weights
    try:
        network, weights_sd = create_lycoris_from_weights(
            multiplier,
            lycoris_path,
            base_model,
            weights_sd=converted_dict
        )

        print(f"Created LyCORIS network with {len(network.loras)} modules")
        return network

    except Exception as e:
        print(f"Failed to create LyCORIS network: {e}")
        # Fallback: try with original state dict
        print("Trying with original state dict...")
        network, weights_sd = create_lycoris_from_weights(
            multiplier,
            lycoris_path,
            base_model
        )
        return network


def get_model_module_names(model) -> Dict[str, str]:
    """Get all module names from a model for debugging."""
    modules = {}
    for name, module in model.named_modules():
        module_type = module.__class__.__name__
        modules[name] = module_type
    return modules


def analyze_lycoris_file(lycoris_path: str) -> Dict[str, Any]:
    """Analyze a LyCORIS file and return information about its structure."""
    from safetensors.torch import load_file

    state_dict = load_file(lycoris_path, device="cpu")

    # Detect architecture
    architecture = detect_architecture(state_dict)

    # Analyze key patterns
    key_patterns = {}
    for key in state_dict.keys():
        # Extract the base pattern (remove numbers and specific names)
        pattern = re.sub(r'_\d+', '_N', key)
        pattern = re.sub(r'\.\d+\.', '.N.', pattern)
        if pattern not in key_patterns:
            key_patterns[pattern] = 0
        key_patterns[pattern] += 1

    # Detect LyCORIS type from keys
    lycoris_type = "unknown"
    for key in state_dict.keys():
        if "lokr" in key.lower():
            lycoris_type = "lokr"
            break
        elif "loha" in key.lower():
            lycoris_type = "loha"
            break
        elif "lora_down" in key.lower() or "lora_up" in key.lower():
            lycoris_type = "lora"
            break
        elif "oft" in key.lower():
            lycoris_type = "oft"
            break
        elif "boft" in key.lower():
            lycoris_type = "boft"
            break
        elif "ia3" in key.lower():
            lycoris_type = "ia3"
            break
        elif "glora" in key.lower():
            lycoris_type = "glora"
            break

    return {
        "path": lycoris_path,
        "architecture": architecture,
        "lycoris_type": lycoris_type,
        "num_keys": len(state_dict),
        "key_patterns": key_patterns,
        "sample_keys": list(state_dict.keys())[:10]
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python lycoris_mapper.py <lycoris_file.safetensors>")
        sys.exit(1)

    # Analyze the file
    info = analyze_lycoris_file(sys.argv[1])

    print(f"\nLyCORIS File Analysis:")
    print(f"  Path: {info['path']}")
    print(f"  Architecture: {info['architecture']}")
    print(f"  LyCORIS Type: {info['lycoris_type']}")
    print(f"  Number of keys: {info['num_keys']}")
    print(f"\nSample keys:")
    for key in info['sample_keys']:
        print(f"  {key}")
    print(f"\nKey patterns ({len(info['key_patterns'])} unique):")
    for pattern, count in list(info['key_patterns'].items())[:10]:
        print(f"  {pattern}: {count}")
