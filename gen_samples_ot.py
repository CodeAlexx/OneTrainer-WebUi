#!/usr/bin/env python3
"""Generate samples using OneTrainer's native LoRA loading."""
import sys
sys.path.insert(0, "/home/alex/OneTrainer")

import torch
import os

from modules.model.Flux2Model import Flux2Model
from modules.modelLoader.Flux2ModelLoader import Flux2LoRAModelLoader
from modules.modelSampler.Flux2Sampler import Flux2Sampler
from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.NoiseScheduler import NoiseScheduler
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.config.TrainConfig import QuantizationConfig

LORA_PATH = "/home/alex/OneTrainer/workspace/run/backup/2026-01-17_16-16-14-backup-2400-25-0/lora/lora.safetensors"
BASE_MODEL = "black-forest-labs/FLUX.2-klein-base-4B"
OUTPUT_DIR = "/home/alex/OneTrainer/lady_samples"

PROMPTS = [
    "a beautiful woman in a red dress standing in a garden at sunset, elegant, photorealistic",
    "a young woman with curly hair reading a book in a cozy cafe, warm lighting, candid portrait",
    "a professional woman in a business suit giving a presentation, confident, modern office",
    "a woman hiking on a mountain trail at sunrise, athletic wear, scenic landscape background",
    "a woman artist painting in her studio, creative, colorful paint splatters, natural light",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Setup model names
model_names = ModelNames(
    base_model=BASE_MODEL,
    lora=LORA_PATH,
)

# Setup weight dtypes
weight_dtypes = ModelWeightDtypes(
    train_dtype=DataType.BFLOAT_16,
    fallback_train_dtype=DataType.BFLOAT_16,
    transformer=DataType.BFLOAT_16,
    unet=DataType.BFLOAT_16,
    prior=DataType.BFLOAT_16,
    text_encoder=DataType.BFLOAT_16,
    text_encoder_2=DataType.BFLOAT_16,
    text_encoder_3=DataType.BFLOAT_16,
    text_encoder_4=DataType.BFLOAT_16,
    vae=DataType.FLOAT_32,
    effnet_encoder=DataType.BFLOAT_16,
    decoder=DataType.BFLOAT_16,
    decoder_text_encoder=DataType.BFLOAT_16,
    decoder_vqgan=DataType.BFLOAT_16,
    lora=DataType.FLOAT_32,
    embedding=DataType.FLOAT_32,
)

print("Loading model with LoRA...")
loader = Flux2LoRAModelLoader()
model = loader.load(
    model_type=ModelType.FLUX_2,
    model_names=model_names,
    weight_dtypes=weight_dtypes,
    quantization=QuantizationConfig.default_values(),
)

train_device = torch.device("cuda")
temp_device = torch.device("cpu")

# Keep model on CPU, sampler will move components as needed
model.train_device = train_device
model.temp_device = temp_device
model.eval()

print("Creating sampler...")
sampler = Flux2Sampler(
    train_device=train_device,
    temp_device=temp_device,
    model=model,
    model_type=ModelType.FLUX_2,
)

for i, prompt in enumerate(PROMPTS):
    print(f"\nGenerating {i+1}/5: {prompt[:50]}...")

    sample_config = SampleConfig.default_values()
    sample_config.prompt = prompt
    sample_config.negative_prompt = ""
    sample_config.width = 512
    sample_config.height = 512
    sample_config.seed = 42 + i
    sample_config.random_seed = False
    sample_config.diffusion_steps = 30
    sample_config.cfg_scale = 3.5
    sample_config.noise_scheduler = NoiseScheduler.EULER

    out_path = os.path.join(OUTPUT_DIR, f"ot_sample_{i+1}.png")
    sampler.sample(
        sample_config=sample_config,
        destination=out_path,
        image_format=ImageFormat.PNG,
    )
    print(f"Saved: {out_path}")

print(f"\nDone! Images in {OUTPUT_DIR}")
