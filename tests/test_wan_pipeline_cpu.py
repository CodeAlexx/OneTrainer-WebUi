
import sys
import os
import torch
import torch.nn as nn
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.getcwd())

from modules.model.WanModel import WanModel
from modules.modelSampler.WanSampler import WanSampler
from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.NoiseScheduler import NoiseScheduler

class MockTokenizer:
    def __call__(self, text, return_mask=True, add_special_tokens=True):
        # Return dummy tokens and mask
        # text is list of strings
        batch_size = len(text)
        length = 16
        tokens = torch.randint(0, 100, (batch_size, length), dtype=torch.long)
        mask = torch.ones((batch_size, length), dtype=torch.long)
        return tokens, mask

class MockTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cpu')
        self.dtype = torch.float32

    def forward(self, input_ids, attention_mask):
        # Return dummy embeddings
        batch_size, length = input_ids.shape
        dim = 64 # Match small transformer config
        return torch.randn(batch_size, length, dim)

class MockVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = torch.zeros(16) # 16 latent channels
        self.std = torch.ones(16)
        # self.scale prop is updated in loop

    @property
    def scale(self):
        return [self.mean, 1.0 / self.std]

    def encode(self, video):
        # Return list of latents. WanModel stacks them.
        # video: (B, C, F, H, W)
        # Latents: 16 channels, downscale strides
        B, C, F, H, W = video.shape
        latent_channels = 16
        return [torch.randn(1, latent_channels, (F-1)//4+1, H//8, W//8) for _ in range(B)]

    def decode(self, latents):
        # latents: (B, C, F, H, W)
        B, C, F, H, W = latents.shape
        # output video
        out_C = 3
        out_F = (F-1)*4 + 1
        out_H = H*8
        out_W = W*8
        return [torch.randn(1, out_C, out_F, out_H, out_W) for _ in range(B)]

def test_wan_pipeline_cpu():
    print("Testing Wan Pipeline on CPU...")
    device = torch.device('cpu')
    
    # 1. Initialize Model
    model = WanModel(ModelType.WAN_T2V)
    model.text_len = 16 # Short text for test
    model.transformer_train_dtype = type('DataType', (), {'torch_dtype': lambda: torch.float32})()

    # 2. Mock Components
    model.tokenizer = MockTokenizer()
    model.text_encoder = MockTextEncoder()
    model.vae = MockVAE()
    
    # 3. Create Transformer (Real or Mock?) 
    # Let's try to instantiate the real one with tiny config if possible.
    # But WanModel transformer init signature is complex.
    # Let's use a Mock Transformer that mimics the signature and return shape.
    # forward(self, x, t, context, seq_len, y=None, clip_fea=None, parasitic_fea=None)
    # Output: noise_pred (same shape as x)
    
    class MockTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type('Config', (), {'guidance_embeds': False})()

        def forward(self, x, t, context, seq_len, y=None, clip_fea=None, parasitic_fea=None):
            # x: latents (B, C, F, H, W)
            return torch.randn_like(x)

    model.transformer = MockTransformer()
    
    # 4. Initialize Sampler
    sampler = WanSampler(
        train_device=device,
        temp_device=device,
        model=model,
        model_type=ModelType.WAN_T2V
    )
    
    # 5. Create Sample Config
    # We need a dummy noise scheduler that supports set_timesteps and step
    # OneTrainer passes an Enum usually, but Sampler logic handles it.
    # In my implemented Sampler, I used `noise_scheduler.set_timesteps`.
    # Let's import a real scheduler from diffusers to mock the behavior?
    # Or mock properties.
    
    class MockScheduler:
        def __init__(self):
            self.timesteps = torch.tensor([999, 500, 0])
        
        def set_timesteps(self, num_inference_steps, device):
            self.timesteps = torch.linspace(999, 0, num_inference_steps, device=device).long()
            
        def step(self, model_output, timestep, sample, return_dict=False):
            # Euler step logic: prev_sample = sample - output * dt (roughly)
            return (sample - model_output,) # Tuple return
            
    sample_config = SampleConfig(
        prompt="A test video",
        negative_prompt="",
        height=128,
        width=128,
        frames=17, # (17-1)/4 + 1 = 5 latent frames
        seed=42,
        random_seed=False,
        diffusion_steps=2,
        cfg_scale=4.0,
        noise_scheduler=MockScheduler(),
        text_encoder_1_layer_skip=0,
        text_encoder_2_layer_skip=0,
        transformer_attention_mask=False
    )
    
    # 6. Run Sample
    print("Running sample()...")
    
    def on_sample(output):
        print(f"Sample generated! Type: {output.file_type}")
        print(f"Data shape: {output.data.shape if hasattr(output.data, 'shape') else 'Image'}")
        
    sampler.sample(
        sample_config=sample_config,
        destination="test_output",
        on_sample=on_sample
    )
    
    print("Test finished successfully.")

if __name__ == "__main__":
    test_wan_pipeline_cpu()
