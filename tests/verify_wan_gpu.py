
import sys
import os
import torch
import torch.nn as nn

# Add project root to path
sys.path.append(os.getcwd())

from modules.model.WanModel import WanModel
from modules.modelSampler.WanSampler import WanSampler
from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.ModelType import ModelType

def verify_wan_gpu():
    print("Verifying Wan Pipeline on GPU...")
    if not torch.cuda.is_available():
        print("Error: CUDA not available!")
        return

    device = torch.device('cuda')
    
    # 1. Initialize Model
    # We use WAN_I2V to test more paths (CLIP conditioning logic if possible, though mock might skip)
    model = WanModel(ModelType.WAN_T2V)
    model.text_len = 16
    
    # Use Float16 or BFloat16 for GPU test
    class MockDataType:
        def torch_dtype(self):
            return torch.float16
    model.transformer_train_dtype = MockDataType()
    model.transformer_autocast_context = torch.autocast(device_type='cuda', dtype=torch.float16)

    # 2. Mock Components (On GPU)
    class MockTokenizer:
        def __call__(self, text, return_mask=True, add_special_tokens=True):
            batch_size = len(text)
            length = 16
            tokens = torch.randint(0, 100, (batch_size, length), dtype=torch.long)
            mask = torch.ones((batch_size, length), dtype=torch.long)
            return tokens, mask

    class MockTextEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.device = torch.device('cuda') # Default
            self.dtype = torch.float16

        def forward(self, input_ids, attention_mask):
            batch_size, length = input_ids.shape
            dim = 64 
            # Check device input
            assert input_ids.device.type == 'cuda'
            return torch.randn(batch_size, length, dim, device=input_ids.device, dtype=self.dtype)
        
        def to(self, device):
            self.device = device
            return super().to(device)

    class MockVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('mean', torch.zeros(16))
            self.register_buffer('std', torch.ones(16))
            self.to(device)

        @property
        def scale(self):
            return [self.mean, 1.0 / self.std]

        def encode(self, video):
            B, C, F, H, W = video.shape
            latent_channels = 16
            return [torch.randn(latent_channels, (F-1)//4+1, H//8, W//8, device=video.device, dtype=video.dtype) for _ in range(B)]

        def decode(self, latents):
            B, C, F, H, W = latents.shape
            out_C = 3
            out_F = (F-1)*4 + 1
            out_H = H*8
            out_W = W*8
            return [torch.randn(out_C, out_F, out_H, out_W, device=latents.device, dtype=latents.dtype) for _ in range(B)]

    model.tokenizer = MockTokenizer()
    model.text_encoder = MockTextEncoder().to(device)
    model.vae = MockVAE().to(device)
    
    class MockTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type('Config', (), {'guidance_embeds': False})()

        def forward(self, x, t, context, seq_len, y=None, clip_fea=None, parasitic_fea=None):
            # Verify inputs are on GPU
            assert x.device.type == 'cuda'
            assert context.device.type == 'cuda'
            assert t.device.type == 'cuda'
            return torch.randn_like(x)

    model.transformer = MockTransformer().to(device)
    
    # 3. Initialize Sampler
    sampler = WanSampler(
        train_device=device,
        temp_device=device,
        model=model,
        model_type=ModelType.WAN_T2V
    )
    
    # 4. Scheduler
    class MockScheduler:
        def __init__(self):
            self.timesteps = torch.tensor([999, 500, 0], device=device)
        
        def set_timesteps(self, num_inference_steps, device):
            self.timesteps = torch.linspace(999, 0, num_inference_steps, device=device).long()
            
        def step(self, model_output, timestep, sample, return_dict=False):
            return (sample - model_output,) 
            
    # SampleConfig expects a list of tuples in init, or we use default_values()
    sample_config = SampleConfig.default_values()
    sample_config.prompt = "A test video on GPU"
    sample_config.negative_prompt = ""
    sample_config.height = 128
    sample_config.width = 128
    sample_config.frames = 17
    sample_config.seed = 42
    sample_config.random_seed = False
    sample_config.diffusion_steps = 5
    sample_config.cfg_scale = 4.0
    sample_config.cfg_scale = 4.0
    mock_scheduler = MockScheduler()
    sample_config.noise_scheduler = mock_scheduler
    model.noise_scheduler = mock_scheduler
    # sample_config does not inherently have these text_encoder_skip fields in default values? 
    # Yes, it does (lines 62-66 in SampleConfig.py view)
    sample_config.text_encoder_1_layer_skip = 0
    sample_config.text_encoder_2_layer_skip = 0
    sample_config.transformer_attention_mask = False
    from modules.util.enum.VideoFormat import VideoFormat
    sample_config.video_format = VideoFormat.MP4
    
    # 5. Run Sample
    print("Running sample() on GPU...")
    model.to(device) # Ensure model is on GPU
    
    def on_sample(output):
        print(f"Sample generated! Type: {output.file_type}")
        if hasattr(output.data, 'shape'):
             print(f"Data shape: {output.data.shape}")
             # Check if data is on CPU (sampler should move output to CPU)
             print(f"Data device: {output.data.device}")
        
    sampler.sample(
        sample_config=sample_config,
        destination="test_output_gpu",
        on_sample=on_sample,
        video_format=sample_config.video_format
    )
    
    print("GPU verification finished successfully.")

if __name__ == "__main__":
    verify_wan_gpu()
