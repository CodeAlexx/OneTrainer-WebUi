
import os
import sys
import torch
import unittest
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.model.Kandinsky5Model import Kandinsky5Model, Kandinsky5Transformer
from modules.modelSetup.Kandinsky5LoRASetup import Kandinsky5LoRASetup
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.TrainProgress import TrainProgress

class TestKandinsky5Setup(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cpu')
        self.config = TrainConfig.default_values()
        # Mock model type logic
        self.model_type = ModelType.KANDINSKY_5 

    def test_model_initialization(self):
        print("Testing Kandinsky5Model initialization...")
        model = Kandinsky5Model(self.model_type)
        self.assertIsInstance(model, Kandinsky5Model)
        
        # Manually attach transformer since loader is skipped
        model.transformer = Kandinsky5Transformer()
        print("Transformer attached.")
        
        # Verify components dictionary
        comps = model.get_components()
        self.assertIn("transformer", comps)
        self.assertIn("text_encoder_qwen", comps)

    def test_lora_setup(self):
        print("Testing Kandinsky5LoRASetup...")
        model = Kandinsky5Model(self.model_type)
        model.transformer = Kandinsky5Transformer()
        
        setup = Kandinsky5LoRASetup(self.device, self.device, debug_mode=True)
        
        # calling setup_model
        setup.setup_model(model, self.config)
        
        # Verify transformer is unfrozen (simulated LoRA)
        self.assertTrue(model.transformer.dummy_param.requires_grad)
        
        # Verify creating parameters
        params = setup.create_parameters(model, self.config)
        files = list(params.parameters())
        self.assertGreater(len(files), 0)
        print("Parameters created successfully.")

    def test_predict_flow(self):
        print("Testing predict flow...")
        model = Kandinsky5Model(self.model_type)
        model.transformer = Kandinsky5Transformer()
        from modules.util.enum.DataType import DataType
        model.transformer_train_dtype = DataType.FLOAT_32 # Use enum directly
        
        setup = Kandinsky5LoRASetup(self.device, self.device, debug_mode=True)
        
        # Mock batch
        batch = {
            'latent_image': torch.randn(1, 16, 1, 64, 64), # B, C, F, H, W
            'text_embed_qwen': torch.randn(1, 10, 1024),
            'text_embed_clip': torch.randn(1, 10, 768),
            'pooled_text_embed_clip': torch.randn(1, 768),
            'loss_weight': 1.0
        }
        
        train_progress = TrainProgress()
        
        # Run predict
        result = setup.predict(
            model=model, batch=batch, config=self.config, train_progress=train_progress
        )
        
        self.assertIn('model_pred', result)
        self.assertIn('target', result)
        
        # Calculate loss
        loss = setup.calculate_loss(model, batch, result, self.config)
        print(f"Loss calculated: {loss.item()}")
        self.assertIsInstance(loss, torch.Tensor)

if __name__ == '__main__':
    unittest.main()
