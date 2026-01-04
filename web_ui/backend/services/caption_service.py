import os
import torch
import math
import warnings
from PIL import Image
import traceback
from typing import Generator, Optional, Tuple, List, Dict, Any
from transformers import BitsAndBytesConfig, AutoProcessor, AutoModelForVision2Seq
from qwen_vl_utils import process_vision_info

# Try to import Qwen specific classes if available, otherwise rely on AutoModel
# Try to import Qwen specific classes if available, otherwise rely on AutoModel
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None

try:
    from transformers import Qwen2VLForConditionalGeneration
except ImportError:
    Qwen2VLForConditionalGeneration = None

class CaptionService:
    _instance = None

    def __init__(self):
        self.model = None
        self.processor = None
        self.current_model_id = None
        self.current_quant = None
        self.should_abort = False
        
        # Defaults
        self.default_model_id = "Qwen/Qwen2.5-VL-7B-Instruct" 
        self.image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')
        self.video_extensions = ('.mp4', '.mov', '.avi', '.webm', '.mkv', ".gif", ".flv")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = CaptionService()
        return cls._instance

    def get_state(self) -> Dict[str, Any]:
        """Returns current model state info."""
        if self.model is None:
            return {
                "loaded": False,
                "model_id": None,
                "device": "N/A",
                "vram_used": "N/A",
                "dtype": "N/A"
            }
        
        vram_used = "N/A"
        if torch.cuda.is_available():
            vram_used = f"{torch.cuda.memory_allocated() / 1e9:.2f} GB"
            
        dtype = str(next(self.model.parameters()).dtype)
        
        return {
            "loaded": True,
            "model_id": self.current_model_id,
            "device": "CUDA" if torch.cuda.is_available() else "CPU",
            "vram_used": vram_used,
            "dtype": dtype
        }

    def unload_model(self):
        """Unloads the current model and clears cache."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.processor is not None:
            del self.processor
            self.processor = None
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.current_model_id = None
        self.current_quant = None

    def load_model(self, model_id: str, quantization: str = "8-bit", attn_impl: str = "flash_attention_2"):
        """Loads the specified Qwen-VL model."""
        if self.model is not None:
            self.unload_model()

        print(f"[CaptionService] Loading model: {model_id} (Quant: {quantization}, Attn: {attn_impl})")
        
        # Prefer bfloat16 for stability (Ampere+), fallback to float16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            default_dtype = torch.bfloat16
        else:
            default_dtype = torch.float16
            print("[CaptionService] WARNING: bfloat16 not supported, using float16. This may cause instability with some models.")

        kwargs = {
            "device_map": "auto",
            "torch_dtype": default_dtype,
            "attn_implementation": attn_impl,
            "trust_remote_code": True 
        }

        # Quantization Config
        if quantization == "8-bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif quantization == "4-bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, 
                bnb_4bit_compute_dtype=torch.float32,
                bnb_4bit_quant_type="nf4"
            )
        
        # Load Model
        try:
            # Select specific class based on Model ID pattern
            if "Qwen2.5-VL" in model_id and Qwen2_5_VLForConditionalGeneration:
                model_cls = Qwen2_5_VLForConditionalGeneration
            elif "Qwen2-VL" in model_id and Qwen2VLForConditionalGeneration:
                 model_cls = Qwen2VLForConditionalGeneration
            else:
                model_cls = AutoModelForVision2Seq

            print(f"[CaptionService] Using model class: {model_cls.__name__}")
            self.model = model_cls.from_pretrained(model_id, **kwargs)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[CaptionService] Primary load failed: {e}")

            # Fallback to eager attention if FA2 fails
            if attn_impl == "flash_attention_2":
                print(f"[CaptionService] FA2 failed ({e}), falling back to eager attention.")
                kwargs["attn_implementation"] = "eager"
                self.model = model_cls.from_pretrained(model_id, **kwargs)
            else:
                raise e

        # Load Processor
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        
        self.current_model_id = model_id
        self.current_quant = quantization
        
        return self.get_state()

            
    def generate_caption(self, media_path: str, prompt: str, max_tokens: int = 256, resolution_mode: str = "auto") -> str:
        if not self.model or not self.processor:
            raise RuntimeError("Model needed")
        
        print(f"[DEBUG] Generating caption for {media_path}...")
        
        is_video = media_path.lower().endswith(self.video_extensions)
        
        try:
            # Helper function for smart_resize
            def smart_resize(height, width, factor, min_pixels, max_pixels):
                current_pixels = height * width
                if min_pixels <= current_pixels <= max_pixels:
                    return height, width
                target_pixels = max_pixels
                scale_factor = math.sqrt(target_pixels / current_pixels)
                new_height = int(height * scale_factor)
                new_width = int(width * scale_factor)
                new_height = (new_height // factor) * factor
                new_width = (new_width // factor) * factor
                if new_height < factor: new_height = factor
                if new_width < factor: new_width = factor
                return new_height, new_width

            # Resolution settings
            if resolution_mode == "auto":
                min_pixels = 3136
                max_pixels = 1003520
            elif resolution_mode == "fast":
                min_pixels = 3136 
                max_pixels = 501760 
            else:  # auto_high / high
                min_pixels = 3136 
                max_pixels = 2007040

            if is_video:
                # Video handling - pass path directly to qwen_vl_utils
                print(f"[DEBUG] Processing video: {media_path}")
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": media_path,
                                "max_pixels": max_pixels,
                                "fps": 1.0,  # Sample 1 frame per second
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
            else:
                # Image handling
                image = Image.open(media_path)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                
                width, height = image.size
                resized_height, resized_width = smart_resize(height, width, factor=28, min_pixels=min_pixels, max_pixels=max_pixels)
                
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image,
                                "resized_height": resized_height,
                                "resized_width": resized_width,
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
            
            print(f"[DEBUG] Processing vision info for {media_path}...")
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            print(f"[DEBUG] Moving inputs to device {self.model.device}...")
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)

            # Ensure inputs match model dtype (critical for FP16/BF16 inference)
            if hasattr(self.model, "dtype"):
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
                        inputs[k] = v.to(dtype=self.model.dtype)

            print(f"[DEBUG] Running model.generate...")
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=max_tokens,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
            
            print(f"[DEBUG] Decoding output...")
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            caption = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            print(f"[DEBUG] Generated: {caption[:50]}...")
            return caption

        except Exception as e:
            print(f"[ERROR] Generate failed: {e}")
            traceback.print_exc()
            raise e

    def run_batch_job(self, data: dict) -> Generator[Dict[str, Any], None, None]:
        folder_path = data.get('folder_path')
        prompt = data.get('prompt')
        skip_existing = data.get('skip_existing', False)
        max_tokens = data.get('max_tokens', 256)
        resolution_mode = data.get('resolution_mode', 'auto')

        print(f"[DEBUG] Starting batch job in {folder_path}")

        # Filter out hidden files and hidden directories (if we were recursive)
        # Using listdir so just checking filename
        files = [
            f for f in os.listdir(folder_path) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')) 
            and not f.startswith('.')
        ]
        files.sort()
        
        total = len(files)
        processed = 0
        skipped = 0
        failed = 0

        for i, filename in enumerate(files):
            if self.stop_signal: # Using self.stop_signal as provided in the edit
                print("[DEBUG] Stop signal received")
                break
                
            full_path = os.path.join(folder_path, filename)
            txt_path = os.path.splitext(full_path)[0] + ".txt"
            
            yield {
                "active": True,
                "current_file": filename,
                "progress": processed / total if total > 0 else 0,
                "total": total,
                "processed": processed,
                "skipped": skipped,
                "failed": failed
            }

            if skip_existing and os.path.exists(txt_path):
                skipped += 1
                continue
                
            try:
                print(f"[DEBUG] Batch processing {filename}...")
                caption = self.generate_caption(full_path, prompt, max_tokens, resolution_mode)
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(caption)
                processed += 1
                yield {"last_caption": caption}
            except Exception as e:
                print(f"[ERROR] Failed to caption {filename}: {e}")
                failed += 1
                
        yield {
            "active": False,
            "progress": 1.0,
            "current_file": None,
            "processed": processed,
            "skipped": skipped,
            "failed": failed
        }
        print("[DEBUG] Batch job finished")

    def process_folder(
        self,
        folder_path: str,
        prompt: str,
        skip_existing: bool,
        max_tokens: int,
        resolution_mode: str
    ) -> Generator[Dict[str, Any], None, None]:
        """Yields progress updates while processing a folder."""
        self.should_abort = False
        
        media_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                ext = os.path.splitext(file)[-1].lower()
                if ext in self.image_extensions or ext in self.video_extensions:
                    media_files.append(os.path.join(root, file))
                    
        total = len(media_files)
        if total == 0:
            yield {"type": "error", "message": "No media found"}
            return

        skipped = 0
        failed = 0
        processed = 0
        
        for idx, media_path in enumerate(media_files):
            if self.should_abort:
                yield {"type": "aborted", "processed": processed, "total": total}
                return

            rel_path = os.path.relpath(media_path, folder_path)
            txt_path = os.path.splitext(media_path)[0] + ".txt"
            
            if skip_existing and os.path.exists(txt_path):
                skipped += 1
                yield {
                    "type": "skipped", 
                    "filename": rel_path, 
                    "progress": (idx + 1) / total,
                    "stats": {"processed": processed, "skipped": skipped, "failed": failed}
                }
                continue
                
            try:
                # Note: The generate_caption method was replaced above.
                # This call will now use the new image-only generate_caption.
                # If video files are present, they will likely cause an error
                # because the new generate_caption expects an image file.
                caption = self.generate_caption(media_path, prompt, max_tokens, resolution_mode)
                
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(caption)
                    
                processed += 1
                yield {
                    "type": "success",
                    "filename": rel_path,
                    "caption": caption,
                    "progress": (idx + 1) / total,
                    "stats": {"processed": processed, "skipped": skipped, "failed": failed}
                }
                
            except Exception as e:
                failed += 1
                print(f"[CaptionService] Error processing {rel_path}: {e}")
                yield {
                    "type": "error_file",
                    "filename": rel_path,
                    "error": str(e),
                    "progress": (idx + 1) / total,
                    "stats": {"processed": processed, "skipped": skipped, "failed": failed}
                }

        yield {"type": "complete", "stats": {"processed": processed, "skipped": skipped, "failed": failed}}

    def stop_processing(self):
        self.should_abort = True

def get_caption_service():
    return CaptionService.get_instance()
