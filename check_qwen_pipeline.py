
try:
    from diffusers import QwenImagePipeline
    print("QwenImagePipeline imported successfully")
    print(f"Has from_single_file: {hasattr(QwenImagePipeline, 'from_single_file')}")
    print(f"Bases: {QwenImagePipeline.__bases__}")
except ImportError:
    print("ImportError: QwenImagePipeline not found")
except Exception as e:
    print(f"Error: {e}")
