import sys
import io
import time
from huggingface_hub import hf_hub_download

# Capture output with fileno delegation
class BytesCapture(io.StringIO):
    def __init__(self, original):
        super().__init__()
        self.original = original

    def write(self, text):
        print(f"RAW WRITE: {repr(text)}")
        return len(text)
    
    def fileno(self):
        return self.original.fileno()
    
    @property
    def encoding(self):
        return getattr(self.original, 'encoding', 'utf-8')

print("--- Starting Download Probe (FILE DELEGATE) ---")
original_stderr = sys.stderr
capture = BytesCapture(original_stderr)
sys.stderr = capture

try:
    # Download a small file
    hf_hub_download(repo_id="bert-base-uncased", filename="config.json")
except Exception as e:
    sys.stderr = original_stderr
    print(f"Error: {e}")
finally:
    sys.stderr = original_stderr
    print("--- Done ---")
