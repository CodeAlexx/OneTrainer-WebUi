import sys
import io
import time
from huggingface_hub import hf_hub_download

# Capture output with all file attributes
class BytesCapture(io.StringIO):
    name = '<stderr>'
    mode = 'w'
    encoding = 'utf-8'

    def write(self, text):
        print(f"RAW WRITE: {repr(text)}")
        return len(text)
    
    def isatty(self):
        return False

    def fileno(self):
        return 2

print("--- Starting Download Probe (FILE ATTRS) ---")
capture = BytesCapture()
original_stderr = sys.stderr
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
