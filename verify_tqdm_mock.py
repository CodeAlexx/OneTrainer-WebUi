import sys
import io
from tqdm import tqdm

class MockTTY(io.StringIO):
    def isatty(self):
        return True
    
    def write(self, s):
        # Print raw repr to original stdout/stderr to prove it tried to write
        sys.__stdout__.write(f"WRITE: {repr(s)}\n")
        return len(s)
    
    def flush(self):
        pass

class MockNoTTY(io.StringIO):
    def isatty(self):
        return False
        
    def write(self, s):
        sys.__stdout__.write(f"WRITE_NOTTY: {repr(s)}\n")
        return len(s)

print("--- Testing MockTTY ---")
sys.stderr = MockTTY()
try:
    for i in tqdm(range(3), desc="TTY"):
        pass
except Exception as e:
    sys.__stdout__.write(f"Error TTY: {e}\n")

print("\n--- Testing MockNoTTY ---")
sys.stderr = MockNoTTY()
try:
    for i in tqdm(range(3), desc="NoTTY"):
        pass
except Exception as e:
    sys.__stdout__.write(f"Error NoTTY: {e}\n")

sys.stderr = sys.__stderr__
