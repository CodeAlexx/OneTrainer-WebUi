import sys
import io
import time
from tqdm import tqdm

class StdoutCapture(io.StringIO):
    def __init__(self, callback, original_stdout):
        super().__init__()
        self.callback = callback
        self.original_stdout = original_stdout
        self.buffer_line = ""

    def write(self, text):
        if self.original_stdout:
            self.original_stdout.write(text)
            self.original_stdout.flush()

        self.buffer_line += text
        while '\n' in self.buffer_line or '\r' in self.buffer_line:
            n_pos = self.buffer_line.find('\n')
            r_pos = self.buffer_line.find('\r')
            if n_pos != -1 and (r_pos == -1 or n_pos < r_pos):
                pos = n_pos
                d_len = 1
            else:
                pos = r_pos
                d_len = 1
            line = self.buffer_line[:pos]
            self.buffer_line = self.buffer_line[pos + d_len:]
            if line.strip():
                self.callback(line)
        return len(text)

    def flush(self):
        if self.original_stdout:
            self.original_stdout.flush()
    
    # Missing isatty might be the issue?
    # def isatty(self): return True 

def on_line(line):
    # Determine if it would be classified as progress
    is_progress = ('%' in line and '|' in line) or ('Downloading' in line and '%' in line)
    print(f"[Captured] (Progress={is_progress}): {line}")

original_stderr = sys.stderr

print("--- Starting capture ---")
capture = StdoutCapture(on_line, original_stderr)
sys.stderr = capture

try:
    # Simulate a loop with tqdm
    # force_tty=True might be needed?
    for i in tqdm(range(10), desc="Downloading"):
        time.sleep(0.1)
except Exception as e:
    sys.stderr = original_stderr
    print(f"Error: {e}")
finally:
    sys.stderr = original_stderr
    print("\n--- Done ---")
