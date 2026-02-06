import io
import time
import sys

# Mock StdoutCapture from TrainerService
class StdoutCapture(io.StringIO):
    def __init__(self, callback, original_stdout):
        super().__init__()
        self.callback = callback
        self.original_stdout = original_stdout
        self.buffer_line = ""

    def write(self, text):
        # Buffer until we get a complete line
        self.buffer_line += text
        
        # Handle both newline and carriage return (for tqdm progress bars)
        while '\n' in self.buffer_line or '\r' in self.buffer_line:
            n_pos = self.buffer_line.find('\n')
            r_pos = self.buffer_line.find('\r')
            
            if n_pos != -1 and (r_pos == -1 or n_pos < r_pos):
                pos = n_pos
                delimiter_len = 1
            else:
                pos = r_pos
                delimiter_len = 1
            
            line = self.buffer_line[:pos]
            self.buffer_line = self.buffer_line[pos + delimiter_len:]
            
            if line.strip():  # Only send non-empty lines
                self.callback(line)

        return len(text)

    def flush(self):
        pass

def callback(line):
    print(f"CAPTURED: '{line}'")
    is_progress = ('%' in line and '|' in line) or ('Downloading' in line and '%' in line)
    print(f"  -> Is Progress: {is_progress}")

# Simulate tqdm output
chunks = [
    "Downloading... 0%", 
    "\r", 
    "Downloading... 10%", 
    "\r", 
    "Downloading... 20%",
    "\r",
    "100%|██████████| 10/10 [00:01<00:00,  8.00it/s]",
    "\n"
]

capture = StdoutCapture(callback, None)

print("--- Simulating Output ---")
for chunk in chunks:
    capture.write(chunk)
