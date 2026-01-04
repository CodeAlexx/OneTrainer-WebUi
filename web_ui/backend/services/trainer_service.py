"""
Trainer service singleton for managing OneTrainer training state.
"""
import asyncio
import threading
import time
import sys
import io
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
import json


class StdoutCapture(io.StringIO):
    """Captures stdout and forwards to a callback while still printing."""

    def __init__(self, callback, original_stdout):
        super().__init__()
        self.callback = callback
        self.original_stdout = original_stdout
        self.buffer_line = ""

    def write(self, text):
        # Write to original stdout
        if self.original_stdout:
            self.original_stdout.write(text)
            self.original_stdout.flush()

        # Buffer until we get a complete line
        self.buffer_line += text
        
        # Handle both newline and carriage return (for tqdm progress bars)
        while '\n' in self.buffer_line or '\r' in self.buffer_line:
            # Find the first delimiter
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
        if self.original_stdout:
            self.original_stdout.flush()
            
    def isatty(self):
        return True
    
    def fileno(self):
        if hasattr(self.original_stdout, 'fileno'):
            try:
                return self.original_stdout.fileno()
            except Exception:
                return 1
        return 1
    
    @property
    def encoding(self):
        return getattr(self.original_stdout, 'encoding', 'utf-8')

from modules.trainer.GenericTrainer import GenericTrainer
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.config.TrainConfig import TrainConfig
from modules.util.TrainProgress import TrainProgress
from modules.modelSampler.BaseModelSampler import ModelSamplerOutput


@dataclass
class TrainingState:
    """Current training state."""
    is_training: bool = False
    status: str = "idle"
    progress: Optional[Dict[str, Any]] = None
    max_step: int = 0
    max_epoch: int = 0
    error: Optional[str] = None


class TrainerService:
    """
    Singleton service for managing OneTrainer training operations.

    Wraps GenericTrainer, TrainCommands, and TrainCallbacks to provide
    a centralized interface for training state management.
    """

    _instance: Optional['TrainerService'] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize once
        if hasattr(self, '_initialized') and self._initialized:
            return

        self._initialized = True
        self._trainer: Optional[GenericTrainer] = None
        self._commands: Optional[TrainCommands] = None
        self._callbacks: Optional[TrainCallbacks] = None
        self._config: Optional[TrainConfig] = None
        self._training_thread: Optional[threading.Thread] = None
        self._state = TrainingState()
        self._state_lock = threading.Lock()

        # WebSocket connections for broadcasting updates
        self._ws_connections: set = set()
        self._ws_lock = threading.Lock()

        # Event loop reference for async broadcasting from threads
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Time tracking for ETA and speed
        self._training_start_time: Optional[float] = None
        self._last_step_time: Optional[float] = None
        self._last_step: int = 0
        
        # Progress broadcasting rate limit
        self._last_progress_broadcast_time: float = 0

    @classmethod
    def get_instance(cls) -> 'TrainerService':
        """Get the singleton instance."""
        return cls()

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        """Set the event loop reference for async broadcasting from threads."""
        self._event_loop = loop

    def register_websocket(self, websocket):
        """Register a WebSocket connection for updates."""
        with self._ws_lock:
            self._ws_connections.add(websocket)

    def unregister_websocket(self, websocket):
        """Unregister a WebSocket connection."""
        with self._ws_lock:
            self._ws_connections.discard(websocket)

    async def broadcast_update(self, message: Dict[str, Any]):
        """Broadcast update to all connected WebSocket clients."""
        with self._ws_lock:
            connections = list(self._ws_connections)

        # Send to all connections
        for ws in connections:
            try:
                await ws.send_json(message)
            except Exception:
                # Remove dead connections
                with self._ws_lock:
                    self._ws_connections.discard(ws)

    def _update_state(self, **kwargs):
        """Thread-safe state update with WebSocket broadcast."""
        with self._state_lock:
            for key, value in kwargs.items():
                if hasattr(self._state, key):
                    setattr(self._state, key, value)

            # Prepare broadcast message
            state_dict = asdict(self._state)

        # Broadcast update asynchronously (don't block training thread)
        self._schedule_broadcast({
            "type": "training_state",
            "data": state_dict
        })

    def _schedule_broadcast(self, message: Dict[str, Any]):
        """Schedule a broadcast to run on the event loop from any thread."""
        if self._event_loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                self.broadcast_update(message),
                self._event_loop
            )
        except Exception as e:
            print(f"Failed to schedule broadcast: {e}")

    def _on_update_train_progress(
            self,
            progress: TrainProgress,
            max_step: int,
            max_epoch: int,
            loss: float | None = None,
            smooth_loss: float | None = None,
    ):
        """Callback for training progress updates."""
        current_time = time.time()
        
        # Initialize training start time on first progress
        if self._training_start_time is None:
            self._training_start_time = current_time
            self._last_step_time = current_time
            self._last_step = progress.global_step
        
        # Calculate speed (steps per second)
        elapsed_since_last = current_time - (self._last_step_time or current_time)
        steps_since_last = progress.global_step - self._last_step
        
        if elapsed_since_last > 0 and steps_since_last > 0:
            steps_per_second = steps_since_last / elapsed_since_last
            seconds_per_step = elapsed_since_last / steps_since_last
        else:
            steps_per_second = 0
            seconds_per_step = 0
        
        self._last_step_time = current_time
        self._last_step = progress.global_step
        
        # Calculate elapsed and remaining time
        total_elapsed = current_time - self._training_start_time
        total_training_steps = max_epoch * max_step
        remaining_steps = total_training_steps - progress.global_step
        
        if steps_per_second > 0:
            remaining_seconds = remaining_steps / steps_per_second
        else:
            remaining_seconds = 0
        
        # Format times as HH:MM:SS
        def format_time(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        
        elapsed_str = format_time(total_elapsed)
        remaining_str = format_time(remaining_seconds)
        
        progress_dict = {
            "epoch": progress.epoch,
            "epoch_step": progress.epoch_step,
            "epoch_sample": progress.epoch_sample,
            "global_step": progress.global_step,
            "loss": loss,
            "smooth_loss": smooth_loss,
        }

        self._update_state(
            progress=progress_dict,
            max_step=max_step,
            max_epoch=max_epoch
        )

        # Broadcast detailed progress update for dashboard
        self._schedule_broadcast({
            "type": "progress",
            "data": {
                "current_epoch": progress.epoch,
                "total_epochs": max_epoch,
                "current_step": progress.global_step,
                "total_steps": total_training_steps,
                "epoch_step": progress.epoch_step,
                "epoch_length": max_step,
                "loss": loss,
                "smooth_loss": smooth_loss,
                "elapsed_time": elapsed_str,
                "remaining_time": remaining_str,
                "samples_per_second": seconds_per_step,
            }
        })

    def _on_update_status(self, status: str):
        """Callback for status updates."""
        # Print status so it appears in console (stdout capture will broadcast it)
        print(status)
        self._update_state(status=status)

    def _on_sample_default(self, output: ModelSamplerOutput):
        """Callback for default sample generation."""
        self._schedule_broadcast({
            "type": "sample_default",
            "data": {
                "sample_count": len(output.samples) if output.samples else 0
            }
        })

    def _on_update_sample_default_progress(self, current: int, total: int):
        """Callback for default sample progress."""
        self._schedule_broadcast({
            "type": "sampling",
            "data": {"current": current, "total": total}
        })

    def _on_sample_custom(self, output: ModelSamplerOutput):
        """Callback for custom sample generation."""
        self._schedule_broadcast({
            "type": "sample_custom",
            "data": {
                "sample_count": len(output.samples) if output.samples else 0
            }
        })

    def _on_update_sample_custom_progress(self, current: int, total: int):
        """Callback for custom sample progress."""
        self._schedule_broadcast({
            "type": "sampling",
            "data": {"current": current, "total": total}
        })

    def _on_command(self, commands: TrainCommands):
        """Callback when command is received."""
        # Handle commands if needed
        pass

    def _resolve_config_paths(self, config: TrainConfig) -> TrainConfig:
        """
        Resolve relative paths in config to absolute paths.
        
        This ensures paths like 'training_concepts/Liza.json' work regardless
        of the working directory when training starts.
        """
        import os
        import re
        
        # Get OneTrainer root directory
        onetrainer_root = Path(os.environ.get(
            "ONETRAINER_ROOT", 
            Path(__file__).parent.parent.parent.parent
        ))
        
        def is_huggingface_id(path_str: str) -> bool:
            """Check if path looks like a HuggingFace model ID (org/model format)."""
            # HuggingFace IDs: single slash, no file extension, alphanumeric with dashes
            if '/' not in path_str:
                return False
            if path_str.startswith('/') or path_str.startswith('.'):
                return False
            # Check for file extensions (local files)
            if any(path_str.endswith(ext) for ext in ['.json', '.safetensors', '.ckpt', '.pt', '.bin']):
                return False
            # HF IDs have format: org/model or user/model (one slash, two parts)
            parts = path_str.split('/')
            if len(parts) == 2 and all(re.match(r'^[\w\-\.]+$', p) for p in parts):
                return True
            return False
        
        def resolve_path(path_str: str, field_name: str) -> str:
            if not path_str:
                return path_str
            
            # Skip HuggingFace model IDs for model-related fields
            if field_name in ['base_model_name', 'lora_model_name'] and is_huggingface_id(path_str):
                return path_str
            
            path = Path(path_str)
            if path.is_absolute():
                return path_str
            
            # Only resolve if it looks like a local path (exists or has typical path structure)
            resolved = onetrainer_root / path
            if resolved.exists():
                return str(resolved.absolute())
            
            # For non-model fields, resolve anyway for better error messages
            if field_name not in ['base_model_name', 'lora_model_name']:
                return str(resolved.absolute())
            
            # For model fields, return original if not found locally
            return path_str
        
        # Resolve common path fields
        path_fields = [
            'concept_file_name',
            'sample_definition_file_name',
            'base_model_name',
            'output_model_destination',
            'workspace_dir',
            'cache_dir',
            'debug_dir',
            'lora_model_name',
        ]
        
        for field in path_fields:
            if hasattr(config, field):
                current_value = getattr(config, field)
                if current_value and isinstance(current_value, str):
                    resolved = resolve_path(current_value, field)
                    setattr(config, field, resolved)
        
        return config

    def initialize_trainer(self, config: TrainConfig) -> bool:
        """
        Initialize the trainer with a configuration.

        Args:
            config: Training configuration

        Returns:
            True if successful, False otherwise
        """
        try:
            # Resolve relative paths to absolute paths
            config = self._resolve_config_paths(config)
            
            # Create callbacks with our handlers
            self._callbacks = TrainCallbacks(
                on_update_train_progress=self._on_update_train_progress,
                on_update_status=self._on_update_status,
                on_sample_default=self._on_sample_default,
                on_update_sample_default_progress=self._on_update_sample_default_progress,
                on_sample_custom=self._on_sample_custom,
                on_update_sample_custom_progress=self._on_update_sample_custom_progress,
            )

            # Create commands
            self._commands = TrainCommands(on_command=self._on_command)

            # Create trainer
            self._trainer = GenericTrainer(config, self._callbacks, self._commands)
            self._config = config

            self._update_state(
                status="initialized",
                error=None
            )

            return True

        except Exception as e:
            self._update_state(
                status="error",
                error=str(e)
            )
            return False

    def _on_stdout_line(self, line: str):
        """Handle captured stdout line."""
        # Filter out some noisy lines
        if not line.strip():
            return
            
        # Check for tqdm progress bars (contain % and | or typical download patterns)
        is_progress = ('%' in line and '|' in line) or ('Downloading' in line and '%' in line)
        
        if is_progress:
            # Rate limit status updates from progress bars to ~2 per second
            current_time = time.time()
            if current_time - self._last_progress_broadcast_time > 0.5:
                # Update status directly without printing to avoid loop
                self._update_state(status=line.strip())
                self._last_progress_broadcast_time = current_time
            return

        if line.startswith('[') and 'it/s]' in line:
            return  # Skip tqdm progress bars that didn't catch above

        # Broadcast as log message
        self._schedule_broadcast({
            "type": "log",
            "data": {"level": "info", "message": line}
        })

    def _run_training(self):
        """Run training in background thread."""
        # Capture stdout to forward to WebSocket
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        stdout_capture = StdoutCapture(self._on_stdout_line, original_stdout)
        stderr_capture = StdoutCapture(self._on_stdout_line, original_stderr) # Capture stderr for tqdm
        
        # Save current directory and change to OneTrainer root
        # This ensures relative paths in model loaders work correctly
        import os
        original_cwd = os.getcwd()
        onetrainer_root = Path(os.environ.get(
            "ONETRAINER_ROOT", 
            Path(__file__).parent.parent.parent.parent
        ))
        os.chdir(onetrainer_root)

        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            self._update_state(
                is_training=True,
                status="starting",
                error=None
            )

            # Broadcast a log message so the console shows something immediately
            self._schedule_broadcast({
                "type": "log",
                "data": {"level": "info", "message": "Training starting..."}
            })

            if self._trainer is None:
                raise ValueError("Trainer not initialized")

            # Start training (sets up model, data loader, etc.)
            self._trainer.start()

            # Run the actual training loop
            self._trainer.train()

            # Save final model and cleanup
            self._trainer.end()
            
            # Check if training was stopped by user or completed naturally
            was_stopped = self._commands and self._commands.get_stop_command()
            
            if was_stopped:
                # Broadcast stopped log
                self._schedule_broadcast({
                    "type": "log",
                    "data": {"level": "info", "message": "Training stopped by user."}
                })
                self._update_state(
                    is_training=False,
                    status="stopped"
                )
            else:
                # Broadcast completion log
                self._schedule_broadcast({
                    "type": "log",
                    "data": {"level": "info", "message": "Training completed successfully."}
                })
                self._update_state(
                    is_training=False,
                    status="completed"
                )

        except Exception as e:
            # Broadcast error log
            self._schedule_broadcast({
                "type": "log",
                "data": {"level": "error", "message": f"Training error: {str(e)}"}
            })

            self._update_state(
                is_training=False,
                status="error",
                error=str(e)
            )

        finally:
            # Always restore stdout and working directory
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            os.chdir(original_cwd)

    def start_training(self) -> bool:
        """
        Start training in background thread.

        Returns:
            True if training started, False if already running or error
        """
        with self._state_lock:
            if self._state.is_training:
                return False

            if self._trainer is None:
                self._update_state(
                    status="error",
                    error="Trainer not initialized"
                )
                return False

        # Start training thread
        self._training_thread = threading.Thread(target=self._run_training, daemon=True)
        self._training_thread.start()

        return True

    def stop_training(self) -> bool:
        """
        Request training to stop.

        Returns:
            True if stop command sent, False otherwise
        """
        if self._commands is None:
            return False

        self._commands.stop()
        self._update_state(status="stopping")
        return True

    def sample_default(self) -> bool:
        """
        Request default sample generation.

        Returns:
            True if command sent, False otherwise
        """
        if self._commands is None:
            return False

        self._commands.sample_default()
        return True

    def sample_custom(self, sample_params) -> bool:
        """
        Request custom sample generation.

        Args:
            sample_params: SampleConfig object

        Returns:
            True if command sent, False otherwise
        """
        if self._commands is None:
            return False

        self._commands.sample_custom(sample_params)
        return True

    def backup(self) -> bool:
        """
        Request backup creation.

        Returns:
            True if command sent, False otherwise
        """
        if self._commands is None:
            return False

        self._commands.backup()
        return True

    def save(self) -> bool:
        """
        Request model save.

        Returns:
            True if command sent, False otherwise
        """
        if self._commands is None:
            return False

        self._commands.save()
        return True

    def get_state(self) -> Dict[str, Any]:
        """
        Get current training state.

        Returns:
            Dictionary with current state
        """
        with self._state_lock:
            return asdict(self._state)

    def get_config(self) -> Optional[Dict[str, Any]]:
        """
        Get current training configuration.

        Returns:
            Configuration dictionary or None
        """
        if self._config is None:
            return None

        # Convert config to dict (TrainConfig should have __dict__ or similar)
        try:
            return vars(self._config)
        except Exception:
            return None
    
    def get_concepts(self) -> list:
        """Get concepts from current config."""
        if self._config is None:
            return []
        
        try:
            return list(self._config.concepts) if hasattr(self._config, 'concepts') else []
        except Exception:
            return []
    
    def add_concept(self, concept_data: Dict[str, Any]) -> bool:
        """
        Add a concept to the current config.
        
        Args:
            concept_data: Dictionary with concept properties
            
        Returns:
            True if successful
        """
        if self._config is None:
            return False
        
        try:
            from modules.util.config.ConceptConfig import ConceptConfig
            
            # Create a new ConceptConfig from the data
            concept = ConceptConfig.default_values()
            for key, value in concept_data.items():
                if hasattr(concept, key):
                    setattr(concept, key, value)
            
            # Add to config's concepts list
            if not hasattr(self._config, 'concepts'):
                self._config.concepts = []
            self._config.concepts.append(concept)
            
            return True
        except Exception as e:
            print(f"Failed to add concept: {e}")
            return False
    
    def update_concept(self, index: int, concept_data: Dict[str, Any]) -> bool:
        """
        Update a concept in the current config.
        
        Args:
            index: Index of concept to update
            concept_data: Dictionary with updated properties
            
        Returns:
            True if successful
        """
        if self._config is None:
            return False
        
        try:
            concepts = getattr(self._config, 'concepts', [])
            if index < 0 or index >= len(concepts):
                return False
            
            # Update properties
            concept = concepts[index]
            for key, value in concept_data.items():
                if hasattr(concept, key):
                    setattr(concept, key, value)
            
            return True
        except Exception as e:
            print(f"Failed to update concept: {e}")
            return False
    
    def delete_concept(self, index: int) -> bool:
        """
        Delete a concept from the current config.
        
        Args:
            index: Index of concept to delete
            
        Returns:
            True if successful
        """
        if self._config is None:
            return False
        
        try:
            concepts = getattr(self._config, 'concepts', [])
            if index < 0 or index >= len(concepts):
                return False
            
            concepts.pop(index)
            return True
        except Exception as e:
            print(f"Failed to delete concept: {e}")
            return False

    def cleanup(self):
        """Clean up resources."""
        # Stop training if running
        if self._state.is_training:
            self.stop_training()

        # Wait for training thread to finish
        if self._training_thread and self._training_thread.is_alive():
            self._training_thread.join(timeout=5.0)

        # Clear references
        self._trainer = None
        self._commands = None
        self._callbacks = None
        self._config = None

        self._update_state(
            is_training=False,
            status="idle",
            progress=None,
            max_step=0,
            max_epoch=0,
            error=None
        )


# Singleton instance getter
def get_trainer_service() -> TrainerService:
    """Get the trainer service singleton instance."""
    return TrainerService.get_instance()
