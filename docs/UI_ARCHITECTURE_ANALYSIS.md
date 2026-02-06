# OneTrainer UI Architecture Analysis

**Purpose:** Documentation for migrating from CustomTkinter to FastAPI + Web Frontend
**Date:** 2024-12-24
**Status:** READ-ONLY Analysis - No modifications made

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current Stack](#2-current-stack)
3. [Entry Points](#3-entry-points)
4. [UI Framework & Patterns](#4-ui-framework--patterns)
5. [Window & Tab Structure](#5-window--tab-structure)
6. [UI-to-Core Coupling](#6-ui-to-core-coupling)
7. [State Management](#7-state-management)
8. [Real-Time Updates](#8-real-time-updates)
9. [Configuration System](#9-configuration-system)
10. [Threading Model](#10-threading-model)
11. [File Inventory](#11-file-inventory)
12. [Migration Path](#12-migration-path)

---

## 1. Executive Summary

OneTrainer uses **CustomTkinter** (modern Tkinter wrapper) for its desktop GUI. The architecture is relatively clean with:

- **Clear separation**: UI ↔ Training logic via Commands/Callbacks
- **State binding**: UIState class provides two-way config↔UI sync
- **Threading**: Training runs in background thread, UI stays responsive
- **Callbacks**: Observer pattern for real-time updates

**Migration complexity: MODERATE**
- Well-defined interfaces exist (TrainCommands, TrainCallbacks)
- Configuration system is JSON-based and reusable
- Main work: Replace UI layer, expose existing interfaces via REST/WebSocket

---

## 2. Current Stack

| Layer | Technology | Notes |
|-------|------------|-------|
| UI Framework | CustomTkinter | Modern Tkinter wrapper with themes |
| State Binding | UIState class | Two-way sync via tkinter Variables |
| Threading | Python threading | Background training thread |
| Config | JSON + dataclasses | Versioned with migrations |
| Monitoring | Tensorboard | Loss, samples, metrics |
| IPC | TrainCommands/Callbacks | Command queue + observer pattern |

---

## 3. Entry Points

### Main Training UI
```
scripts/train_ui.py
    └── modules/ui/TrainUI.py (TrainUI class)
        └── ctk.CTk.mainloop()
```

### Other Entry Points
```
scripts/train.py          - CLI training (no UI)
scripts/caption_ui.py     - Captioning tool
scripts/convert_model.py  - Model conversion
```

**Key Insight:** `train.py` proves training works without UI - good for API migration.

---

## 4. UI Framework & Patterns

### CustomTkinter Widgets Used
- `CTk` - Main window
- `CTkToplevel` - Modal windows
- `CTkFrame` / `CTkScrollableFrame` - Containers
- `CTkTabview` - Tab container
- `CTkLabel`, `CTkEntry`, `CTkButton`, `CTkSwitch`, `CTkOptionMenu`
- `CTkProgressBar` - Progress indicators
- `CTkImage` - Image display

### Layout Pattern
All components use **grid layout**:
```python
frame.grid_columnconfigure(0, weight=1)
widget.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
```

### Component Factory
`modules/util/ui/components.py` provides factory functions:
```python
components.label(master, row, col, "Text", tooltip="Help")
components.entry(master, row, col, ui_state, "field_name")
components.switch(master, row, col, ui_state, "bool_field")
components.options_kv(master, row, col, options, ui_state, "enum_field")
components.file_entry(master, row, col, ui_state, "path_field")
```

### Input Validation
- Debounce delay: 1500ms
- Error indication: Red border (#dc3545)
- Auto-revert on validation failure

---

## 5. Window & Tab Structure

### Main Window Layout (TrainUI)
```
┌─────────────────────────────────────────────────┐
│ TopBar                                          │
│ [Config Preset ▼] [Model Type ▼] [Method ▼]    │
├─────────────────────────────────────────────────┤
│ CTkTabview (10 tabs)                            │
│ ┌─────┬───────┬──────┬──────────┬─────────┐    │
│ │Gen. │Model  │Data  │Concepts  │Training │    │
│ ├─────┴───────┴──────┴──────────┴─────────┤    │
│ │                                          │    │
│ │  [Tab Content - ScrollableFrame]         │    │
│ │                                          │    │
│ └──────────────────────────────────────────┘    │
├─────────────────────────────────────────────────┤
│ Bottom Bar                                      │
│ [Progress] [Status] [Export] [TB] [Start]      │
└─────────────────────────────────────────────────┘
```

### Tab List
1. **general** - Workspace, cache, debug, tensorboard, validation
2. **model** - Model selection (SD, SDXL, Flux, etc.)
3. **data** - Dataset configuration
4. **concepts** - Training concepts/datasets (ConfigList)
5. **training** - Optimizer, scheduler, loss, etc.
6. **sampling** - Sample generation config (ConfigList)
7. **backup** - Backup and recovery
8. **tools** - Utility tools
9. **additional embeddings** - Custom embeddings (ConfigList)
10. **cloud** - RunPod, SSH training

### Modal Windows (CTkToplevel)
| Window | Purpose |
|--------|---------|
| ConceptWindow | Concept/dataset editor |
| SampleWindow | Live sampling preview |
| CaptionUI | Image captioning tool |
| VideoToolUI | Video processing |
| OptimizerParamsWindow | Optimizer config |
| SchedulerParamsWindow | LR scheduler config |
| TimestepDistributionWindow | Timestep visualization |
| OffloadingWindow | Memory offloading |
| ProfilingWindow | Performance profiling |

---

## 6. UI-to-Core Coupling

### Command Pattern (UI → Trainer)
**File:** `modules/util/commands/TrainCommands.py`

```python
class TrainCommands:
    def stop(self)                    # Signal training to stop
    def backup(self)                  # Trigger backup
    def save(self)                    # Save checkpoint
    def sample_default(self)          # Generate samples
    def sample_custom(config)         # Custom sample request

    # Trainer polls these:
    def get_stop_command(self) -> bool
    def get_and_reset_backup_command(self) -> bool
    def get_and_reset_sample_default_command(self) -> bool
    def get_and_reset_sample_custom_commands(self) -> list
```

### Callback Pattern (Trainer → UI)
**File:** `modules/util/callbacks/TrainCallbacks.py`

```python
class TrainCallbacks:
    on_update_train_progress(train_progress, max_step, max_epoch)
    on_update_status(status_string)
    on_sample_default(sampler_output)
    on_sample_custom(sampler_output)
    on_update_sample_default_progress(step, max_step)
    on_update_sample_custom_progress(progress, max_progress)
```

### Data Flow Diagram
```
┌──────────────┐     TrainCommands      ┌──────────────┐
│              │ ──────────────────────→│              │
│   TrainUI    │                        │   Trainer    │
│   (UI Thread)│ ←──────────────────────│(Worker Thread)│
│              │     TrainCallbacks     │              │
└──────────────┘                        └──────────────┘
       ↕                                       ↕
   UIState ←→ TrainConfig              TrainConfig (readonly)
```

---

## 7. State Management

### UIState Class
**File:** `modules/util/ui/UIState.py`

Two-way binding between Python objects and Tkinter variables:

```python
ui_state = UIState(master, train_config)

# Get tkinter variable for widget binding
var = ui_state.get_var("learning_rate")

# Add change listener
trace_id = ui_state.add_var_trace("learning_rate", callback)

# Update from new config
ui_state.update(new_config)
```

**Type Mapping:**
- `str`, `Enum` → `tk.StringVar`
- `bool` → `tk.BooleanVar`
- `int`, `float` → `tk.StringVar` (with validation)
- Nested `BaseConfig` → Nested `UIState`

### State Persistence
| What | Where |
|------|-------|
| Default config | `training_presets/#.json` |
| User presets | `training_presets/{name}.json` |
| Secrets | `secrets.json` (separate) |
| Training config | `workspace/config/{timestamp}.json` |
| Concepts | `training_concepts/*.json` |
| Samples | `training_samples/*.json` |

---

## 8. Real-Time Updates

### Training Progress
**File:** `modules/util/TrainProgress.py`

```python
class TrainProgress:
    epoch: int           # Current epoch
    epoch_step: int      # Step within epoch
    epoch_sample: int    # Sample count
    global_step: int     # Total steps
```

### Update Flow
```
Trainer.train() loop
    │
    ├── Loss calculation → Tensorboard
    │
    ├── callbacks.on_update_train_progress()
    │   └── UI updates progress bars, ETA
    │
    ├── callbacks.on_update_status()
    │   └── UI updates status label
    │
    └── Every N steps: sample generation
        └── callbacks.on_sample_default()
            └── UI displays in SampleWindow
            └── Tensorboard add_image()
```

### Sample Output
**File:** `modules/modelSampler/BaseModelSampler.py`

```python
class ModelSamplerOutput:
    file_type: FileType  # IMAGE, VIDEO, AUDIO
    data: Image | Tensor | bytes
```

### Tensorboard Integration
```python
tensorboard_log_dir = workspace_dir / "tensorboard"
self.tensorboard = SummaryWriter(tensorboard_log_dir)

# Logged metrics:
self.tensorboard.add_scalar("loss/train_step", loss, step)
self.tensorboard.add_scalar("smooth_loss/train_step", ema_loss, step)
self.tensorboard.add_image("sample - {prompt}", image, step)
```

---

## 9. Configuration System

### TrainConfig Structure
**File:** `modules/util/config/TrainConfig.py`

```python
@dataclass
class TrainConfig(BaseConfig):
    # General
    training_method: TrainingMethod
    model_type: ModelType
    workspace_dir: str

    # Model
    base_model_name: str
    output_dtype: DataType

    # Training
    learning_rate: float
    epochs: int
    batch_size: int
    optimizer: TrainOptimizerConfig  # Nested config

    # Data
    concepts: list[ConceptConfig]

    # And ~200 more fields...
```

### Config Serialization
```python
# To JSON dict (for saving)
config.to_dict()
config.to_settings_dict(secrets=False)  # Without secrets
config.to_pack_dict()  # Full with concepts/samples

# From JSON dict
TrainConfig.from_dict(data)
```

### Config Versioning
- Current version: 10
- Automatic migrations on load
- Backward compatible

---

## 10. Threading Model

### Thread Architecture
```
Main Thread (UI)              Worker Thread (Training)
─────────────────             ──────────────────────────
• Tkinter mainloop            • GenericTrainer.train()
• Handle user input           • Check commands each step
• Update widgets              • Fire callbacks
• Spawn training thread       • Tensorboard logging
                              • Sample generation
```

### Thread Communication
```python
# UI → Trainer (commands)
self.training_commands.stop()

# Trainer → UI (callbacks)
self.callbacks.on_update_train_progress(...)

# Thread-safe UI updates
self.after(0, self._update_ui_method)  # Queue on UI thread
```

### Thread Lifecycle
```python
def start_training(self):
    self.training_thread = Thread(target=self.__training_thread_function)
    self.training_thread.start()

def __training_thread_function(self):
    trainer = create_trainer(config, callbacks, commands)
    trainer.start()
    trainer.train()
    trainer.end()
    self.training_thread = None  # Cleanup
```

---

## 11. File Inventory

### UI Files (25 modules)
```
modules/ui/
├── TrainUI.py              (913 lines)  Main window
├── TopBar.py               (250 lines)  Header bar
├── TrainingTab.py          (848 lines)  Training params
├── ModelTab.py             (600 lines)  Model config
├── ConceptTab.py           (300 lines)  Concepts list
├── ConceptWindow.py        (1200 lines) Concept editor
├── SamplingTab.py          (150 lines)  Samples list
├── SampleWindow.py         (250 lines)  Sample preview
├── SampleFrame.py          (150 lines)  Sample UI
├── LoraTab.py              (260 lines)  LoRA/PEFT config
├── AdditionalEmbeddingsTab.py (150 lines)
├── CloudTab.py             (350 lines)  Cloud training
├── CaptionUI.py            (600 lines)  Captioning tool
├── VideoToolUI.py          (900 lines)  Video tools
├── ConvertModelUI.py       (200 lines)  Model conversion
├── ConfigList.py           (350 lines)  Abstract list base
├── OptimizerParamsWindow.py (600 lines)
├── SchedulerParamsWindow.py (150 lines)
├── TimestepDistributionWindow.py (200 lines)
├── GenerateCaptionsWindow.py (150 lines)
├── GenerateMasksWindow.py  (180 lines)
├── OffloadingWindow.py     (100 lines)
├── MuonAdamWindow.py       (180 lines)
├── ProfilingWindow.py      (60 lines)
└── SampleParamsWindow.py   (40 lines)

modules/util/ui/
├── UIState.py              (200 lines)  State binding
├── components.py           (500 lines)  Component factory
├── ToolTip.py              (60 lines)   Tooltips
├── ui_utils.py             (100 lines)  Utilities
└── dialogs.py              (50 lines)   Dialogs
```

### Core Bridge Files
```
modules/util/commands/TrainCommands.py    (91 lines)
modules/util/callbacks/TrainCallbacks.py  (96 lines)
modules/util/TrainProgress.py             (50 lines)
modules/util/config/TrainConfig.py        (1200+ lines)
modules/util/config/BaseConfig.py         (600 lines)
```

### Training Core (UI-Independent)
```
modules/trainer/GenericTrainer.py         (1000 lines)
modules/trainer/BaseTrainer.py            (100 lines)
modules/util/create.py                    (1800 lines)
```

---

## 12. Migration Path

### Phase 1: FastAPI Backend

**Create API layer without touching existing code:**

```
api/
├── main.py                 # FastAPI app
├── routes/
│   ├── config.py          # Config CRUD endpoints
│   ├── training.py        # Start/stop/status
│   ├── samples.py         # Sample requests
│   └── monitoring.py      # Progress, loss, etc.
├── websocket/
│   └── training_ws.py     # Real-time updates
└── services/
    └── trainer_service.py # Wraps existing trainer
```

**Key Endpoints:**
```
POST   /api/training/start     # Start training
POST   /api/training/stop      # Stop training
GET    /api/training/status    # Current status
WS     /api/training/stream    # Real-time updates

GET    /api/config             # Get current config
PUT    /api/config             # Update config
GET    /api/config/presets     # List presets
POST   /api/config/presets     # Save preset

POST   /api/samples/generate   # Trigger sampling
GET    /api/samples/latest     # Get latest samples
```

### Phase 2: Adapt Existing Interfaces

**TrainCallbacks → WebSocket events:**
```python
class WebSocketCallbacks(TrainCallbacks):
    def __init__(self, ws_manager):
        self.ws = ws_manager

    def on_update_train_progress(self, progress, max_step, max_epoch):
        self.ws.broadcast({
            "type": "progress",
            "epoch": progress.epoch,
            "step": progress.epoch_step,
            "max_step": max_step
        })
```

**TrainCommands → REST endpoints:**
```python
@router.post("/training/stop")
async def stop_training():
    trainer_service.commands.stop()
    return {"status": "stopping"}
```

### Phase 3: Frontend (Pick One)

**Option A: Svelte** (recommended for simplicity)
```
frontend/
├── src/
│   ├── routes/           # Pages
│   ├── components/       # UI components
│   ├── stores/           # State management
│   └── lib/api.ts        # API client
```

**Option B: Vue 3**
```
frontend/
├── src/
│   ├── views/
│   ├── components/
│   ├── composables/
│   └── api/
```

**Option C: Flutter** (for native + web)
```
frontend/
├── lib/
│   ├── screens/
│   ├── widgets/
│   ├── providers/
│   └── services/
```

### What Can Be Reused

| Component | Reusability |
|-----------|-------------|
| TrainConfig | 100% - JSON serialization ready |
| TrainCommands | 100% - Already decoupled |
| TrainCallbacks | 90% - Wrap for WebSocket |
| GenericTrainer | 100% - No UI dependency |
| All model/training logic | 100% |
| UIState | 0% - Tkinter specific |
| UI components | 0% - Need rewrite |

### Migration Effort Estimate

| Task | Complexity |
|------|------------|
| FastAPI skeleton | Low |
| Config endpoints | Low |
| Training control endpoints | Low |
| WebSocket real-time | Medium |
| Sample streaming | Medium |
| Frontend basics | Medium |
| Full UI parity | High |
| Testing | Medium |

**Recommended approach:** Incremental - get basic training working via API first, then add features progressively.

---

## Appendix: Key Code Locations

### Starting Training (Current Flow)
```python
# TrainUI.py:784
def start_training(self):
    self.training_thread = Thread(target=self.__training_thread_function)
    self.training_thread.start()

# TrainUI.py:740
def __training_thread_function(self):
    trainer = create.create_trainer(
        self.train_config,
        self.training_callbacks,
        self.training_commands
    )
    trainer.start()
    trainer.train()
    trainer.end()
```

### Creating Trainer (Reusable)
```python
# modules/util/create.py
def create_trainer(config, callbacks, commands) -> GenericTrainer:
    # Creates trainer based on model_type and training_method
    # This is the main entry point for training
```

### Training Loop (Core)
```python
# modules/trainer/GenericTrainer.py:600+
def train(self):
    for epoch in range(epochs):
        for batch in dataloader:
            # Check commands
            if self.commands.get_stop_command():
                break

            # Forward/backward
            loss = self.model_setup.calculate_loss(...)
            loss.backward()
            optimizer.step()

            # Callbacks
            self.callbacks.on_update_train_progress(...)

            # Periodic sampling
            if should_sample:
                self.__sample_during_training()
```

---

*End of Analysis*
