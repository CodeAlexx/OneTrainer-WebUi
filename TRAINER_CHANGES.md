# OneTrainer Trainer Changes Log

Living document tracking improvements to GenericTrainer.py.

## Changes Made (2026-01-04)

### Fix 1: Atomic Backup Pattern
**Location**: `__backup()` method (lines ~508-568)
**Problem**: Crash during backup write = corrupted backup, data loss
**Solution**:
- Write to `backup_path.tmp` first
- If final exists, move to `.old` first
- Atomic move `.tmp` → final
- Clean up `.old` on success, `.tmp` on failure

**Risk**: Very low - only affects backup process

---

### Fix 2: NaN Skip + Emergency Save
**Location**: Training loop (lines ~902-920)
**Problem**: NaN loss → immediate crash, lose all progress
**Solution**:
- Track consecutive NaN count (`_consecutive_nan_count`)
- Skip batch on NaN, log warning
- After 10 consecutive NaNs → emergency backup + stop
- Reset counter on successful step

**New instance variables**:
- `_consecutive_nan_count: int = 0`
- `_max_consecutive_nan: int = 10`

**Risk**: Very low - prevents crashes, adds safety net

---

### Fix 3: SampleTask Dataclass
**Location**: Top of file + `__enqueue_sample_during_training()` + callers
**Problem**: Lambda captures mutable `train_progress` by reference - may execute with wrong values
**Solution**:
- Added `SampleTask` dataclass to snapshot state at enqueue time
- Changed `sample_queue: list[Callable]` → `sample_queue: list[SampleTask]`
- `__execute_sample_during_training()` now creates temporary TrainProgress from snapshot

**New class**:
```python
@dataclass
class SampleTask:
    global_step: int
    epoch: int
    epoch_step: int
    sample_params: list = None
```

**Risk**: Low - fixes potential race condition

---

### Fix 4: mtime-based Config Reload
**Location**: `__reload_epochs_from_config()` method (lines ~247-282)
**Problem**: Reads config file every 5 minutes even if unchanged
**Solution**:
- Check `os.path.getmtime()` first
- Only read file if mtime changed
- Reduced check interval to 1 minute (mtime check is cheap)

**New instance variable**:
- `_config_mtime: float = 0`

**Risk**: Very low - optimization only

---

### Fix 5: GC Batching
**Location**: Top of file + `__init__` + training loop
**Problem**: Scattered `torch_gc()` calls cause sync overhead
**Solution**:
- Added `GCScheduler` class with 30-second minimum interval
- Replace hot-path `torch_gc()` calls with `_gc_scheduler.request_gc()`
- Keep critical `torch_gc()` calls (after save, after sample) unchanged

**New class**:
```python
class GCScheduler:
    def __init__(self, min_interval: float = 30.0)
    def request_gc(self, force: bool = False) -> bool
    def force_gc(self)
```

**Risk**: Low - reduces overhead, maintains safety

---

## Files Modified
- `/home/alex/OneTrainer/modules/trainer/GenericTrainer.py`
- `/home/alex/OneTrainer/modules/modelLoader/Kandinsky5ModelLoader.py` (INT8 Qwen loading)

## Testing Status
- [x] Syntax check (python -m py_compile) - PASSED
- [x] Import test - PASSED
- [x] GCScheduler unit test - PASSED
- [x] SampleTask unit test - PASSED
- [ ] Short training run (needs manual test)
- [ ] NaN injection test (needs manual test)
- [ ] Backup corruption test (needs manual test)

## Rollback
If issues occur, git can restore the original:
```bash
git checkout -- modules/trainer/GenericTrainer.py
```
