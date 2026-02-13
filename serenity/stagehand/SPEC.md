# STAGEHAND Implementation Spec — Serenity (Python)

**Version:** 1.0
**Target:** 3090 24GB VRAM / 64GB System RAM
**Stack:** Serenity (Python, PyTorch)
**Models:** FLUX, WAN (video), SD 3.5, future diffusion architectures

---

## 0. Lesson from v1: Why It Died

Previous implementation got SIG 9 (OOM-killed by kernel). Root cause: unbounded host memory allocation. Pinned buffers were allocated on demand with no ceiling, evicted blocks kept host copies alive, and inflight transfers accumulated without backpressure. **This spec treats host memory discipline as Phase 1, not a feature.**

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Stagehand Runtime                     │
│                                                         │
│  ┌──────────┐  ┌───────────┐  ┌──────────────────────┐  │
│  │  Block    │  │ Residency │  │   Transfer Engine    │  │
│  │ Registry  │──│   Map     │──│  (PinnedPool + Async)│  │
│  └──────────┘  └───────────┘  └──────────────────────┘  │
│       │              │                    │              │
│  ┌──────────┐  ┌───────────┐  ┌──────────────────────┐  │
│  │Scheduler │  │  Budget   │  │     Telemetry        │  │
│  │ (Policy) │──│  Manager  │──│                      │  │
│  └──────────┘  └───────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

Components are implemented as separate classes with explicit state. No globals, no singletons.

---

## 2. Implementation Phases

Build and test each phase before starting the next. Each phase has its own acceptance test.

### Phase 1: PinnedPool — The Foundation

**Why first:** This is what killed v1. Everything else depends on bounded host memory.

#### 2.1.1 PinnedPool Class

```python
class PinnedPool:
    """
    Fixed-size pool of pinned host memory buffers.
    Pre-allocated at init. Never grows. Ring-buffer recycling.
    """
```

**Constructor args:**
- `total_mb: int` — Total pinned memory budget (default: 8192 for 8GB on a 64GB system)
- `slab_mb: int` — Individual slab size (default: 256MB, must evenly divide total_mb)
- `alignment: int` — Byte alignment for slabs (default: 512, for DMA)

**Behavior:**
- At `__init__`, pre-allocate `total_mb / slab_mb` pinned tensors via `torch.cuda.HostAllocator` or `torch.empty(..., pin_memory=True)`.
- Store slabs in a free-list (deque).
- `acquire(size_bytes) -> PinnedSlab` — pops from free-list. If free-list is empty, **blocks and waits** (backpressure). Never allocates new memory. Logs a warning after 100ms wait.
- `release(slab: PinnedSlab)` — returns slab to free-list. Asserts slab belongs to this pool.
- `stats() -> dict` — returns `{total, free, in_use, peak_in_use, wait_count, total_wait_ms}`.

**Critical constraints:**
- Pool size is fixed at init. No dynamic growth. Ever.
- `acquire()` must never call `torch.empty()` or any allocator.
- If a block is larger than `slab_mb`, acquire multiple contiguous slabs. Max block size = `total_mb`. If a single block exceeds pool capacity, raise `StagehandOOMError` at init (during registry validation), not at runtime.
- Thread-safe via `threading.Lock` on the free-list.

**Acceptance test (Phase 1):**
```
1. Create pool with total_mb=2048, slab_mb=256 (8 slabs).
2. Acquire all 8 slabs. Verify free-list is empty.
3. Attempt acquire from another thread — must block.
4. Release 1 slab — blocked thread must wake and succeed.
5. After full cycle, RSS must not have grown beyond initial allocation + epsilon.
6. Run 1000 acquire/release cycles — RSS must remain flat.
7. Verify stats() reports correct counts throughout.
```

#### 2.1.2 Host Memory Budget

Total host memory Stagehand may use = `pinned_pool_mb` + metadata overhead (~100MB max).

**Hard rule:** Stagehand must never hold host tensors outside the PinnedPool. No `torch.empty()` calls anywhere in the runtime after init. All host-side block storage goes through PinnedPool.

On a 64GB system with the OS + PyTorch + dataloader, safe default is `pinned_pool_mb = 8192` (8GB). This leaves ~50GB+ for everything else.

---

### Phase 2: Block Registry + Residency Map

#### 2.2.1 BlockRegistry

```python
@dataclass
class BlockEntry:
    block_id: str              # e.g., "dit.block.14" or "unet.down.2.attn"
    module_ref: nn.Module      # weak reference to actual module
    size_bytes: int            # parameter size in target dtype
    dtype: torch.dtype         # bf16, fp16, etc.
    dependencies: list[str]    # block_ids that must be co-resident
    group: str                 # e.g., "dit", "te1", "te2", "vae"
    exec_order: int            # position in forward pass execution
    quant_format: str | None   # e.g., "w4g128", None for full precision
    quant_meta_bytes: int      # size of scales/zeros/outlier sidecars
```

**Registry construction:**
- Built once at model load time by walking `model.named_modules()`.
- Validates that no single block exceeds PinnedPool capacity.
- Sorts by `exec_order` (derived from model topology).
- Immutable after construction. If model topology changes, rebuild.

#### 2.2.2 ResidencyMap

```python
class BlockState(Enum):
    UNLOADED = 0      # not on GPU, not on host (or freed from host)
    HOST_STAGED = 1   # in PinnedPool, ready for H2D
    PREFETCHING = 2   # H2D transfer in progress
    GPU_READY = 3     # on GPU, available for compute
    EVICTING = 4      # D2H transfer in progress (if saving back)
    GPU_FREEING = 5   # being freed from GPU (no save-back needed)

@dataclass
class ResidencyEntry:
    state: BlockState
    gpu_tensor: torch.Tensor | None
    host_slab: PinnedSlab | None     # reference to pool slab
    refcount: int                     # >0 means in-use, cannot evict
    last_used_step: int
    next_use_step: int | None        # from scheduler lookahead
    transfer_event: torch.cuda.Event | None  # for async sync
```

**Key behaviors:**
- `refcount > 0` blocks eviction. Forward pass increments on entry, decrements on exit.
- `GPU_READY` with `refcount == 0` and `next_use_step` far away = eviction candidate.
- On eviction: if block is frozen (inference or non-trained LoRA base), skip D2H (just free GPU). If block has gradients, D2H first.
- On eviction completion: release PinnedSlab back to pool.
- **No block may exist on host outside the PinnedPool.** If a block is evicted from GPU and not needed on host, its slab is released immediately.

**Acceptance test (Phase 2):**
```
1. Build registry from a real model (e.g., FLUX DiT).
2. Verify block count, sizes, exec_order correctness.
3. Walk state transitions: UNLOADED -> HOST_STAGED -> PREFETCHING -> GPU_READY -> GPU_FREEING -> UNLOADED.
4. Verify refcount guards prevent eviction during simulated forward.
5. Verify no host memory leaks after full load/evict cycle.
```

---

### Phase 3: Transfer Engine

#### 2.3.1 AsyncTransferEngine

```python
class AsyncTransferEngine:
    """
    Manages async H2D/D2H copies using CUDA streams and events.
    Bounded by max_inflight and PinnedPool availability.
    """
```

**Constructor args:**
- `pool: PinnedPool`
- `max_inflight: int` — Max concurrent transfers (default: 2 for 3090, PCIe 4.0 x16 doesn't benefit from more)
- `copy_stream: torch.cuda.Stream` — Dedicated stream for copies (not the compute stream)

**Methods:**
- `submit_h2d(block_id, host_slab, gpu_dest) -> TransferHandle`
  - Copies from pinned slab to GPU tensor on `copy_stream`.
  - Records a CUDA event on completion.
  - Returns handle for polling/waiting.
- `submit_d2h(block_id, gpu_src, host_slab) -> TransferHandle`
  - Copies from GPU to pinned slab on `copy_stream`.
  - For frozen blocks during inference, this is typically skipped (just free GPU).
- `poll(handle) -> bool` — Non-blocking check if transfer completed.
- `wait(handle)` — Blocking wait. Logs stall if wait > 1ms.
- `inflight_count() -> int`
- `drain()` — Wait for all inflight transfers to complete.

**Backpressure:**
- If `inflight_count() >= max_inflight`, `submit_*` blocks until a slot opens.
- If PinnedPool has no free slabs, `submit_h2d` blocks (inherits pool backpressure).
- Both blocking paths log warnings with timing.

**Overlap contract:**
- Compute runs on `torch.cuda.default_stream()`.
- Copies run on `copy_stream`.
- Before compute uses a prefetched block: `default_stream.wait_event(transfer_event)`.
- This is the core latency hiding mechanism.

**Acceptance test (Phase 3):**
```
1. Create engine with max_inflight=2.
2. Submit 4 H2D transfers. Verify only 2 are inflight, others queue.
3. As first 2 complete, next 2 start automatically.
4. Verify CUDA events synchronize correctly (compute waits for copy).
5. Verify PinnedPool slabs are returned after D2H completes.
6. Measure: H2D throughput should approach PCIe 4.0 bandwidth (~25 GB/s).
7. Run 100 transfers — no memory growth.
```

---

### Phase 4: Scheduler + Policy

#### 2.4.1 Scheduler

```python
class StagehandScheduler:
    """
    Decides what to prefetch and what to evict at each step.
    """
```

**Inputs per step:**
- `current_step: int`
- `execution_cursor: int` — Which block in exec_order is currently computing.
- `token_budget: int | None` — For variable-length batches (optional).

**Core loop (called once per block execution):**

```
1. Advance cursor.
2. Check: is current block GPU_READY?
   - Yes: proceed to compute.
   - No: STALL. Wait on its transfer. Log stall event.
3. Compute current block (on default stream).
4. Decrement refcount on completed blocks.
5. Issue prefetch for blocks in lookahead window that aren't GPU_READY.
6. If VRAM above high watermark: run eviction pass.
7. Update telemetry.
```

#### 2.4.2 Static Lookahead Policy (Ship First)

```python
class StaticLookaheadPolicy:
    """
    Prefetch a fixed number of blocks ahead in execution order.
    Deterministic. Easy to debug.
    """
```

**Config:**
- `prefetch_window: int` — Number of blocks ahead to prefetch (default: 3).

**Eviction scoring:**
Simple formula: `score = next_use_distance * size_bytes`
Higher score = better eviction candidate (far away and large).

**Eviction rules:**
- Never evict a block with `refcount > 0`.
- Never evict a block within the prefetch window.
- Evict in descending score order until below `vram_low_watermark`.
- Eviction cooldown: block cannot be evicted if `last_used_step` was within last 2 steps (prevents thrash on small windows).

#### 2.4.3 Watermark Budget Manager

```python
class BudgetManager:
    """
    Tracks VRAM usage and enforces watermark thresholds.
    """
```

**Config:**
- `vram_high_watermark_mb: int` — Trigger eviction above this (default: 22000 for 3090, leaving 2GB headroom for activations/optimizer).
- `vram_low_watermark_mb: int` — Stop evicting below this (default: 18000).

**Behavior:**
- Queries `torch.cuda.memory_allocated()` and `torch.cuda.memory_reserved()`.
- Above high watermark: scheduler must evict before prefetching more.
- Below low watermark: opportunistic prefetch allowed.
- Between: normal operation per policy.

**3090-specific defaults:**
- Total VRAM: 24576 MB
- High watermark: 22000 MB (reserve 2.5GB for activations + optimizer states + PyTorch cache)
- Low watermark: 18000 MB
- These are conservative. Tune up after validating stability.

**Acceptance test (Phase 4):**
```
1. Load a model that exceeds VRAM (e.g., full FLUX DiT, ~24GB params).
2. Run forward pass with static lookahead window=3.
3. Verify: all blocks compute successfully, no OOM.
4. Verify: stall count < 10% of block executions.
5. Verify: VRAM stays below high watermark after initial warmup.
6. Verify: no thrash (same block evicted+loaded more than twice per epoch).
7. Compare throughput to musubi-style sync swap baseline.
```

---

### Phase 5: Telemetry

#### 2.5.1 StagehandTelemetry

```python
class StagehandTelemetry:
    """
    Per-step and rolling-window metrics. Prints summary every N steps.
    """
```

**Per-step metrics:**
- `h2d_bytes: int`
- `d2h_bytes: int`
- `copy_time_ms: float`
- `compute_time_ms: float`
- `stall_time_ms: float` — Time compute waited on a transfer.
- `stall_count: int` — Number of stall events.
- `evictions: int`
- `prefetch_hits: int` — Block was GPU_READY when needed.
- `prefetch_misses: int` — Block needed sync wait.
- `vram_used_mb: float`
- `vram_reserved_mb: float`
- `host_pool_free: int` — Free slabs in PinnedPool.
- `host_pool_in_use: int`
- `dtype_promotions: int` — Must be 0 in strict mode.
- `nan_count: int`
- `inf_count: int`

**Rolling window (last 100 steps):**
- Mean/max stall time
- Prefetch hit rate (%)
- Effective throughput (it/s or tokens/s)
- VRAM usage trend (growing/stable/shrinking)

**Output:**
- Log line every `telemetry_interval_steps` (default: 10).
- Full dump to JSONL file for post-analysis.
- Format: `[STAGEHAND step=42] hit=95.2% stall=1.3ms evict=2 vram=21.4G pool=6/8`

**Acceptance test (Phase 5):**
```
1. Run 100 steps with telemetry enabled.
2. Verify JSONL file has 100 entries with all fields populated.
3. Verify hit rate calculation matches manual count.
4. Verify stall_time_ms correlates with actual wall-clock delays.
5. Verify nan_count and inf_count work (inject a NaN, confirm detection).
```

---

### Phase 6: Numeric Guards

#### 2.6.1 BF16 Enforcement

**Config:**
- `strict_bf16: bool` — Default True. Any F32 tensor touching a Stagehand-managed block = hard error.
- `fail_on_dtype_promotion: bool` — Default True.

**Checks (performed in transfer engine):**
- On H2D submit: assert source dtype matches registry dtype.
- On GPU allocation: assert allocated dtype matches.
- On compute entry: spot-check first tensor dtype (optional, enable with `stagehand_debug_trace`).

#### 2.6.2 NaN/Inf Detection

- After each block's forward pass, check output tensor for NaN/Inf.
- Lightweight: `torch.isfinite(output).all()` — single kernel, negligible cost.
- On detection: log block_id, step, and tensor stats. Optionally halt.

---

### Phase 7: Integration with Serenity Training Loop

#### 2.7.1 Entry Points

Stagehand integrates via hooks, not by rewriting the training loop:

```python
# In trainer init:
stagehand = StagehandRuntime(model, config)

# In training loop:
for step, batch in enumerate(dataloader):
    stagehand.begin_step(step)

    # Forward pass — stagehand manages block residency
    with stagehand.managed_forward():
        loss = model(batch)

    # Backward pass — stagehand manages gradient block residency
    with stagehand.managed_backward():
        loss.backward()

    optimizer.step()
    stagehand.end_step()
```

#### 2.7.2 managed_forward / managed_backward

These context managers install forward/backward hooks on each registered module:

**Forward hook (per block):**
```
1. Ensure block is GPU_READY (wait if PREFETCHING, error if UNLOADED).
2. Increment refcount.
3. Signal scheduler: cursor advanced.
4. [compute runs normally]
5. On exit: decrement refcount, update last_used_step.
```

**Backward hook (per block):**
Same residency management, but in reverse exec_order.

#### 2.7.3 Inference Mode

Same runtime, but:
- No backward hooks.
- No D2H on eviction (blocks are frozen, no gradients to save).
- More aggressive eviction (no gradient state to preserve).

---

## 3. Configuration

```yaml
# stagehand_config.yaml

stagehand_enabled: true

# Memory
pinned_pool_mb: 8192          # 8GB of 64GB system RAM
pinned_slab_mb: 256           # Individual slab size
vram_high_watermark_mb: 22000 # Trigger eviction (3090: 24GB - 2.5GB headroom)
vram_low_watermark_mb: 18000  # Stop evicting

# Transfer
max_inflight_transfers: 2     # PCIe 4.0 x16, 2 is sufficient
copy_stream_priority: -1      # High priority CUDA stream

# Scheduler
policy: static                # static | adaptive (ship static first)
prefetch_window_blocks: 3     # Blocks ahead to prefetch
eviction_cooldown_steps: 2    # Min steps before re-eviction

# Numeric
strict_bf16: true
fail_on_dtype_promotion: true
nan_inf_check: true           # Per-block output validation

# Telemetry
telemetry_enabled: true
telemetry_interval_steps: 10
telemetry_file: stagehand_telemetry.jsonl

# Debug
debug_trace: false            # Verbose per-transfer logging
```

---

## 4. File Layout in Serenity

```
serenity/
  stagehand/
    __init__.py               # StagehandRuntime (top-level API)
    pool.py                   # PinnedPool
    registry.py               # BlockRegistry, BlockEntry
    residency.py              # ResidencyMap, BlockState, ResidencyEntry
    transfer.py               # AsyncTransferEngine, TransferHandle
    scheduler.py              # StagehandScheduler, StaticLookaheadPolicy
    budget.py                 # BudgetManager (watermark logic)
    telemetry.py              # StagehandTelemetry
    guards.py                 # BF16 enforcement, NaN/Inf checks
    config.py                 # StagehandConfig (pydantic or dataclass)
    errors.py                 # StagehandOOMError, DtypeMismatchError, etc.
    tests/
      test_pool.py            # Phase 1 acceptance tests
      test_registry.py        # Phase 2 acceptance tests
      test_transfer.py        # Phase 3 acceptance tests
      test_scheduler.py       # Phase 4 acceptance tests
      test_telemetry.py       # Phase 5 acceptance tests
      test_integration.py     # Phase 7 end-to-end
      stress_test.py          # Long-running stability (memory leak detection)
```

---

## 5. 3090-Specific Notes

- **PCIe 4.0 x16 bandwidth:** ~25 GB/s theoretical, ~20 GB/s practical. A 256MB block takes ~13ms to transfer. With window=3 and ~50ms compute per block, prefetch easily hides latency.
- **VRAM fragmentation:** PyTorch's caching allocator on 3090 can fragment badly with mixed sizes. Stagehand should allocate GPU block tensors in a contiguous region if possible, or use `torch.cuda.memory.CUDAPluggableAllocator` if available.
- **Compute/copy overlap:** 3090 has 1 copy engine. True overlap requires compute on default stream + copy on dedicated stream. Verify with `nsys` that they actually overlap (not serialized).
- **Power limit:** 3090 at 350W under full load may throttle. Telemetry should track `torch.cuda.utilization()` to detect thermal throttle causing apparent stalls.

---

## 6. Model-Specific Integration Notes

### FLUX (DiT)
- ~57 DiT blocks, each ~400MB in BF16.
- Text encoders (T5-XXL + CLIP) can be offloaded entirely after encoding.
- Execution order is strictly sequential — ideal for static lookahead.

### WAN (Video)
- Temporal + spatial attention blocks, larger per-block due to video dimensions.
- Token counts vary significantly with video length — scheduler must handle variable compute time per block.
- Higher host memory pressure due to larger activations. May need to reduce `pinned_pool_mb` if dataloader is also memory-heavy.

### SD 3.5 (MMDiT)
- Dual-stream architecture. Blocks may have co-residency requirements (text + image streams).
- Use `dependencies` field in BlockEntry to ensure paired blocks are co-loaded.

---

## 7. Acceptance Criteria (Full System)

The system is "done" when ALL of these pass:

1. **No SIG 9.** System RAM stays bounded. PinnedPool never grows beyond init. Run for 1000 steps, RSS stays flat (+-100MB).
2. **No OOM.** VRAM stays below high watermark after warmup (first 10 steps excluded). Run full training epoch without `torch.cuda.OutOfMemoryError`.
3. **Throughput.** Degradation vs full-VRAM baseline <= 25% for static policy. (Musubi is typically 30-50% degradation, so beating that is the bar.)
4. **Stall rate.** Prefetch hit rate >= 90% with window=3 on sequential models.
5. **Numeric parity.** BF16 strict mode: zero dtype promotions. NaN/Inf count matches non-swapped baseline (i.e., Stagehand doesn't introduce numeric issues).
6. **Stability.** 10-epoch run shows no increasing memory trend, no increasing stall trend, no thrash patterns.
7. **Telemetry.** All metrics populated and accurate for full run. JSONL parseable.

---

## 8. What NOT to Build (Scope Limits)

- **No adaptive policy yet.** Static only until acceptance criteria pass. Adaptive is Phase 8+.
- **No quantized slab support yet.** BF16 full blocks only. SquareQ integration is a future phase.
- **No multi-GPU.** Single GPU only. DDP/FSDP integration is separate.
- **No custom CUDA kernels.** Pure PyTorch ops for pool/transfer. Optimize later if needed.
- **No UI integration.** CLI config only. EriUI hooks come after core is validated.

---

## 9. Testing Strategy for CC/Codex

Each phase should be implemented and tested independently. Tests should be runnable without a GPU where possible (mock CUDA for pool/registry tests), and with a GPU for transfer/scheduler/integration tests.

**Test order matches phase order.** Do not start Phase N+1 until Phase N tests pass.

**Stress test (run last):**
```bash
python -m serenity.stagehand.tests.stress_test \
    --model flux-dit \
    --steps 1000 \
    --batch_size 1 \
    --monitor_rss \
    --monitor_vram \
    --output stress_results.json
```

Must show:
- RSS delta < 100MB over 1000 steps
- VRAM peak < high_watermark after step 10
- Zero SIG 9, zero OOM
- Stall rate logged and within threshold
