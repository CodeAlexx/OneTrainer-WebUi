# STAGEHAND ACCEPTANCE CHECKLIST
# Pass/fail gate for each phase. Do not proceed to next phase until current passes.
# Target: Serenity (Python) / 3090 24GB / 64GB System RAM

## PHASE 1: PinnedPool
- [ ] Pool pre-allocates exact number of slabs at init (no lazy alloc)
- [ ] acquire() never calls torch.empty() or any allocator
- [ ] acquire() blocks when pool exhausted (does not crash or allocate)
- [ ] release() returns slab to free-list (verified by acquire succeeding after)
- [ ] 1000 acquire/release cycles: RSS stays flat (+-10MB)
- [ ] Blocked acquire wakes within 10ms of release from another thread
- [ ] stats() reports accurate counts at every point
- [ ] Pool rejects init if any registered block exceeds pool capacity
- [ ] Thread-safety: concurrent acquire/release from 4 threads, no corruption

## PHASE 2: Block Registry + Residency Map
- [ ] Registry built from real model (FLUX DiT or equivalent)
- [ ] Block count matches expected architecture
- [ ] exec_order matches forward pass order
- [ ] size_bytes accurate per block (verified against manual calculation)
- [ ] State transitions follow valid paths only (no UNLOADED -> GPU_READY skip)
- [ ] refcount prevents eviction (evict attempt on refcount>0 returns False)
- [ ] Full load/evict cycle: no host memory leak (RSS check)
- [ ] Dependencies correctly enforced (co-resident blocks load together)

## PHASE 3: Transfer Engine
- [ ] max_inflight enforced (submit blocks at limit, doesn't exceed)
- [ ] CUDA events synchronize correctly (compute waits for prefetch)
- [ ] H2D throughput >= 15 GB/s on 3090 PCIe 4.0 (measured)
- [ ] PinnedPool slabs released after D2H completes
- [ ] 100 transfer cycles: no memory growth (host or GPU)
- [ ] Stall logging triggers when wait > 1ms
- [ ] Backpressure: pool exhaustion causes transfer queue to block, not crash
- [ ] drain() completes all inflight before returning

## PHASE 4: Scheduler + Policy
- [ ] Static lookahead prefetches correct blocks (window=3, verified against exec_order)
- [ ] Eviction scores computed correctly (next_use_distance * size_bytes)
- [ ] Eviction respects cooldown (recently used blocks not immediately evicted)
- [ ] Eviction respects refcount (in-use blocks never evicted)
- [ ] Watermark triggers: above high -> evict, below low -> allow prefetch
- [ ] Full FLUX forward pass completes without OOM on 3090
- [ ] Stall count < 10% of block executions with window=3
- [ ] No thrash: no block evicted+loaded more than 2x per forward pass

## PHASE 5: Telemetry
- [ ] All per-step fields populated in JSONL output
- [ ] hit rate calculation matches manual count
- [ ] stall_time_ms correlates with actual wall-clock (+-10%)
- [ ] NaN detection works (inject NaN, verify counter increments)
- [ ] Inf detection works (inject Inf, verify counter increments)
- [ ] Rolling window stats computed correctly
- [ ] Log line format matches spec: [STAGEHAND step=N] hit=X% stall=Xms ...
- [ ] JSONL file parseable by standard tools (python json, jq)

## PHASE 6: Numeric Guards
- [ ] strict_bf16: F32 tensor in transfer -> hard error raised
- [ ] fail_on_dtype_promotion: any promotion detected -> hard error
- [ ] Guards disabled when strict_bf16=false (no false positives)
- [ ] dtype check overhead < 0.1ms per block (not a bottleneck)

## PHASE 7: Integration
- [ ] Training loop runs with stagehand.managed_forward/backward hooks
- [ ] Loss values match non-Stagehand baseline (within BF16 tolerance)
- [ ] Optimizer step works correctly with swapped blocks
- [ ] Gradient accumulation works across swap boundaries
- [ ] Inference mode: no D2H on eviction (frozen blocks just freed)
- [ ] Config loads from YAML correctly
- [ ] Graceful shutdown: drain transfers, release pool, no leaked tensors

## FULL SYSTEM ACCEPTANCE
- [ ] 1000 steps: RSS delta < 100MB (no host memory leak)
- [ ] 1000 steps: VRAM peak < 22000MB after warmup
- [ ] 1000 steps: zero SIG 9
- [ ] 1000 steps: zero torch.cuda.OutOfMemoryError
- [ ] Throughput degradation <= 25% vs full-VRAM baseline
- [ ] Prefetch hit rate >= 90% with window=3
- [ ] Zero dtype promotions in strict mode
- [ ] 10-epoch run: no increasing memory trend
- [ ] 10-epoch run: no increasing stall trend
- [ ] Telemetry JSONL complete and accurate for full run
- [ ] Works with FLUX DiT
- [ ] Works with at least one other model architecture (SD 3.5 or WAN)
