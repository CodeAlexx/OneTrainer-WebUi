# OneTrainer Bucketing System Audit

**Date:** 2024-12-24
**Status:** Analysis Complete - Issues Identified

## Architecture Overview

```
Image → CalcAspect → AspectBucketing → AspectBatchSorting → Training
                          ↓
              (assigns to nearest bucket)
```

The bucketing system is implemented in the `mgds` library (Modular Generative Data System), maintained as a submodule at `/venv/src/mgds/`.

## Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `AspectBucketing` | `mgds/pipelineModules/AspectBucketing.py` | Assigns images to resolution buckets based on aspect ratio |
| `AspectBatchSorting` | `mgds/pipelineModules/AspectBatchSorting.py` | Groups same-resolution images into batches (with latent caching) |
| `InlineAspectBatchSorting` | `mgds/pipelineModules/InlineAspectBatchSorting.py` | Streaming version for non-cached training |
| `SingleAspectCalculation` | `mgds/pipelineModules/SingleAspectCalculation.py` | Fallback when bucketing disabled (crops to square) |

## Data Flow

1. **CalcAspect**: Reads image dimensions, outputs `original_resolution`
2. **AspectBucketing**:
   - Takes original resolution + target resolution(s)
   - Finds nearest aspect ratio bucket from hardcoded list
   - Outputs `scale_resolution` and `crop_resolution`
3. **ScaleCropImage**: Scales and crops image to bucket dimensions
4. **AspectBatchSorting**:
   - Groups images by `crop_resolution`
   - Creates batches of `batch_size` images with same resolution
   - **Drops images that don't fill a complete batch**

---

## Issues Found

### 1. Fixed Hardcoded Aspect Ratios ❌ CRITICAL

**Location:** `mgds/pipelineModules/AspectBucketing.py:17-28`

```python
all_possible_input_aspects = [
    (1.0, 1.0),   # 1:1   = 1.000
    (1.0, 1.25),  # 4:5   = 0.800
    (1.0, 1.5),   # 2:3   = 0.667
    (1.0, 1.75),  # 4:7   = 0.571
    (1.0, 2.0),   # 1:2   = 0.500
    (1.0, 2.5),   # 2:5   = 0.400
    (1.0, 3.0),   # 1:3   = 0.333
    (1.0, 3.5),   # 2:7   = 0.286
    (1.0, 4.0),   # 1:4   = 0.250
]
```

**Problem:**
- Only 9 base ratios × 2 (portrait/landscape) = 18 total buckets
- Missing common ratios:
  - 16:9 (1.778) - standard widescreen
  - 21:9 (2.333) - ultrawide
  - 3:2 (1.500) - DSLR photos
  - 5:4 (1.250) - medium format
  - 4:3 (1.333) - classic photos
- Gaps between ratios cause unnecessary cropping

**Impact:** Images with common aspect ratios get cropped to fit nearest bucket.

---

### 2. Silent Sample Dropping ❌ CRITICAL

**Location:** `mgds/pipelineModules/AspectBatchSorting.py:44-49`

```python
for bucket_key in bucket_dict.keys():
    samples = bucket_dict[bucket_key]
    samples_to_drop = len(samples) % self.batch_size
    for i in range(samples_to_drop):
        # print('dropping sample from bucket ' + str(bucket_key))  # COMMENTED OUT!
        samples.pop()
```

**Problem:**
- Silently drops images that don't fill a complete batch
- Debug print is commented out - no logging at all
- User has no visibility into data loss
- With batch_size=4 and 7 images in a bucket, 3 get dropped (43% loss!)

**Impact:** Training data silently discarded. User may think all images are being used.

---

### 3. Inconsistent Quantization Values ⚠️ MODERATE

**Location:** Various `*BaseDataLoader.py` files

| Model | Quantization | File |
|-------|--------------|------|
| SD 1.5 | 8 | `StableDiffusionBaseDataLoader.py:240` |
| SDXL | 64 | `StableDiffusionXLBaseDataLoader.py:260` |
| SD3 | 64 | `StableDiffusion3BaseDataLoader.py:292` |
| Flux | 64 | `FluxBaseDataLoader.py:269` |
| Sana | 32 | `SanaBaseDataLoader.py:227` |
| PixArt | 16 | `PixArtAlphaBaseDataLoader.py:234` |
| Wuerstchen | 128 | `WuerstchenBaseDataLoader.py:212` |
| HunyuanVideo | 64 | `HunyuanVideoBaseDataLoader.py:264` |

**Problem:**
- Quantization is hardcoded per model - user cannot override
- No documentation explaining why different values
- Affects bucket granularity and VRAM usage

---

### 4. No Bucket Balancing ❌ MAJOR

**Location:** `AspectBatchSorting.py`

The batch sorting algorithm:
1. Shuffles images within each bucket
2. Creates batches of `batch_size`
3. Drops overflow images
4. Shuffles batch order

**Missing features:**
- No oversampling of minority buckets
- No weighted sampling based on bucket size
- No bucket count limits
- No epoch-aware balancing

**Impact:** Buckets with fewer images are underrepresented in training. If you have 1000 landscape and 10 portrait images, portraits are severely undertrained.

---

### 5. Poor Multi-Resolution Support ⚠️ MODERATE

**Location:** `mgds/pipelineModules/AspectBucketing.py:188-191`

```python
target_resolutions = [int(res.strip()) for res in target_resolutions.split(',')]
target_resolution = rand.choice(target_resolutions)  # Random choice per image!
```

**Problem:**
- When multiple resolutions specified (e.g., "512, 768, 1024"), randomly picks one per image
- No smart distribution across resolutions
- No way to weight resolution preferences
- Can lead to uneven resolution distribution

---

### 6. No Minimum Bucket Size Configuration ❌ MAJOR

**Problem:**
- Buckets with < `batch_size` images are completely ignored
- ALL images in those buckets are dropped
- No configuration options to:
  - Set minimum bucket threshold
  - Merge small buckets into nearby ones
  - Pad small buckets with duplicates
  - Warn user about problematic buckets

**Impact:** With batch_size=8 and a bucket containing 7 images, all 7 are discarded.

---

### 7. Fixed-Resolution Override Parsing Issues ⚠️ MINOR

**Location:** `mgds/pipelineModules/AspectBucketing.py:132-140`

```python
if 'x' in resolutions and ',' not in resolutions:
    res = resolutions.strip().split('x')
    # Only handles single "WIDTHxHEIGHT" format
```

**Problem:**
- Cannot specify multiple fixed resolutions like "1024x768, 768x1024"
- Mixed format "512, 1024x768" may parse incorrectly

---

### 8. No Aspect Ratio Tolerance ❌ MODERATE

**Location:** `mgds/pipelineModules/AspectBucketing.py:116-117`

```python
bucket_index = np.argmin(abs(self.bucket_aspects[target_resolution] - aspect))
return self.bucket_resolutions[target_resolution][bucket_index]
```

**Problem:**
- Uses simple nearest-neighbor matching
- No tolerance threshold - always picks closest bucket regardless of crop amount
- A 16:9 image (1.778) gets forced into 2:1 (2.0) bucket
  - Difference: 12.5% of image cropped
- No option to create new bucket if crop would be excessive

---

### 9. UI Shows Problems But Can't Fix Them ⚠️ MINOR

**Location:** `modules/ui/ConceptWindow.py:499-562`

The UI provides:
- Bucket distribution bar chart
- Warning about smallest buckets
- Shows which buckets have < batch_size images

**But lacks:**
- Tools to rebalance buckets
- Option to enable sample repeating
- Bucket merging controls
- Preview of which images will be dropped

---

## Current Configuration Options

### Available (Limited)

| Option | Type | Location | Effect |
|--------|------|----------|--------|
| `aspect_ratio_bucketing` | bool | TrainConfig | Enable/disable bucketing entirely |
| `enable_resolution_override` | bool | ConceptConfig | Enable per-concept resolution |
| `resolution_override` | str | ConceptConfig | Custom resolution (e.g., "512" or "1024x768") |

### Missing Options

| Option | Purpose |
|--------|---------|
| `custom_aspect_ratios` | User-defined bucket aspect ratios |
| `bucket_merge_threshold` | Merge buckets with < N images |
| `min_bucket_size` | Minimum images per bucket |
| `repeat_small_buckets` | Duplicate samples to fill batches |
| `quantization_override` | Override model-specific quantization |
| `aspect_tolerance` | Max crop % before creating new bucket |
| `bucket_balancing` | Enable weighted sampling |
| `log_dropped_samples` | Log/warn about dropped images |

---

## Recommendations

### Priority 1: Critical Fixes

1. **Log dropped samples**
   - At minimum, emit warning with count of dropped images per bucket
   - Add to training summary/logs

2. **Add bucket merging option**
   - Merge buckets with < batch_size into nearest neighbor
   - Configurable threshold

3. **Add sample repeating**
   - Option to duplicate samples in small buckets to fill batches
   - Prevents data loss, though may cause some overfitting

### Priority 2: Important Improvements

4. **Add configurable aspect ratios**
   - Allow users to define custom bucket list
   - Include presets: "photo" (3:2, 4:3, 16:9), "art" (current), "video" (16:9, 21:9, 9:16)

5. **Add aspect tolerance**
   - Config option for max acceptable crop percentage
   - Create dynamic buckets if crop would exceed threshold

6. **Add balanced sampling**
   - Weight bucket selection by inverse frequency
   - Ensure underrepresented aspects get trained proportionally

### Priority 3: Nice to Have

7. **Add quantization config**
   - Let users override per-model defaults
   - Useful for memory optimization

8. **Support multiple fixed resolutions**
   - Parse "1024x768, 768x1024" correctly
   - Allow mixing fixed and dynamic resolutions

9. **Add bucket preview in UI**
   - Show exactly which images are in each bucket
   - Preview crop regions before training

---

## Files to Modify

### Core Changes (mgds library)

- `venv/src/mgds/src/mgds/pipelineModules/AspectBucketing.py`
  - Add configurable aspect ratios
  - Add aspect tolerance

- `venv/src/mgds/src/mgds/pipelineModules/AspectBatchSorting.py`
  - Add logging for dropped samples
  - Add bucket merging
  - Add sample repeating option
  - Add balanced sampling

### OneTrainer Integration

- `modules/util/config/TrainConfig.py`
  - Add new bucketing config options

- `modules/dataLoader/mixin/DataLoaderText2ImageMixin.py`
  - Pass new config options to AspectBucketing

- `modules/ui/TrainingTab.py` or new `BucketingTab.py`
  - Add UI controls for new options

---

## References

- mgds repo: https://github.com/Nerogar/mgds
- AspectBucketing implementation: `venv/src/mgds/src/mgds/pipelineModules/AspectBucketing.py`
- Batch sorting: `venv/src/mgds/src/mgds/pipelineModules/AspectBatchSorting.py`
- OneTrainer integration: `modules/dataLoader/mixin/DataLoaderText2ImageMixin.py`
