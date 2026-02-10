# Serenity Inference — Standalone Project Fix Prompt

## WHY THIS EXISTS

Serenity's inference engine (10K+ lines, 64 files) was written by a previous AI that made 154 mocked tests pass while the real system generates zero images. Every architecture is broken. We're splitting inference into its own project to fix it in isolation — when it works end-to-end, it goes back into serenity.

---

## PROJECT STRUCTURE

```
/home/alex/serenity-inference/          ← NEW standalone project
├── models/                             ← Pure PyTorch model definitions
│   ├── unet.py                         ← SD15, SDXL, SDXL Refiner
│   ├── flux_dit.py                     ← Flux 1 Dev/Schnell DiT
│   ├── flux2_dit.py                    ← Flux 2 Dev (Mistral) + Klein 4B/9B (Qwen3)
│   ├── sd3_transformer.py             ← SD3 joint transformer
│   ├── chroma_dit.py                   ← Chroma (Flux variant, T5)
│   ├── zimage_dit.py                   ← ZImage (Qwen2, different from Klein)
│   ├── vae.py                          ← Standard KL VAE (SD15/SDXL/Flux1/SD3)
│   ├── vae_flux2.py                    ← Flux 2 VAE (BN + pixel shuffle)
│   └── detection.py                    ← Architecture detection from checkpoint keys
├── text/                               ← Text encoder implementations
│   ├── clip.py                         ← CLIP-L, CLIP-G encoders
│   ├── t5.py                           ← T5-XXL encoder
│   ├── qwen3.py                        ← Qwen3 encoder (Klein — stacked layers)
│   ├── mistral.py                      ← Mistral encoder (Flux 2 Dev)
│   ├── qwen2.py                        ← Qwen2 encoder (ZImage — hidden_states[-2])
│   └── manager.py                      ← Encoder routing per architecture
├── engine/                             ← Inference pipeline
│   ├── loader.py                       ← Checkpoint loading, extract_submodel
│   ├── pipeline.py                     ← Main generate() loop
│   ├── conditioning.py                 ← Architecture-specific conditioning
│   └── denoise.py                      ← Denoise function (calls model with correct kwargs)
├── sampling/                           ← Samplers, schedulers (COPY from serenity — works)
├── vae/                                ← Tiled decoding (COPY from serenity — works)
├── attention/                          ← Backend selection (COPY from serenity — works)
├── memory/                             ← Offload/VRAM budget (COPY from serenity — works)
├── quantization/                       ← Quantized layers (COPY from serenity — works)
├── lora/                               ← LoRA loading/merging (COPY from serenity — works)
├── cache/                              ← Text cache (COPY from serenity — works)
├── ui/                                 ← Desktop UI (NEW)
│   └── (see Agent 4 section)
├── config.py                           ← Inference config
├── tests/                              ← Tests (real model tests, not mocks)
└── pyproject.toml
```

**Source to copy working subsystems from:** `/home/alex/serenity/serenity/inference/`
**Training code reference (READ ONLY):** `/home/alex/serenity/serenity/models/`
**Models directory:** `/home/alex/EriDiffusion/Models/`
**Python:** `/home/alex/serenity/venv/bin/python`

---

## HARD RULES

1. **NO DIFFUSERS** — Zero `from diffusers` imports anywhere. Using `transformers` (HuggingFace) for text encoders is fine. Using `safetensors` is fine. Only `diffusers` is banned.
2. **DON'T TOUCH SERENITY** — `/home/alex/serenity/serenity/` is read-only reference. All work happens in the new standalone project.
3. **TEST WITH REAL MODELS** — No mocked tests. The previous implementation passed 154 mocked tests and generated zero images. Every fix must be verified with a real checkpoint.
4. **ComfyUI-style loading** — Model key names match the checkpoint format. Weights load with `model.load_state_dict()` directly. If you need key conversion, your model definition is wrong.
5. **Code style** — `from __future__ import annotations`, dataclasses, `__all__` exports, type hints, short docstrings.

---

## ARCHITECTURE REFERENCE — VERIFIED FROM TRAINING CODE

### Text Encoder Mapping (CRITICAL — previous attempt got this completely wrong)

| Architecture | Text Encoder(s) | Notes |
|---|---|---|
| SD 1.5 | CLIP (single) | `openai/clip-vit-large-patch14` |
| SDXL | CLIP-L + CLIP-G | Dual CLIP, pooled from CLIP-G |
| SDXL Refiner | CLIP-G only | Single CLIP |
| SD3 / SD3.5 | CLIP + CLIP + T5 | Triple encoder |
| Flux 1 Dev/Schnell | CLIP-L + T5-XXL | Dual text encoders |
| **Flux 2 Dev** | **Mistral** | **Single encoder — NOT CLIP+T5** |
| **Flux 2 Klein 4B** | **Qwen3** | **Stacked layers [9,18,27] → joint_dim=7680** |
| **Flux 2 Klein 9B** | **Qwen3** | **Stacked layers [9,18,27] → joint_dim=12288** |
| Chroma | T5 (single) | Distilled guidance, no guidance embeddings |
| **ZImage** | **Qwen2 (causal LM)** | **hidden_states[-2], hidden_dim=3840. Different from Klein's Qwen3** |
| Qwen Image | Qwen2.5-VL | Vision-language model |

### Model Architecture Constants

**Flux 1 Dev/Schnell:**
```
double_blocks: 19, single_blocks: 38
text_id_dim: 3
text_encoders: CLIP-L + T5-XXL
guidance_embeds: True (Dev), False (Schnell)
in_channels: 64
```

**Flux 2 Dev:**
```
text_id_dim: 4
text_encoder: Mistral (single)
patchify: 2x2 spatial patchification
BN normalization on patchified latents
block counts: TBD from checkpoint inspection
```

**Flux 2 Klein 4B:**
```
double_blocks: 5, single_blocks: 20
num_attention_heads: 24, attention_head_dim: 128
inner_dim: 3072 (24 * 128)
in_channels: 128 (after VAE patchification)
joint_attention_dim: 7680 (3 * 2560 stacked Qwen3 layers)
text_encoder: Qwen3
text_encoder_hidden: 2560
text_encoder_layers: (9, 18, 27)
guidance_embeds: False
mlp_ratio: 3.0, rope_theta: 2000
axes_dims_rope: (32, 32, 32, 32)
VAE: batch normalization + pixel shuffle (32 → 128 channels)
```

**Flux 2 Klein 9B:**
```
double_blocks: 8, single_blocks: 24
num_attention_heads: 32, attention_head_dim: 128
inner_dim: 4096 (32 * 128)
in_channels: 128
joint_attention_dim: 12288 (3 * 4096 stacked Qwen3 layers)
text_encoder_hidden: 4096
text_encoder_layers: (9, 18, 27)
guidance_embeds: False
```

**SD 1.5:**
```
model_channels: 320, context_dim: 768
no ADM conditioning
res_mult: 8
```

**SDXL:**
```
model_channels: 320, context_dim: 2048
ADM conditioning (label_emb), linear attention
res_mult: 8
```

**SD3:**
```
joint transformer (separate text/image streams)
3 text encoders: CLIP + CLIP + T5
res_mult: 16
```

### VAE Types

| Used By | VAE Type | Channels | Key Detail |
|---|---|---|---|
| SD15, SDXL | Standard KL | 4ch | scaling_factor=0.18215 |
| Flux 1, Chroma | Flux KL | 16ch | scaling_factor=0.3611 |
| SD3 | Standard KL | 16ch | — |
| Flux 2 Dev, Klein | Flux 2 VAE | 128ch (patchified) | Batch norm + pixel shuffle |
| ZImage | Standard KL | — | — |

### Flux 2 vs Flux 1 Differences
- **text_id_dim**: 4 (Flux 2) vs 3 (Flux 1)
- **Patchify**: Flux 2 does 2x2 spatial patchification of latents
- **BN normalization**: Flux 2 normalizes patchified latents using VAE batch norm stats
- **Text encoder**: Flux 2 Dev = Mistral (single), Klein = Qwen3 (single). Flux 1 = CLIP-L + T5-XXL (dual)
- **Klein**: No guidance embeddings, different block counts/dims

---

## TEAM STRUCTURE — 4 BUILDERS + 1 SKEPTIC

### Agent 1: Model Architect
**Scope:** `models/` — Pure PyTorch model definitions

**Deliverables:**
1. `models/unet.py` — UNet with LDM key names for SD15/SDXL/SDXL-Refiner
2. `models/flux_dit.py` — Flux 1 DiT (19+38 blocks, BFL key format)
3. `models/flux2_dit.py` — Flux 2 DiT configurable for Dev and Klein variants
4. `models/sd3_transformer.py` — SD3 joint transformer
5. `models/chroma_dit.py` — Chroma DiT (Flux variant, T5-only, no guidance)
6. `models/zimage_dit.py` — ZImage DiT
7. `models/vae.py` — Standard KL VAE (4ch and 16ch variants)
8. `models/vae_flux2.py` — Flux 2 VAE with batch norm + pixel shuffle

**Rule:** Every model's parameter names must match the checkpoint keys exactly. If `load_state_dict(checkpoint)` has missing keys, the model definition is wrong.

**Reference:** Read `/home/alex/serenity/serenity/models/flux2_klein.py`, `flux2.py`, `flux1.py`, `sd15.py`, `sdxl.py`, `sd3.py`, `chroma.py`, `zimage.py` for architecture constants. Also study ComfyUI source for native implementations: `comfy/ldm/flux/model.py`, `comfy/ldm/modules/diffusionmodules/openaimodel.py`.

### Agent 2: Pipeline Engineer
**Scope:** `engine/` — Loading, conditioning, denoising loop

**Deliverables:**
1. `engine/loader.py` — Checkpoint loading + `extract_submodel()` prefix stripping
2. `engine/conditioning.py` — Per-architecture conditioning builder (replaces broken hardcoded kwargs)
3. `engine/denoise.py` — Denoise function that calls `conditioning.prepare()` then `model(**kwargs)`
4. `engine/pipeline.py` — Main `generate()` function: load → encode text → prepare latents → denoise → decode VAE → return image
5. `config.py` — Inference config dataclass

**Critical fix:** The old engine hardcoded `model(inp, timestep, encoder_hidden_states=cond)` for every architecture. Each architecture needs different kwargs:
- Flux: `encoder_hidden_states`, `pooled_projections`, `img_ids`, `txt_ids`, `timestep`, `guidance`
- SDXL: `encoder_hidden_states`, `added_cond_kwargs` (text_embeds, time_ids)
- SD15: `encoder_hidden_states`
- Klein: Qwen3 embeddings (stacked), no guidance

### Agent 3: Encoder Specialist
**Scope:** `text/` — Text encoders and their routing

**Deliverables:**
1. `text/clip.py` — CLIP-L and CLIP-G (uses `transformers` — this is fine)
2. `text/t5.py` — T5-XXL encoder
3. `text/qwen3.py` — Qwen3 encoder for Klein: load model, extract hidden states at layers [9,18,27], stack → joint_attention_dim
4. `text/mistral.py` — Mistral encoder for Flux 2 Dev
5. `text/qwen2.py` — Qwen2 causal LM encoder for ZImage: hidden_states[-2], different from Qwen3
6. `text/manager.py` — Routes architecture → correct encoder(s), returns embeddings in expected format

**Critical:** Klein Qwen3 stacking produces:
- 4B: 3 layers * 2560 hidden = 7680 joint_attention_dim
- 9B: 3 layers * 4096 hidden = 12288 joint_attention_dim
ZImage Qwen2 is a completely different encoder (causal LM, hidden_states[-2], dim=3840). Do NOT confuse them.

### Agent 4: UI Developer
**Scope:** `ui/` — Desktop application for inference

**Deliverables:**
1. Desktop GUI (NOT web — no Flask, no Gradio, no Streamlit)
2. Use Qt (PySide6/PyQt6) or similar native desktop toolkit
3. Features:
   - Model selection (browse for .safetensors checkpoint)
   - Prompt input
   - Generation parameters (steps, CFG, width, height, seed)
   - Generate button → display result image
   - Progress bar during generation
   - Save output image
4. Clean separation: UI calls `engine/pipeline.py` generate() function
5. No business logic in the UI — it's just a frontend

### Agent 5: Skeptic (see INFERENCE_SKEPTIC_PROMPT.md)
Runs after every change. Verifies with real models. Blocks fake fixes. Reports evidence.

---

## PRIORITY ORDER

Build bottom-up, verify each layer before moving up:

1. **Models first** (Agent 1) — Get pure PyTorch models that load real checkpoints with 0 missing keys
2. **Encoders next** (Agent 3) — Get text encoders producing correct-shaped embeddings
3. **Pipeline** (Agent 2) — Wire models + encoders into generate() pipeline
4. **Verify end-to-end** (Agent 5) — Generate real images for each architecture
5. **UI last** (Agent 4) — Only after the engine works

### Minimum Viable Target
Get ONE architecture working end-to-end first. Recommended order:
1. **SD 1.5** — Simplest (single CLIP, standard UNet, standard VAE)
2. **Flux 2 Klein 4B** — Most important to the user, smallest Flux variant
3. **SDXL** — Adds dual CLIP + ADM conditioning
4. **Flux 1 Dev** — Adds dual encoder + img_ids/txt_ids
5. **Everything else** — SD3, Chroma, ZImage, Flux 2 Dev, etc.

---

## WHAT TO COPY FROM SERENITY (WORKING — DON'T REWRITE)

These subsystems from `/home/alex/serenity/serenity/inference/` are standalone, diffusers-free, and work correctly. Copy them into the new project:

- `sampling/` — Samplers, schedulers, prediction types, CFG
- `vae/decoder.py` — Tiled decoding logic
- `attention/` — Backend selection (SDP, Flash, xFormers, SAGE)
- `memory/` — Offload, streams, pinned memory, VRAM budget
- `quantization/` — Quantized layer factories
- `lora/` — LoRA loading and merging
- `cache/` — Text encoding cache
- `models/detection.py` — Architecture detection from checkpoint keys (may need updates for Flux 2 Dev enum)

---

## VERIFICATION CRITERIA — IT WORKS WHEN:

1. Load a real SD 1.5 checkpoint → generate a recognizable image
2. Load a real SDXL checkpoint → generate a recognizable image
3. Load a real Flux 1 Dev checkpoint → generate with CLIP+T5 → recognizable image
4. Load a real Klein 4B checkpoint → generate with Qwen3 → recognizable image
5. Load a real Flux 2 Dev checkpoint → generate with Mistral → recognizable image
6. Load a real SD3 checkpoint → generate with CLIP+CLIP+T5 → recognizable image
7. Load a real Chroma checkpoint → generate with T5 → recognizable image
8. Zero `from diffusers` imports anywhere in the project
9. Zero or near-zero missing keys on every model load
10. Desktop UI can select a model, type a prompt, generate, and display the result
11. Serenity training is completely untouched and still passes all tests

**"Recognizable image"** means: not blank, not noise, shows content related to the prompt. std between 0.05-0.5, visually coherent when saved to PNG.
