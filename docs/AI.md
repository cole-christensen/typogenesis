# AI Feature Implementation Plan - Parallel Workstreams

## Summary
Implement real AI/ML features for Typogenesis to replace the current placeholder/fallback implementations. This plan organizes work into 7 parallel workstreams that can be executed simultaneously using subagents.

---

## Current State Analysis

### Existing Infrastructure
The app has well-structured AI service stubs ready for real models:

| Service | Current Implementation | Target |
|---------|----------------------|--------|
| **GlyphGenerator** | Template-based geometric shapes | Diffusion model generating real glyphs |
| **StyleEncoder** | Geometric analysis (stroke weight, contrast, etc.) | CNN encoder producing style embeddings |
| **KerningPredictor** | Edge distance heuristics | Siamese CNN predicting optimal kerning |
| **ModelManager** | CoreML loading infrastructure (ready) | Load and manage trained models |

### Key Files
- `Typogenesis/Services/AI/GlyphGenerator.swift` - Lines 65-180 contain fallback generation
- `Typogenesis/Services/AI/StyleEncoder.swift` - Lines 420-435 contain placeholder embedding
- `Typogenesis/Services/AI/KerningPredictor.swift` - Lines 45-120 contain heuristic prediction
- `Typogenesis/Services/AI/ModelManager.swift` - Already has CoreML loading infrastructure
- `Typogenesis/Services/AI/GlyphTemplates.swift` - Template shapes (keep as fallback)
- `Typogenesis/Services/AI/StrokeBuilder.swift` - Stroke generation utilities

---

## Model Architectures

### GlyphDiffusion (Flow-Matching Diffusion)

**Purpose:** Generate glyph images conditioned on character and style

**Architecture:**
```
┌─────────────────────────────────────────────────────────────────┐
│                      GlyphDiffusion Model                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Inputs:                                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Noise Tensor │  │  Character   │  │    Style     │          │
│  │  (64x64x1)   │  │  Embedding   │  │  Embedding   │          │
│  │              │  │   (26-dim)   │  │  (128-dim)   │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                   │
│         └────────────┬────┴────────────────┘                   │
│                      ▼                                          │
│         ┌────────────────────────┐                             │
│         │   Conditioning Layer   │                             │
│         │   (FiLM / AdaIN)       │                             │
│         └───────────┬────────────┘                             │
│                     ▼                                          │
│         ┌────────────────────────┐                             │
│         │       UNet Core        │                             │
│         │  ┌─────────────────┐   │                             │
│         │  │ Encoder (Conv)  │   │                             │
│         │  │  64→128→256     │   │                             │
│         │  └────────┬────────┘   │                             │
│         │           ▼            │                             │
│         │  ┌─────────────────┐   │                             │
│         │  │   Bottleneck    │   │                             │
│         │  │   (Attention)   │   │                             │
│         │  └────────┬────────┘   │                             │
│         │           ▼            │                             │
│         │  ┌─────────────────┐   │                             │
│         │  │ Decoder (Conv)  │   │                             │
│         │  │  256→128→64     │   │                             │
│         │  └─────────────────┘   │                             │
│         └───────────┬────────────┘                             │
│                     ▼                                          │
│         ┌────────────────────────┐                             │
│         │  Output: Glyph Image   │                             │
│         │      (64x64x1)         │                             │
│         └────────────────────────┘                             │
│                                                                 │
│  Training: Flow-Matching (OT-CFM)                              │
│  - Straight paths between noise and data                       │
│  - Single-step or few-step inference                           │
│  - Loss: MSE on velocity field                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Post-Processing Pipeline:**
```
Generated Image → Threshold → Contour Trace → Bezier Fit → GlyphOutline
       │              │            │              │
       ▼              ▼            ▼              ▼
    64x64         Binary        Points[]      PathPoints[]
   grayscale      mask          (x,y)       with handles
```

**Training Configuration:**
- Input resolution: 64x64 (can upscale to 128x128)
- Batch size: 64
- Learning rate: 1e-4 with cosine decay
- Training steps: 100K-500K
- Optimizer: AdamW
- Hardware: Apple Silicon MPS or CUDA GPU

---

### StyleEncoder (Contrastive CNN)

**Purpose:** Extract 128-dimensional style embeddings from glyph images

**Architecture:**
```
┌─────────────────────────────────────────────────────────────────┐
│                      StyleEncoder Model                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input:                                                         │
│  ┌──────────────────┐                                          │
│  │   Glyph Image    │                                          │
│  │    (64x64x1)     │                                          │
│  └────────┬─────────┘                                          │
│           ▼                                                     │
│  ┌──────────────────────────────────────┐                      │
│  │         ResNet-18 Backbone           │                      │
│  │  ┌──────────┐  ┌──────────┐         │                      │
│  │  │ Conv 7x7 │→ │ MaxPool  │         │                      │
│  │  │  64 ch   │  │   2x2    │         │                      │
│  │  └──────────┘  └────┬─────┘         │                      │
│  │                     ▼               │                      │
│  │  ┌──────────────────────────┐       │                      │
│  │  │   ResBlock × 2 (64 ch)   │       │                      │
│  │  └───────────┬──────────────┘       │                      │
│  │              ▼                      │                      │
│  │  ┌──────────────────────────┐       │                      │
│  │  │   ResBlock × 2 (128 ch)  │       │                      │
│  │  └───────────┬──────────────┘       │                      │
│  │              ▼                      │                      │
│  │  ┌──────────────────────────┐       │                      │
│  │  │   ResBlock × 2 (256 ch)  │       │                      │
│  │  └───────────┬──────────────┘       │                      │
│  │              ▼                      │                      │
│  │  ┌──────────────────────────┐       │                      │
│  │  │   ResBlock × 2 (512 ch)  │       │                      │
│  │  └──────────────────────────┘       │                      │
│  └──────────────────┬───────────────────┘                      │
│                     ▼                                          │
│  ┌──────────────────────────────────────┐                      │
│  │       Global Average Pooling         │                      │
│  │            512 → 512                 │                      │
│  └──────────────────┬───────────────────┘                      │
│                     ▼                                          │
│  ┌──────────────────────────────────────┐                      │
│  │         Projection Head              │                      │
│  │   FC(512→256) → ReLU → FC(256→128)  │                      │
│  │            + L2 Normalize            │                      │
│  └──────────────────┬───────────────────┘                      │
│                     ▼                                          │
│  ┌──────────────────────────────────────┐                      │
│  │      Output: Style Embedding         │                      │
│  │           (128-dim, L2 norm)         │                      │
│  └──────────────────────────────────────┘                      │
│                                                                 │
│  Training: Contrastive Learning (NT-Xent / SimCLR)             │
│  - Positive pairs: Same font, different glyphs                 │
│  - Negative pairs: Different fonts                             │
│  - Temperature: 0.07                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Contrastive Loss:**
```
L = -log( exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ) )

where:
  z_i, z_j = embeddings of same-font glyphs (positive pair)
  z_k = embeddings from different fonts (negatives)
  τ = temperature (0.07)
  sim() = cosine similarity
```

---

### KerningNet (Siamese CNN)

**Purpose:** Predict optimal kerning value for glyph pairs

**Architecture:**
```
┌─────────────────────────────────────────────────────────────────┐
│                       KerningNet Model                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Inputs:                                                        │
│  ┌──────────────┐           ┌──────────────┐                   │
│  │  Left Glyph  │           │ Right Glyph  │                   │
│  │   (64x64)    │           │   (64x64)    │                   │
│  └──────┬───────┘           └──────┬───────┘                   │
│         │                          │                            │
│         ▼                          ▼                            │
│  ┌────────────────────────────────────────┐                    │
│  │         Shared CNN Encoder             │                    │
│  │  (weights shared between both paths)   │                    │
│  │                                        │                    │
│  │  Conv(1→32, 3x3) → BN → ReLU → Pool   │                    │
│  │  Conv(32→64, 3x3) → BN → ReLU → Pool  │                    │
│  │  Conv(64→128, 3x3) → BN → ReLU → Pool │                    │
│  │  Conv(128→256, 3x3) → BN → ReLU → GAP │                    │
│  │                                        │                    │
│  │  Output: 256-dim embedding per glyph  │                    │
│  └────────────────┬───────────────────────┘                    │
│                   │                                             │
│      ┌────────────┴────────────┐                               │
│      ▼                         ▼                               │
│  ┌────────┐               ┌────────┐                           │
│  │ 256-d  │               │ 256-d  │                           │
│  │  Left  │               │ Right  │                           │
│  └───┬────┘               └───┬────┘                           │
│      │                        │                                 │
│      └────────┬───────────────┘                                │
│               ▼                                                 │
│  ┌──────────────────────────────────────┐                      │
│  │        Concatenate + Metrics         │                      │
│  │     [left_emb, right_emb, metrics]   │                      │
│  │           512 + 4 = 516 dim          │                      │
│  └──────────────────┬───────────────────┘                      │
│                     ▼                                          │
│  ┌──────────────────────────────────────┐                      │
│  │          Regression Head             │                      │
│  │   FC(516→256) → ReLU → Dropout(0.3) │                      │
│  │   FC(256→64) → ReLU → FC(64→1)      │                      │
│  └──────────────────┬───────────────────┘                      │
│                     ▼                                          │
│  ┌──────────────────────────────────────┐                      │
│  │      Output: Kerning Value           │                      │
│  │   (normalized, scale to UPM units)   │                      │
│  └──────────────────────────────────────┘                      │
│                                                                 │
│  Metrics input (normalized):                                   │
│  - Left glyph advance width / UPM                              │
│  - Right glyph left side bearing / UPM                         │
│  - x-height ratio                                              │
│  - Cap height ratio                                            │
│                                                                 │
│  Training: MSE Loss on kerning values                          │
│  Ground truth: Extracted from professional fonts               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Parallel Workstreams

### Workstream A: Training Data Pipeline
**Purpose:** Build dataset preparation infrastructure for all models

**Deliverables:**
```
scripts/
├── data/
│   ├── download_fonts.py      # Download open-source fonts
│   ├── extract_glyphs.py      # Extract glyph images + outlines
│   ├── generate_pairs.py      # Generate kerning pair images
│   └── augment_data.py        # Data augmentation pipeline
├── datasets/
│   ├── glyph_dataset.py       # PyTorch dataset for glyphs
│   ├── style_dataset.py       # Dataset for style pairs
│   └── kerning_dataset.py     # Dataset for kerning pairs
└── requirements.txt           # Python dependencies
```

**Data Sources:**
- Google Fonts (2000+ open fonts)
- Open Font Library
- Font Squirrel (with licenses)
- Custom handwriting samples

**Target Dataset Sizes:**
- GlyphDiffusion: 500K+ glyph images (64x64, 128x128)
- StyleEncoder: 100K+ font-glyph pairs with style labels
- KerningNet: 1M+ kerning pair images with spacing values

---

### Workstream B: GlyphDiffusion Model
**Purpose:** Train diffusion model for glyph generation

**Deliverables:**
```
scripts/
├── models/
│   └── glyph_diffusion/
│       ├── model.py           # UNet + conditioning
│       ├── noise_schedule.py  # Flow-matching schedule
│       ├── train.py           # Training loop
│       └── sample.py          # Sampling/inference
```

---

### Workstream C: StyleEncoder Model
**Purpose:** Train CNN to extract style embeddings from glyphs

**Deliverables:**
```
scripts/
├── models/
│   └── style_encoder/
│       ├── model.py           # CNN encoder
│       ├── losses.py          # Contrastive losses
│       ├── train.py           # Training loop
│       └── embed.py           # Embedding extraction
```

---

### Workstream D: KerningNet Model
**Purpose:** Train model to predict optimal kerning values

**Deliverables:**
```
scripts/
├── models/
│   └── kerning_net/
│       ├── model.py           # Siamese CNN
│       ├── train.py           # Training loop
│       └── predict.py         # Inference
```

---

### Workstream E: Model Hosting Infrastructure
**Purpose:** Set up optional cloud inference for users without local GPU

**Options:**
1. **Ollama Integration** (already mentioned in codebase)
   - Package models as Ollama modelfiles
   - Local or remote inference

2. **HuggingFace Spaces**
   - Host models on HF
   - API calls for inference

3. **CoreML-only** (used in v1)
   - All models run locally on Apple Silicon
   - No cloud dependency

**Deliverables:**
- API client in `Typogenesis/Services/AI/CloudInference.swift`
  - **NOTE:** This file exists as infrastructure for future versions but is NOT integrated in v1.
  - Contains stubbed `OllamaProvider` and `HuggingFaceProvider` implementations.
  - To integrate: wire `CloudInferenceManager` into `GlyphGenerator`, `StyleEncoder`, `KerningPredictor`.
- Model download/update system in `ModelManager.swift`

---

### Workstream F: CoreML Conversion
**Purpose:** Convert trained PyTorch models to CoreML

**Pipeline:**
```
PyTorch (.pt) → ONNX (.onnx) → CoreML (.mlpackage)
```

**Deliverables:**
```
scripts/
├── convert/
│   ├── convert_glyph_diffusion.py
│   ├── convert_style_encoder.py
│   ├── convert_kerning_net.py
│   └── verify_conversion.py    # Numerical accuracy tests

Typogenesis/Resources/Models/
├── GlyphDiffusion.mlpackage
├── StyleEncoder.mlpackage
└── KerningNet.mlpackage
```

**Conversion Requirements:**
- coremltools >= 7.0
- onnx >= 1.14
- Input/output shapes must be static or use flexible shapes
- Test numerical accuracy: PyTorch vs CoreML < 1e-4 difference

---

### Workstream G: App Integration & Testing
**Purpose:** Wire trained models into existing Swift infrastructure

**Files to Modify:**

1. **GlyphGenerator.swift**
   - Add `generateWithModel()` using CoreML
   - Keep `generateGeometric()` as fallback
   - Add image → outline post-processing

2. **StyleEncoder.swift**
   - Replace placeholder `encodeImage()` with real inference
   - Cache embeddings for performance

3. **KerningPredictor.swift**
   - Replace heuristic with model inference
   - Batch prediction for efficiency

4. **ModelManager.swift**
   - Model versioning and updates
   - Download progress UI
   - Graceful fallback when models unavailable

**Tests to Add:**
```
TypogenesisTests/
├── GlyphDiffusionIntegrationTests.swift
├── StyleEncoderIntegrationTests.swift
├── KerningNetIntegrationTests.swift
└── ModelManagerIntegrationTests.swift
```

---

## Execution Timeline (Parallel)

```
Week 1-2:  [A] Data pipeline ─────────────────────────────────────►
           [E] Infrastructure planning ──────►

Week 2-4:  [B] GlyphDiffusion training ──────────────────────────►
           [C] StyleEncoder training ────────────────────────────►
           [D] KerningNet training ──────────────────────────────►

Week 4-5:  [F] CoreML conversion ────────────────────────────────►

Week 5-6:  [G] App integration ──────────────────────────────────►
           [G] Testing & refinement ─────────────────────────────►
```

---

## Subagent Assignment

When implementing, launch these subagents in parallel:

```
# Phase 1: Data & Infrastructure (Week 1-2)
Agent 1: Workstream A - Training data pipeline
Agent 2: Workstream E - Infrastructure setup

# Phase 2: Model Training (Week 2-4) - can run in parallel if hardware allows
Agent 3: Workstream B - GlyphDiffusion training
Agent 4: Workstream C - StyleEncoder training
Agent 5: Workstream D - KerningNet training

# Phase 3: Integration (Week 4-6)
Agent 6: Workstream F - CoreML conversion
Agent 7: Workstream G - App integration
```

---

## Verification

### Model Quality Tests
1. **GlyphDiffusion**: Generate 26 lowercase letters, visually inspect for:
   - Consistent stroke weight
   - Proper character recognition
   - Style adherence to conditioning

2. **StyleEncoder**:
   - Same-font similarity > 0.9
   - Different-font similarity < 0.5
   - Style interpolation produces smooth transitions

3. **KerningNet**:
   - Predict negative values for AV, AT, LT, etc.
   - Predict ~0 for HH, II, nn, etc.
   - Correlation with professional font kerning > 0.8

### Integration Tests
1. Build: `xcodebuild -scheme Typogenesis build`
2. Run tests: `xcodebuild test -scheme Typogenesis`
3. Manual test: Generate a full alphabet, export as TTF, render in TextEdit

### Performance Targets
- Glyph generation: < 2s per glyph on M1
- Style extraction: < 100ms per glyph
- Kerning prediction: < 50ms per pair

---

## Dependencies

**Python (Training):**
- PyTorch >= 2.0
- torchvision
- coremltools >= 7.0
- onnx >= 1.14
- fonttools
- Pillow
- numpy

**Swift (App):**
- CoreML framework (already integrated)
- Metal Performance Shaders (already integrated)
- No new dependencies required

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Model quality insufficient | Keep geometric fallback, iterate on training |
| Training data licensing | Use only open-licensed fonts, document sources |
| CoreML conversion fails | Use simpler architectures, test early |
| Performance too slow | Optimize model size, use quantization |
| Apple Silicon only | Document requirements, consider cloud option |

---

## Files to Create

| File | Workstream | Description |
|------|------------|-------------|
| `scripts/data/*.py` | A | Data pipeline scripts |
| `scripts/models/glyph_diffusion/*.py` | B | Diffusion model code |
| `scripts/models/style_encoder/*.py` | C | Style encoder code |
| `scripts/models/kerning_net/*.py` | D | Kerning model code |
| `scripts/convert/*.py` | F | CoreML conversion |
| `Typogenesis/Resources/Models/*.mlpackage` | F | Trained models |
| `TypogenesisTests/*IntegrationTests.swift` | G | Integration tests |

## Files to Modify

| File | Workstream | Changes |
|------|------------|---------|
| `GlyphGenerator.swift` | G | Add real model inference |
| `StyleEncoder.swift` | G | Replace placeholder embedding |
| `KerningPredictor.swift` | G | Replace heuristic with model |
| `ModelManager.swift` | G | Model versioning, download UI |
