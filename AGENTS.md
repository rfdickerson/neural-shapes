# Agents and Responsibilities

This file describes roles for automated code generation (AI assistants, Copilot, ChatGPT, etc).
Use these as prompts to guide code scaffolding and implementation.

---

## TRAINING AGENT

**Purpose:** Generate training code for PyTorch cloud residual model.

### Responsibilities

- Create dataset loader for Disney cloud volumes
- Implement procedural baseline in Python
- Implement positional encoding (Fourier features)
- Define a small MLP (configurable layers)
- Write training loop (sampling, loss, batching)
- Add evaluation code and visualization utilities
- Write export script to output:
  - weight `.bin` files
  - metadata `.json` file

### Training Config Schema

`configs/*.yaml` should include:

```yaml
dataset:
  path: str
  sample_count: int
  sampling_strategy: str
mlp:
  layers: [int, int, ...]
  activation: str
  encoding:
    type: fourier
    levels: int
optimizer:
  type: adam
  lr: float
loss:
  type: mse
  clip_range: [float, float]
export:
  weight_dtype: fp16
  output_dir: str
```

---

## VIZ AGENT

**Purpose:** Create a WebGPU viewer that loads volumes, weights, and renders them using raymarch.

### Responsibilities

* Bootstrap Vite + WebGPU
* Write shader modules:

  * raymarcher
  * blit/quad
  * (optional) MLP inference
* Implement volume loaders and metadata binding
* Build UI toggles for:

  * show GT
  * show baseline
  * show residual
  * difference mode
* Provide camera controls
* Provide slice view tools

---

## SHADER AGENT

**Purpose:** Generate WGSL or Slang shaders for:

* volume sampling
* raymarch
* MLP inference kernels
* procedural SDF baseline (if needed)

### Responsibilities

* Define shared structs for metadata
* Bind group layouts for:

  * volume textures
  * weight buffer
  * push constants
* Implement positional encoding in WGSL
* Implement tiny MLP in WGSL

---

## EXPORT AGENT

**Purpose:** Translate PyTorch model to GPU-friendly blobs.

### Responsibilities

* Serialize weights to FP16 binary
* Compute offsets for each layer and bias
* Write metadata JSON with:

  * input dims
  * layers
  * weight offsets
  * encoding params
* Validate sizes

---

## TESTING & UTILS AGENT

**Purpose:** Create small utilities to support testing and data handling.

### Responsibilities

* Volume file readers/writers
* Binary I/O converters
* Data normalization helpers
* Sampling utilities (CPU & GPU)
* CLI scripts for debugging volumes
