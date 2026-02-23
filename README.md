# Neural Residual Cloud Project

This repo contains two main components:

1. **Visualization (viz/)** — a Vite + WebGPU viewer for exploring 3D volumes, procedural models, and learned residuals.
2. **Training (train/)** — a PyTorch project that:
   - loads Disney Animation cloud data
   - computes procedural baseline
   - trains an MLP residual model
   - exports weights and metadata for runtime usage

---

## 📁 Repository Structure

```
cloud-neural-residuals/
README.md
AGENTS.md

data/
raw/
processed/

viz/
package.json
vite.config.ts
src/
main.ts
webgpu/

train/
pyproject.toml
src/
cloudtrain/
```

---

## 🧠 Objectives

This project will help you:

- Learn how to train a neural residual field for volumetric clouds
- Export network weights to GPU-friendly formats
- Visualize GT volumes, procedural baselines, residuals, and reconstructions
- Build a WebGPU viewer to inspect volumes interactively

---

## 🚀 Setup Instructions

### Visualization

```bash
cd viz
npm install
npm run dev
```

This starts a local server with your WebGPU viewer.

---

### Training

Install dependencies (recommended in virtual environment):

```bash
cd train
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

To train the residual model:

```bash
python -m cloudtrain.train \
    --data-dir ../data/processed \
    --output-dir outputs \
    --config configs/default.yaml
```

---

## 📦 Data Preparation

1. Clone the Disney cloud dataset (via Datalad or git):

   ```bash
   datalad install -r https://github.com/datalad-datasets/disneyanimation-cloud
   ```
2. Convert or normalize volumes into a consistent coordinate system.
3. Place processed volumes in `data/processed`.

---

## ⚙️ Training Flow

1. **Load GT volume**
2. **Compute procedural baseline**
3. **Sample random 3D points**
4. **Train MLP residual**
5. **Export weights + metadata**

---

## 🧪 Visualization Modes

In the viewer, you can toggle:

* `GT volume`
* `procedural baseline`
* `residual only`
* `baseline + residual`
* Differences between GT and recon

Slice views and volume rendering are supported.

---

## 🧩 Runtime Integration

Weights and metadata exported from training are consumed by:

* WebGPU compute for inference
* Volume baking
* Raymarch rendering

Weight blob + metadata JSON describes MLP architecture and encodings.

---

## 📌 File Formats

| Purpose      | Type         | Notes                       |
| ------------ | ------------ | --------------------------- |
| Volume       | .bin + .json | Raw R16F/R32F + metadata    |
| MLP weights  | .bin         | FP16 weight blob            |
| MLP metadata | .json        | Input dims, layers, offsets |
