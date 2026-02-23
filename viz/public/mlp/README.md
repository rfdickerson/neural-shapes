Place exported training artifacts here so the WebGPU renderer can load them at startup:

- `residual_mlp_metadata.json`
- `residual_mlp_weights.bin`

Example from repo root:

```bash
python3 train/copy_mlp_to_viz.py
```

Equivalent manual copy:

```bash
mkdir -p viz/public/mlp
cp train/outputs/residual_mlp_metadata.json viz/public/mlp/
cp train/outputs/residual_mlp_weights.bin viz/public/mlp/
```
