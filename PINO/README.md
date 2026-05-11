# `PINO/` — PINO operator training notebook

This Jupyter notebook reproduces the PINO checkpoint shipped in
`../ML_model/`.

| Notebook | Output checkpoint | Description |
|----------|-------------------|-------------|
| `PINN_mutilhead.ipynb` | `model_mutilhead_1.9.pth` | Cross-attention multi-output-head PINO operator described in Sec. 3.1 of the AutoThread paper. The notebook implements the closed-form M/M/1/K performance model, the physics-informed loss `L = L_phy + λ · L_sup + L_reg`, and the Residual-Based Attention (RBA) re-weighting of the physical loss. |

The notebook is self-contained:

1. Loads the pre-processed CSVs produced by `preprocessing/writer_allinf_*.py`.
2. Trains the PINO operator with the physics-informed objective.
3. Saves `model_mutilhead_1.9.pth` and `feature_scaler.pkl` into `../ML_model/`.
