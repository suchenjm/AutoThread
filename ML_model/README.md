# `ML_model/` — Pretrained PINO weights

| File | Description |
|------|-------------|
| `model_mutilhead_1.9.pth` | **PINO checkpoint** loaded by `python_algorithm/PINN.py`. Cross-attention multi-output-head architecture matching Sec. 3.1 of the paper. Trained by `PINO/PINN_mutilhead.ipynb`. Validation MAE = 1.9 threads. |
| `feature_scaler.pkl`      | `sklearn` `StandardScaler` for the 4-dim hardware feature vector `(user_cpu, system_cpu, VSZ, cswch)`. |

The C++ scheduler (`scheduler/ScheduleManager.cpp`) hard-codes these paths:

```
../ML_model/model_mutilhead_1.9.pth
../ML_model/feature_scaler.pkl
```

If you retrain the model from scratch, drop the freshly produced
`.pth` and `feature_scaler.pkl` here.
