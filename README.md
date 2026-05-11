# AutoThread

> Reference implementation accompanying *Hybrid-Adaptive Thread Tuning to
> Mitigate Simulation Execution Bottlenecks in High-Performance
> Reinforcement Learning Inference* (AutoThread, IJCAI/ECAI 2026).

AutoThread is a hybrid-adaptive worker-thread tuner for high-fidelity
Discrete Event Simulation (DES) used inside Simulation-in-the-Loop
Reinforcement Learning. It couples a **Physics-Informed Neural Operator
(PINO)** that predicts the optimal worker-thread count from current
workload / hardware features, with a **load-aware online fine-tuner**
that smoothly tracks workload shifts between PINO calls.

This repository contains:

- the integration of the PINO predictor in the C++ scheduler;
- the PINO Python inference module that the scheduler calls via `pybind11`;
- the pretrained PINO checkpoints;
- the Jupyter notebooks that train them;
- the dataset preprocessing utilities that turn raw multithreaded
  traces into PINO training tables.

The companion **dataset** (PCS / UAV / Dynamic) is released separately;
see its `README.md` for the trace-file schema.

---

## Repository layout

```
AutoThread/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── scheduler/                  # C++ scheduler integration (reference)
│   ├── ScheduleManager.h       # PINO-only fields and methods
│   └── ScheduleManager.cpp     # AutoThread (dPINN) loop only
│
├── python_algorithm/           # Python inference module (called from C++)
│   ├── __init__.py
│   └── PINN.py                 # PINO operator + singleton predictor
│
├── ML_model/                   # Pretrained PINO weights
│   ├── model_mutilhead_1.9.pth
│   ├── feature_scaler.pkl
│   └── README.md
│
├── PINO/                       # PINO training notebook
│   ├── PINN_mutilhead.ipynb
│   └── README.md
│
└── preprocessing/              # Dataset preprocessing scripts
    ├── create_data.py
    ├── writer_allinf_PCS.py
    ├── writer_allinf_Airline.py
    ├── thread_siminf.py
    ├── read_vtunReport.py
    └── README.md
```

## How the pieces fit together

```
                ┌────────────────────────────────────────┐
                │   simulation main loop (C++)           │
                │   scheduler/ScheduleManager.cpp        │
                │                                        │
                │   1. workload-shift trigger ───────────┼───────┐
                │   2. invoke PINO via pybind11          │       │
                │   3. load-aware online fine-tune       │       ▼
                └────────────────────────────────────────┘   ┌────────────┐
                                                             │  PINO      │
                                                             │  python_   │
                                                             │  algorithm │
                                                             │  /PINN.py  │
                                                             └─────┬──────┘
                                                                   │ loads
                                                                   ▼
                                                             ┌─────────────┐
                                                             │  ML_model/  │
                                                             │  .pth/.pkl  │
                                                             └─────────────┘
```

## Requirements

```bash
pip install -r requirements.txt
```

The C++ side additionally requires `pybind11` (header-only) and a
working `python3-dev`. Tested on Linux (Ubuntu 22.04, kernel 6.8) with
AMD EPYC 9734 and Intel Xeon Gold 6338.

## Quick start (PINO predictor only)

```python
from python_algorithm import initialize_model, pinn_predict

initialize_model(
    model_path="ML_model/model_mutilhead_1.9.pth",
    scaler_path="ML_model/feature_scaler.pkl",
)

# N = pending tasks, mu_e = 1 / event_duration, k = current threads
# hw = [user_cpu, system_cpu, VSZ, cswch]
k_opt = pinn_predict(N=800, mu_e=0.00222, k=6,
                     hardware_params=[580.27, 21.72, 684540, 1972.19])
print(f"Recommended worker-thread count: {k_opt}")
```

## Reproducing the PINO checkpoint

1. Run the preprocessing pipeline in `preprocessing/` against the
   released dataset to produce per-run feature CSVs.
2. Open `PINO/PINN_mutilhead.ipynb` and run all cells.
3. Drop the generated `model_mutilhead_1.9.pth` and `feature_scaler.pkl`
   into `ML_model/`.

## Citation

```bibtex

```

## License

Released under the MIT License — see `LICENSE` for details.
