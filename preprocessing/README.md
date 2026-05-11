# `preprocessing/` — Dataset preprocessing utilities

These scripts parse the raw multithreaded trace files released alongside
the **AutoThread** dataset (`PCS/`, `UAV/`, `Dynamic/`) and turn them
into the tabular feature matrices used to train the PINO predictor.

| File | Purpose |
|------|---------|
| `create_data.py`           | Core parser for `pidstat_out_*.txt`, `simulation_out_*.txt`, `perf_out_*.txt`; builds per-run time-aligned tensors. |
| `writer_allinf_PCS.py`     | Aggregate per-run features into one row per (round × method × complexity × thread) for the **PCS** application. |
| `writer_allinf_Airline.py` | Same as above for the **UAV / Airline** application. |
| `thread_siminf.py`         | Build the "thread sweep" summary tables (one row per worker-thread count) used by the static-scaling figures in the paper. |
| `read_vtunReport.py`       | Parse Intel VTune hotspot reports (`Elapsed_Time`, `Spin_Time`, etc.) into a DataFrame. |

All scripts read directly from the raw `pidstat_out_K.txt`,
`simulation_out_K.txt`, and `perf_out_K.txt` files; the trace-file
schema is documented in the dataset README.
