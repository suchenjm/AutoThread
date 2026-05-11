"""
PINO predictor -- online inference module called from the C++ scheduler
(ScheduleManager.cpp) via pybind11.

This is the production inference path of the PINO operator described in
the AutoThread paper (IJCAI/ECAI 2026, Sec. 3.1).  The architecture follows
the paper: a *cross-attention, multi-output-head* neural operator that

  * separately encodes the application parameters (N, mu_e, k_obs) and
    the four hardware features (usr_cpu, system_cpu, VSZ, cswch);
  * fuses them with a cross-attention block (hw_feat as query,
    param_feat as key/value);
  * decodes the fused representation with multiple heads that emit
    the optimal worker-thread count k* together with the contention
    coefficients (a1, b1) and (a2, b2) of the piece-wise scheduler
    service-rate model used by the physical loss.

Two C-callable entry points are exposed (kept stable across paper
revisions so the C++ side does not have to change):

    initialize_model(model_path, scaler_path)
    pinn_predict(N, mu_e, k, hardware_params)
"""

import joblib
import numpy as np
import torch
import torch.nn as nn


# Hardware-feature order MUST match the order used at training time
# (see PINO/PINN_mutilhead.ipynb : HARDWARE_FEATURES).
HARDWARE_FEATURES = ("usr_cpu", "system_cpu", "VSZ", "cswch")
HW_DIM     = len(HARDWARE_FEATURES)          # 4
INPUT_DIM  = 3 + HW_DIM                       # N, mu_e, k + hw_features


# ---------------------------------------------------------------------------
#  Physics-informed primitives  (Eq. 1 of the paper)
# ---------------------------------------------------------------------------
def calc_mu_b(k_transfor, k, a1, b1, a2, b2):
    """Piece-wise scheduler service rate mu_b(k) with sigmoid blending.

    The contention model from Sec. 3.1 is linear in k with two regimes
    that meet at k = k_transfor; we blend them softly so gradients stay
    well-defined during training.
    """
    sig_weight = torch.sigmoid(10 * (k_transfor.detach() - k))
    return sig_weight * (a1 * k + b1) + (1 - sig_weight) * (a2 * k + b2)


def calc_exec_time(N, k, mu_e, mu_b):
    """Closed-form per-cycle execution time (paper Eq. 1, two-branch form)."""
    ceil_val = torch.ceil(N / k)
    cond     = (mu_b > k * mu_e).float()
    t_case1  = ceil_val / mu_e + N / (mu_b - k * mu_e + 1e-8)
    t_case2  = N / mu_b + 1 / mu_e
    return cond * t_case1 + (1 - cond) * t_case2


# ---------------------------------------------------------------------------
#  Cross-attention block  (paper Sec. 3.1, Fig. AutoThread framework)
# ---------------------------------------------------------------------------
class CrossAttention(nn.Module):
    """Single-head cross attention used to fuse hardware and parameter feats."""

    def __init__(self, d_q: int, d_k: int, d_v: int, d_out: int):
        super().__init__()
        self.w_q   = nn.Linear(d_q, d_out, bias=False)
        self.w_k   = nn.Linear(d_k, d_out, bias=False)
        self.w_v   = nn.Linear(d_v, d_out, bias=False)
        self.scale = d_k ** -0.5

    def forward(self, query, key, value):
        # query / key / value : (B, L, d_model)
        q    = self.w_q(query)
        k    = self.w_k(key)
        v    = self.w_v(value)
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        return attn @ v


# ---------------------------------------------------------------------------
#  PINO operator  (cross-attention + multi-output heads)
# ---------------------------------------------------------------------------
class OptimalThreadsPredictor(nn.Module):
    """PINO operator G_theta : a -> (k*, a1, b1, a2, b2, T_pred).

    Input feature vector  ``a = [N, mu_e, k_obs, usr_cpu, system_cpu, VSZ, cswch]``
    matches the definition in Sec. 3.1 of the paper.

    Outputs (dict)
    --------------
    k_opt          : predicted optimal worker-thread count (>= 1)
    k_opt_phys     : physics-derived target  mu_b(k*) / (2 mu_e)
    k_transfor     : transition point  mu_b(k*) / mu_e
    a1, b1, a2, b2 : piece-wise contention coefficients of mu_b(k)
    mu_b_at_kopt   : mu_b evaluated at k_opt
    mu_b_at_thread : mu_b evaluated at the observed thread count
    time_at_thread : T_pred evaluated at the observed thread count
    """

    def __init__(self, input_dim: int = INPUT_DIM,
                 hw_dim: int = HW_DIM,
                 hidden_dim: int = 128,
                 d_attn: int = 64):
        super().__init__()
        self.hw_dim = hw_dim

        # 1) Hardware-feature encoder
        self.hw_encoder = nn.Sequential(
            nn.Linear(hw_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
        )
        # 2) Parameter encoder: (N, mu_e, thread)
        self.param_encoder = nn.Sequential(
            nn.Linear(3, 32), nn.LayerNorm(32), nn.ReLU(),
            nn.Linear(32, d_attn), nn.LayerNorm(d_attn), nn.ReLU(),
        )
        # 3) Cross attention: hw_feat (Q) x param_feat (K, V)
        self.cross_attn = CrossAttention(
            d_q=hidden_dim, d_k=d_attn, d_v=d_attn, d_out=d_attn,
        )
        # 4) Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(hidden_dim + d_attn, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
        )
        # 5) Multi-output heads
        self.head_k_opt = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.head_a1   = nn.Linear(hidden_dim, 1)
        self.head_b1   = nn.Linear(hidden_dim, 1)
        self.head_a2   = nn.Linear(hidden_dim, 1)
        self.head_b2   = nn.Linear(hidden_dim, 1)
        self.head_time = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> dict:
        # x : (B, INPUT_DIM)   layout = [N, mu_e, thread, hw_features...]
        N      = x[:, 0:1]
        mu_e   = x[:, 1:2]
        thread = x[:, 2:3]
        hw     = x[:, 3:3 + self.hw_dim]

        # Encode
        hw_feat    = self.hw_encoder(hw)                                       # (B, hidden)
        param_feat = self.param_encoder(torch.cat([N, mu_e, thread], dim=1))  # (B, d_attn)

        # Cross-attention fusion (treat each as a length-1 sequence)
        hw_seq    = hw_feat.unsqueeze(1)
        param_seq = param_feat.unsqueeze(1)
        fused     = self.cross_attn(hw_seq, param_seq, param_seq).squeeze(1)
        fused     = torch.cat([hw_feat, fused], dim=1)                        # (B, hidden+d_attn)

        shared = self.backbone(fused)                                          # (B, hidden)

        # Multi-output heads
        k_raw     = self.head_k_opt(shared)
        k_opt     = torch.exp(k_raw) + 1.0          # guarantee k_opt > 1
        a1        = self.head_a1(shared)
        b1        = self.head_b1(shared)
        a2        = self.head_a2(shared)
        b2        = self.head_b2(shared)

        # Physics-informed derived quantities (paper Eq. 1)
        mu_b_at_kopt   = a1 * k_opt + b1
        k_opt_phys     = mu_b_at_kopt / (2 * mu_e)
        k_transfor     = mu_b_at_kopt / mu_e
        mu_b_at_thread = calc_mu_b(k_transfor, thread, a1, b1, a2, b2)
        time_at_thread = calc_exec_time(N, thread, mu_e, mu_b_at_thread)

        return {
            "k_opt":          k_opt,
            "k_opt_phys":     k_opt_phys,
            "k_transfor":     k_transfor,
            "a1": a1, "b1": b1, "a2": a2, "b2": b2,
            "mu_b_at_kopt":   mu_b_at_kopt,
            "mu_b_at_thread": mu_b_at_thread,
            "time_at_thread": time_at_thread,
        }


# ---------------------------------------------------------------------------
#  Singleton inference wrapper used from the C++ scheduler
# ---------------------------------------------------------------------------
class PINNPredictor:
    _instance = None

    def __init__(self, model_path: str = None, scaler_path: str = None):
        if model_path is None:
            self.model = None
            self.feature_scaler = None
            return
        self.model = OptimalThreadsPredictor(INPUT_DIM)
        self.model.load_state_dict(
            torch.load(model_path, map_location=torch.device("cpu"))
        )
        self.model.eval()
        self.feature_scaler = joblib.load(scaler_path) if scaler_path else None

    @classmethod
    def get_instance(cls, model_path: str = None, scaler_path: str = None):
        if cls._instance is None:
            if model_path is None:
                raise ValueError("Model path must be provided for initial load")
            cls._instance = cls(model_path, scaler_path)
        return cls._instance

    def predict(self, N, mu_e, k, hardware_features) -> int:
        """Predict the recommended worker-thread count.

        Parameters
        ----------
        N : int
            Pending task count for the current simulation cycle.
        mu_e : float
            Per-task execution rate, i.e. 1 / event_duration.
        k : int
            Currently configured worker-thread count.
        hardware_features : list[float]
            ``[user_cpu, system_cpu, VSZ, cswch]`` -- order MUST match
            ``HARDWARE_FEATURES``.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")

        hw = np.asarray(hardware_features, dtype=np.float32)
        if self.feature_scaler is not None:
            hw = self.feature_scaler.transform(hw.reshape(1, -1)).flatten()

        feats = np.concatenate(
            (np.array([N, mu_e, k], dtype=np.float32), hw), axis=0
        ).reshape(1, -1)

        with torch.no_grad():
            out   = self.model(torch.tensor(feats, dtype=torch.float32))
            k_opt = out["k_opt"].item()
        return int(round(max(k_opt, 3.0)))


# ---------------------------------------------------------------------------
#  C-callable entry points (invoked by pybind11 from ScheduleManager.cpp)
# ---------------------------------------------------------------------------
def initialize_model(model_path: str, scaler_path: str = None) -> None:
    PINNPredictor.get_instance(model_path, scaler_path)


def pinn_predict(N, mu_e, k, hardware_params) -> int:
    return PINNPredictor.get_instance().predict(N, mu_e, k, hardware_params)
