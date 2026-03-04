"""
Learnable Heat Flux Parameterization q(x, t)

The unknown boundary heat flux is represented as a spatial Gaussian
with a temporal ramp function. Four parameters are jointly learned
with the PINN during inverse training.

Part of M.Tech Thesis — IIST
Author: Raghavendra M

Note: Partial showcase. Full training integration not included.
"""

import torch
import numpy as np


# Physical domain (used for non-dimensionalization)
L_REF   = 0.005   # m  (wall thickness, reference length)
K_CU    = 387.6   # W/m·K  (copper thermal conductivity)
T_SCALE = 1000.0  # K  (temperature scale)
T_INF   = 300.0   # K  (coolant / initial temperature)

# Sigma bounds: prevents flux width from collapsing or blowing up
# Physical range: 5 mm to 60 mm
LOG_SIGMA_MIN = np.log(1.0 / L_REF * 0.005)   # 5 mm in ND
LOG_SIGMA_MAX = np.log(1.0 / L_REF * 0.060)   # 60 mm in ND

# Soft x0 bounds (quadratic penalty in loss)
X0_ND_MIN = 2.0    # 10 mm in ND
X0_ND_MAX = 38.0   # 190 mm in ND


class GaussianHeatFlux:
    """
    Learnable Gaussian heat flux with temporal ramp.

    q(x, t) = [q_base + q_amp * exp(-(x - x0)^2 / sigma^2)] * ramp(t)

    ramp(t) = clamp(t_physical / 0.2s, max=1.0)
              linear rise over 0.2s, then steady-state

    Parameters stored in log-space to ensure strict positivity:
        log_q_base, log_q_amp  ->  q_base = exp(log_q_base), etc.
        x0_nd                  ->  Gaussian center (ND units)
        log_sigma              ->  log of Gaussian width (ND)
    """

    def __init__(self, log_q_base, log_q_amp, x0_nd, log_sigma, t_ref, device):
        """
        Args:
            log_q_base : nn.Parameter  (log of baseline flux, ND)
            log_q_amp  : nn.Parameter  (log of peak amplitude, ND)
            x0_nd      : nn.Parameter  (Gaussian center, ND)
            log_sigma  : nn.Parameter  (log of width, ND)
            t_ref      : float         (reference time for ND->physical conversion)
            device     : torch.device
        """
        self.log_q_base = log_q_base
        self.log_q_amp  = log_q_amp
        self.x0_nd      = x0_nd
        self.log_sigma  = log_sigma
        self.t_ref      = t_ref
        self.device     = device

    def __call__(self, x_nd, t_nd=None):
        """
        Evaluate q(x, t) in non-dimensional units.

        Args:
            x_nd : (N, 1) tensor  — ND x-coordinate
            t_nd : (N, 1) tensor or None
                   If None: returns spatial profile only (ramp=1, steady-state)

        Returns:
            q    : (N, 1) tensor  — ND heat flux
        """
        q_base = torch.exp(self.log_q_base)
        q_amp  = torch.exp(self.log_q_amp)

        # Hard clamp sigma to physical range
        log_sig_clamped = torch.clamp(self.log_sigma, LOG_SIGMA_MIN, LOG_SIGMA_MAX)
        sigma = torch.exp(log_sig_clamped)

        # Spatial Gaussian profile
        spatial = q_base + q_amp * torch.exp(
            -((x_nd - self.x0_nd) / sigma) ** 2
        )

        if t_nd is not None:
            t_phys = t_nd * self.t_ref
            ramp   = torch.clamp(t_phys / 0.2, max=1.0)
            return spatial * ramp
        return spatial

    def x0_penalty(self):
        """
        Soft quadratic penalty if x0 drifts outside [X0_ND_MIN, X0_ND_MAX].
        Returns 0 when x0 is inside bounds.
        Used in total loss with small weight (0.1–0.5).
        """
        lower = torch.clamp(X0_ND_MIN - self.x0_nd, min=0.0) ** 2
        upper = torch.clamp(self.x0_nd - X0_ND_MAX, min=0.0) ** 2
        return lower + upper

    def to_physical(self):
        """
        Convert current learned parameters to physical units for reporting.

        Returns dict with keys: q_base_MW, q_amp_MW, q_peak_MW, x0_mm, sigma_mm
        """
        with torch.no_grad():
            q_base_mw = torch.exp(self.log_q_base).item() * K_CU * T_SCALE / L_REF / 1e6
            q_amp_mw  = torch.exp(self.log_q_amp).item()  * K_CU * T_SCALE / L_REF / 1e6
            x0_mm     = self.x0_nd.item() * L_REF * 1e3
            sig_mm    = torch.exp(self.log_sigma).item()  * L_REF * 1e3
        return {
            'q_base_MW' : q_base_mw,
            'q_amp_MW'  : q_amp_mw,
            'q_peak_MW' : q_base_mw + q_amp_mw,
            'x0_mm'     : x0_mm,
            'sigma_mm'  : sig_mm,
        }
