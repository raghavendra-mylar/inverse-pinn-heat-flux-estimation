"""
Boundary Condition Helpers

Biot number function for the coolant-side Robin BC and
helper utilities for the adiabatic side BCs.

Physics:
  Coolant BC (y = delta):  ∂T/∂y + Bi(x) · T = 0
  where Bi(x) = h(x) · L_ref / k_cu

  h(x) varies linearly from 30,000 to 40,000 W/m²·K along the channel,
  representing higher convective cooling near the nozzle throat region.

Part of M.Tech Thesis — IIST
Author: Raghavendra M
"""

import torch
import numpy as np


# Material & reference values
L_REF = 0.005   # m  (reference length = wall thickness)
K_CU  = 387.6   # W/m·K
X_MAX = 200e-3  # m  (physical channel length)

# Coolant convective heat transfer coefficient bounds
H_MIN = 30_000.0   # W/m²·K  at x = 0
H_MAX = 40_000.0   # W/m²·K  at x = L


def biot_number_nd(x_nd):
    """
    Spatially varying Biot number in non-dimensional form.

    Bi(x) = h(x) · L_ref / k_cu

    h(x) varies linearly: H_MIN at x=0 to H_MAX at x=L
    This represents increasing convective cooling along channel length.

    Args:
        x_nd : (N, 1) tensor or scalar — ND x-coordinate (range: 0 to x_max_nd)

    Returns:
        Bi   : (N, 1) tensor — dimensionless Biot number
    """
    # Convert ND x to physical meters
    x_phys = x_nd * L_REF  # [m]

    # Linear interpolation of h over channel length
    h_dim = H_MIN + (H_MAX - H_MIN) * (x_phys / X_MAX)

    # Biot number: Bi = h * L_ref / k
    bi = h_dim * L_REF / K_CU
    return bi


def biot_range_check():
    """Sanity check: print Bi range across the domain."""
    x_test = np.linspace(0, 40, 10)   # ND domain [0, x_max_nd]
    bi_test = biot_number_nd(
        torch.tensor(x_test, dtype=torch.float32).unsqueeze(-1)
    ).numpy().flatten()
    print(f"Bi(x) range: [{bi_test.min():.4f}, {bi_test.max():.4f}]  (O(1) ✓)")
    return bi_test


if __name__ == "__main__":
    biot_range_check()
