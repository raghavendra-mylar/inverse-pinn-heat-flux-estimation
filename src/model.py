"""
Physics-Informed Neural Network (PINN) Architecture
for 2D Transient Inverse Heat Conduction.

Part of M.Tech Thesis — IIST
Author: Raghavendra M

Note: This is a partial showcase of the architecture.
Full training pipeline is part of ongoing thesis work.
"""

import torch
import torch.nn as nn
import numpy as np


class PINN(nn.Module):
    """
    Physics-Informed Neural Network for 2D Transient Inverse Heat Conduction.

    Maps (x, y, t) -> T(x, y, t)  [non-dimensional temperature]

    Coordinates are internally normalized to [-1, 1] before passing
    through the network (improves gradient flow for Tanh activations).

    Architecture: 8 hidden layers x 256 neurons, Tanh activation
    Total parameters: 461,825
    """

    def __init__(self, layers: list):
        """
        Args:
            layers: list of layer sizes, e.g. [3, 256, 256, ..., 1]
                    First element must be 3 (x, y, t input)
                    Last element must be 1 (temperature output)
        """
        super(PINN, self).__init__()
        self.activation = nn.Tanh()

        self.linears = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1])
             for i in range(len(layers) - 1)]
        )

        # Xavier initialization for hidden layers, small gain for output
        for layer in self.linears[:-1]:
            nn.init.xavier_normal_(layer.weight, gain=1.0)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_normal_(self.linears[-1].weight, gain=0.1)
        nn.init.zeros_(self.linears[-1].bias)

    def _normalize(self, x_nd, y_nd, t_nd,
                   x_min, x_max, y_min, y_max, t_min, t_max):
        """Map ND coordinates to [-1, 1] for each dimension."""
        xn = 2.0 * (x_nd - x_min) / (x_max - x_min) - 1.0
        yn = 2.0 * (y_nd - y_min) / (y_max - y_min) - 1.0
        tn = 2.0 * (t_nd - t_min) / (t_max - t_min) - 1.0
        return xn, yn, tn

    def forward(self, x_nd, y_nd, t_nd):
        """
        Forward pass through the network.

        Args:
            x_nd, y_nd, t_nd: (N, 1) tensors, non-dimensional coordinates

        Returns:
            T_nd: (N, 1) tensor, non-dimensional temperature in [0, 1]
        """
        u = torch.cat([x_nd, y_nd, t_nd], dim=-1)
        for linear in self.linears[:-1]:
            u = self.activation(linear(u))
        return self.linears[-1](u)

    def predict_T_nd(self, x_nd, y_nd, t_nd):
        """Predict non-dimensional temperature."""
        return self.forward(x_nd, y_nd, t_nd)

    def predict_T_physical(self, x_nd, y_nd, t_nd, T_scale, T_inf):
        """
        Predict physical temperature in Kelvin.

        T_physical = T_nd * T_scale + T_inf
        """
        return self.predict_T_nd(x_nd, y_nd, t_nd) * T_scale + T_inf


# Default configuration used in thesis
LAYER_CONFIG = [3, 256, 256, 256, 256, 256, 256, 256, 256, 1]


def build_model(device='cuda'):
    """Instantiate the PINN model with thesis configuration."""
    model = PINN(LAYER_CONFIG).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"PINN Architecture: {LAYER_CONFIG}")
    print(f"Total parameters : {n_params:,}")
    print(f"Device           : {device}")
    return model
