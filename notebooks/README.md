# Notebooks

## Available Notebooks

### `inverse_pinn_heat_flux.ipynb` (Full Version)

The complete training notebook is **not publicly shared** as this is part of an ongoing M.Tech thesis at IIST.

The notebook contains:

| Cell | Description |
|------|-------------|
| Cell 1 | ANSYS sensor data loading & preprocessing |
| Cell 2 | Non-dimensionalization & coordinate flip |
| Cell 3 | PINN architecture + Gaussian q-param initialization |
| Cell 4 | Collocation sampler + loss functions (7 components) |
| Cell 5 | Adam training (25k epochs, 3-stage curriculum) |
| Cell 6 | L-BFGS fine-tuning (500 outer iterations) |
| Cell 7 | Validation: q(x,t) vs ANSYS + sensor scatter plots |

### Initialization Modes Tested

| File | Init Mode | Description |
|------|-----------|-------------|
| `perfect-with-gaussian.ipynb` | **blind** | Deliberately wrong init — robustness test |
| `truth-init.ipynb` | truth | ANSYS exact values — debugging baseline |

---

## Requesting Access

If you are a researcher or faculty with academic interest in this work,
feel free to reach out via GitHub issues or LinkedIn.

Please mention:
- Your institution and research context
- Specific aspect of the implementation you're interested in

> *Code will be fully open-sourced after thesis submission and publication.*
