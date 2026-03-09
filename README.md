# Inverse PINN for Heat Flux Estimation in Regenerative Cooling Channels

<p align="center">
  <img src="https://img.shields.io/badge/Framework-PyTorch-orange?style=for-the-badge&logo=pytorch" />
  <img src="https://img.shields.io/badge/Type-Inverse%20PINN-blueviolet?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Application-Rocket%20Propulsion-red?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Status-Under%20Review-yellow?style=for-the-badge" />
</p>

> **M.Tech Thesis Work** — Indian Institute of Space Science and Technology (IIST)  
> Specialization: Aerospace Engineering — Thermal & Propulsion

---

## 📄 Publication Status

Manuscript under preparation for submission to:
- *Journal of Computational Physics* (target)
- *Applied Thermal Engineering* (alternate)

> 🔬 **Research Disclosure Notice**  
> Full training implementation is withheld pending journal submission.  
> A complete reproducible codebase will be released upon acceptance.  
> Architecture and parameterization files are provided for reference.  
> For academic inquiries: **ragharit586@gmail.com**

Full code release planned post-acceptance.  
⭐ Star this repo to get notified when the paper and code drop.

---

## Problem Statement

In regenerative cooling channels of rocket engines, the **heat flux applied at the hot gas wall is unknown** — it cannot be measured directly during operation. Only sparse temperature sensor readings are available at a few interior locations.

This work uses a **Physics-Informed Neural Network in inverse mode** to recover the full spatiotemporal heat flux distribution `q(x, t)` from only **15 thermocouple sensors**, without any direct flux measurement.

```
 Hot Gas Side  →  q(x,t) = ?  [UNKNOWN - what we recover]
 ┌─────────────────────────────────────────────┐
 │        Copper Wall (2D Transient)           │
 │   ● ● ● ● ●   ← Sensor Row 1 (y=1.25mm)   │
 │   ● ● ● ● ●   ← Sensor Row 2 (y=2.50mm)   │
 │   ● ● ● ● ●   ← Sensor Row 3 (y=3.75mm)   │
 └─────────────────────────────────────────────┘
 Coolant Side  →  Robin BC: ∂T/∂y + Bi(x)·T = 0
```

---

## Results

### Gaussian Heat Flux Recovery — Blind Initialization

The model was initialized **deliberately far from truth** to test genuine robustness:

| Parameter | Init (Blind) | PINN Recovered | ANSYS Truth | Error |
|-----------|-------------|----------------|-------------|-------|
| q_base | 10.0 MW/m² | **14.73 MW/m²** | 15.0 MW/m² | 1.82% |
| q_amp | 10.0 MW/m² | **29.95 MW/m²** | 30.0 MW/m² | 0.17% |
| **q_peak** | **20.0 MW/m²** | **44.67 MW/m²** | **45.0 MW/m²** | **0.72%** |
| x₀ (center) | 80.0 mm | **100.43 mm** | 100.0 mm | 0.43 mm |
| σ (width) | 50.0 mm | **30.76 mm** | 30.0 mm | 2.52% |

**Sensor RMSE: 0.978 K** (from 15 sensors, 201 time steps = 3,015 data points)

> Starting 55% below true peak → Recovered to within **0.72% of true peak**

### Three Initialization Modes Tested

| Mode | Description | q_peak Error |
|------|-------------|-------------|
| `truth` | ANSYS exact values as start | Baseline |
| `blind` | Deliberately wrong (55% off peak, 20mm off center) | **0.72%** ✅ |
| `random` | Fixed random seed, unknown basin | In progress |

> 📊 Loss convergence curves, heat flux recovery plots, and temperature field comparisons are available in [`results/RESULTS.md`](results/RESULTS.md).

---

## Method Overview

### 1. Governing PDE (Non-Dimensional)

```
∂T/∂t = ∂²T/∂x² + ∂²T/∂y²     (in Ω = [0,L] × [0,δ] × [0,T_end])
```

### 2. Boundary Conditions

| Boundary | Condition | Physical Meaning |
|----------|-----------|------------------|
| y = 0 (hot wall) | `∂T/∂y = -q(x,t)` | **Unknown flux — LEARNED** |
| y = δ (coolant) | `∂T/∂y + Bi(x)·T = 0` | Convective Robin BC |
| x = 0, L | `∂T/∂x = 0` | Adiabatic sides |
| t = 0 | `T = T_∞ = 300K` | Uniform initial condition |

### 3. Learnable Heat Flux Parameterization

The unknown flux is parameterized as a **Gaussian with temporal ramp**:

```python
q(x, t) = [q_base + q_amp · exp(-(x - x₀)² / σ²)] · ramp(t)

ramp(t) = clamp(t / 0.2s, max=1.0)   # linear ramp over 0.2s, then steady
```

Four learnable parameters: `{log_q_base, log_q_amp, x₀, log_σ}`  
Log-parameterization ensures strict positivity.

---

## Architecture

> ✅ **Architecture is fully disclosed** — the network design is shared for reproducibility and reference.

```
Input: [x, y, t]  (non-dimensional, mapped to [-1, 1])
         ↓
   8 × 256 Tanh layers
         ↓
   Output: T_ND(x, y, t)

Total PINN parameters: 461,825
Learnable q parameters: 4  (jointly optimized)
```

See [`src/model.py`](src/model.py) for the full architecture implementation.

### Training Strategy

```
Stage 1 (ep 1–8000)   : Data + IC only → T-field settles, q frozen
Stage 2 (ep 8001–13000): Ramp physics + hot BC → q starts moving
Stage 3 (ep 13001–25000): Full physics, w_hot=50 → q converges
         ↓
L-BFGS fine-tuning (500 outer iters) → final polish
```

### Loss Function

```
L_total = w_data·L_data + w_pde·L_pde + w_hot·L_hot
        + w_bot·L_bot + w_ad·L_ad + w_ic·L_ic + w_x0·L_x0_pen
```

- `L_data` : Weighted MSE on sensors (Row-1 sensors get 3× weight — closest to hot wall)
- `L_pde`  : 2D transient heat equation residual
- `L_hot`  : Neumann BC at hot gas wall → drives q recovery
- `L_bot`  : Robin BC at coolant side
- `L_ad`   : Adiabatic BCs at x=0, L
- `L_ic`   : Uniform initial temperature
- `L_x0_pen`: Soft quadratic penalty to keep x₀ in domain

---

## Domain & Material Properties

| Property | Value |
|----------|-------|
| Material | Copper (Cu) |
| Thermal conductivity k | 387.6 W/m·K |
| Density ρ | 8978 kg/m³ |
| Specific heat Cₚ | 381 J/kg·K |
| Wall length L | 200 mm |
| Wall thickness δ | 5 mm |
| Coolant T_∞ | 300 K |
| h(x) coolant | 30,000–40,000 W/m²·K (variable) |
| Simulation time | ~1 second (transient) |

---

## Reference Solution

Ground truth generated in **ANSYS Fluent** with:
- Applied Gaussian flux: q_base=15 MW/m², q_amp=30 MW/m², x₀=100mm, σ=30mm
- Temporal ramp: linear 0→1 over 0.2s, steady after
- 3,015 sensor readings exported (15 sensors × 201 time steps)

---

## Repository Structure

```
inverse-pinn-heat-flux-estimation/
│
├── src/
│   ├── model.py          # PINN architecture — fully disclosed
│   ├── heat_flux.py      # Gaussian q(x,t) parameterization
│   └── boundary.py       # Biot number function and BC helpers
│
├── results/
│   └── RESULTS.md        # Quantitative recovery results
│
├── notebooks/
│   └── README.md         # Full notebook release planned post-publication
│
├── requirements.txt
└── README.md
```

---

## Relevance to Rocket Engine Design

Accurate heat flux estimation is critical for:
- **Regenerative cooling channel design** — determining required coolant flow
- **Thermal margin analysis** — avoiding copper melt (T_melt = 1358 K)
- **Engine health monitoring** — detecting anomalous heat loads
- **Reducing experimental cost** — replace expensive flux gauges with sensors

This framework can be extended to real flight data where only thermocouple readings are available.

---

## Tech Stack

- **PyTorch** — PINN + automatic differentiation
- **ANSYS Fluent** — Ground truth CFD simulation
- **CUDA (A100/H100)** — GPU training
- **L-BFGS + Adam** — Two-phase optimizer
- **NumPy / Matplotlib** — Post-processing

---

## Contact

**Raghavendra M**  
M.Tech Aerospace Engineering (Thermal & Propulsion)  
Indian Institute of Space Science and Technology (IIST)  
Thiruvananthapuram, Kerala, India  

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/raghavendra-mylar-b00b95240/)  
[![GitHub](https://img.shields.io/badge/GitHub-raghavendra--mylar-black?style=flat&logo=github)](https://github.com/raghavendra-mylar)

---

*This work is part of an M.Tech thesis at IIST. Please cite appropriately if referencing.*
