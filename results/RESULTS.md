# Inverse PINN — Quantitative Results

## Test Configuration

- **PINN**: 8 × 256 Tanh layers (461,825 parameters)
- **q-params**: 4 learnable (log_q_base, log_q_amp, x₀, log_σ)
- **Training**: Adam 25k epochs → L-BFGS 500 outer iterations
- **GPU**: NVIDIA A100 / H100
- **Ground Truth**: ANSYS Fluent 2D transient simulation
- **Sensor data**: 15 thermocouples × 201 time steps = **3,015 points**

---

## Blind Initialization (Primary Result)

Initialization deliberately set far from truth to test robustness:

| Parameter | Init (Blind) | After Adam | After L-BFGS | ANSYS Truth | Final Error |
|-----------|-------------|-----------|-------------|-------------|-------------|
| q_base    | 10.0 MW/m²  | ~12 MW/m² | **14.73 MW/m²** | 15.0 MW/m² | **1.82%** |
| q_amp     | 10.0 MW/m²  | ~25 MW/m² | **29.95 MW/m²** | 30.0 MW/m² | **0.17%** |
| **q_peak**| **20.0 MW/m²** | ~37 MW/m² | **44.67 MW/m²** | **45.0 MW/m²** | **0.72%** |
| x₀        | 80.0 mm     | ~99 mm    | **100.43 mm**   | 100.0 mm   | **0.43 mm** |
| σ         | 50.0 mm     | ~33 mm    | **30.76 mm**    | 30.0 mm    | **2.52%** |

**Sensor RMSE (L-BFGS final): 0.978 K**  
**Starting error: ~55% below true peak → Final error: 0.72%**

---

## Sensor Fit Quality

| Metric | Value |
|--------|-------|
| Sensor RMSE (Adam best) | 1.394 K |
| Sensor RMSE (L-BFGS best) | **0.978 K** |
| L-BFGS improvement over Adam | 0.987 K |
| Max sensor T in data | 1276.91 K (ANSYS) |
| T_wall max (PINN, y=0) | < 1358 K (Cu melt limit ✓) |

---

## q(x, t=steady) Validation Against ANSYS

| Metric | Value |
|--------|-------|
| q RMSE (steady-state) | < 0.5 MW/m² |
| q Mean relative error | < 3% |
| ANSYS q range | 15.0 – 45.0 MW/m² |
| PINN q range (steady) | 14.73 – 44.67 MW/m² |

---

## Training Time

| Stage | Duration | Epochs |
|-------|----------|--------|
| Stage 1 (Data only) | ~1 min | 8,000 |
| Stage 2 (Ramp physics) | ~9 min | 5,000 |
| Stage 3 (Full physics) | ~14 min | 12,000 |
| L-BFGS fine-tuning | ~25 min | 500 outer |
| **Total** | **~49 min** | — |

*Timings on NVIDIA A100 GPU*

---

## Initialization Robustness Summary

| Init Mode | q_peak Init | q_peak Recovered | Error |
|-----------|------------|-----------------|-------|
| `truth`   | 45.0 MW/m² (exact) | ~45.0 MW/m² | ~0% (baseline) |
| `blind`   | 20.0 MW/m² (55% off) | **44.67 MW/m²** | **0.72%** ✅ |
| `random`  | varies | thesis in progress | — |
