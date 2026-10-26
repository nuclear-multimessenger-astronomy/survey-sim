# Type IIb Supernovae (Shock Cooling)

Type IIb supernovae (SNe IIb) are core-collapse events of massive stars whose hydrogen envelopes have been largely stripped by binary interactions or stellar winds, leaving only a thin residual hydrogen shell ($M_\mathrm{env} \sim 0.01\text{--}0.3\,M_\odot$, $R_\mathrm{env} \sim 30\text{--}500\,R_\odot$).

When the supernova shock wave breaks out through this extended, low-mass envelope, radiative diffusion powers a bright, fast-cooling optical/UV flash lasting 1–5 days before the slower radioactive nickel-56 decay peak emerges weeks later (Arcavi et al. 2017; Barna et al. 2025).

Because the volumetric rate of SNe IIb ($\approx 1200\,\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}$, $\approx 12\%$ of all core-collapse SNe) exceeds the kilonova rate ($\approx 10\text{--}300\,\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}$) by more than an order of magnitude, early shock-cooling transients are primary contaminants (imposters) in rapid kilonova follow-up searches.

## Population Generator

The `SupernovaIIbPopulation` samples physical shock-cooling envelope properties matching the Piro (2021) and FiestaEM `ShockCoolingModel` parameterizations:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rate` | float | `1200.0` | Volumetric rate in $\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}$ |
| `z_max` | float | `0.15` | Maximum redshift |
| `peak_abs_mag` | float | `-16.5` | Peak absolute magnitude of early shock cooling |
| `log10_Menv` | float (sampled) | Uniform $[-2.0, -0.5]$ | Envelope mass $\log_{10}(M_\mathrm{env}/M_\odot)$ |
| `log10_Renv` | float (sampled) | Uniform $[1.48, 2.70]$ | Progenitor radius $\log_{10}(R_\mathrm{env}/R_\odot)$ |
| `log10_Ee` | float (sampled) | Normal $51.0 \pm 0.2$ | Shock energy $\log_{10}(E_e/\mathrm{erg})$ |

### Python Usage

```python
from survey_sim import SupernovaIIbPopulation, SimulationPipeline, DetectionCriteria

pop = SupernovaIIbPopulation(
    rate=1200.0,
    z_max=0.15,
    peak_abs_mag=-16.5,
)
```

## Light-Curve Modeling

Survey-sim supports two pathways for Type IIb shock-cooling light curves:

1. **Rust Native (High-Throughput)**: Parametric model using `SviModelName::Villar` or fast analytical blackbody shock cooling for multithreaded Rayon execution.
2. **FiestaEM Analytical Model**: `FiestaShockCoolingModel` wraps the JAX-differentiable `fiesta.models.analytical_models.ShockCoolingModel` (Piro 2021 piecewise solution) via `PythonCallbackModel`.

### Using FiestaShockCoolingModel

```python
from survey_sim import SimulationPipeline, DetectionCriteria, SupernovaIIbPopulation
from survey_sim.fiesta_shock_cooling_model import FiestaShockCoolingModel

model = FiestaShockCoolingModel(filters=["lsstg", "lsstr", "lssti"])

pipeline = SimulationPipeline(
    survey=survey,
    populations=[pop],
    models={"SNIIb": model},
    detection=DetectionCriteria(min_detections=2, snr_threshold=5.0),
)
result = pipeline.run()
```
