# Relativistic Shocked Cocoons

Relativistic shocked cocoons arise when a relativistic jet launched in a compact binary merger or core-collapse event is choked or propagates through surrounding ejecta or a stellar envelope (Gottlieb et al. 2018; Piro & Kollmeier 2018).

As the shocked cocoon expands and becomes optically thin, radiative diffusion releases a fast, bright optical/UV transient lasting $\sim 0.5\text{--}3$ days with peak absolute magnitude $M_g \sim -15\text{--}-17$, acting as a prominent kilonova imposter.

## Population Generator

The `ShockedCocoonPopulation` samples physical cocoon properties matching the Redback / fiestaEM parameterizations:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rate` | float | `30.0` | Volumetric rate in $\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}$ |
| `z_max` | float | `0.30` | Maximum redshift |
| `peak_abs_mag` | float | `-16.0` | Peak absolute magnitude |
| `log10_mej` | float (sampled) | Uniform $[-3.0, -1.0]$ | Cocoon ejecta mass $\log_{10}(M_\mathrm{ej}/M_\odot)$ |
| `log10_vej` | float (sampled) | Uniform $[-1.0, -0.4]$ | Ejecta velocity $\log_{10}(v_\mathrm{ej}/c)$ |
| `eta` | float (sampled) | Normal $1.5 \pm 0.5$ | Density profile power-law slope |
| `log10_tshock` | float (sampled) | Normal $2.2 \pm 0.4$ | Shock breakout time $\log_{10}(t_\mathrm{shock}/\mathrm{s})$ |

### Python Usage

```python
from survey_sim import ShockedCocoonPopulation
from survey_sim.fiesta_cocoon_model import FiestaCocoonModel

pop = ShockedCocoonPopulation(rate=30.0, z_max=0.30, peak_abs_mag=-16.0)
model = FiestaCocoonModel(filters=["g", "r", "i"])
```
