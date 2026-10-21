# Subluminous Type Iax Supernovae

Type Iax supernovae (e.g., SN 2002cx, SN 2008ha) are subluminous, low-velocity thermonuclear transients believed to originate from the deflagration of carbon-oxygen white dwarfs that fail to completely unbind, leaving behind a bound stellar remnant (Foley et al. 2013; Jha 2017).

They represent $\sim 15\%\text{--}30\%$ of the local Type Ia rate ($R_0 \sim 3{,}000\text{--}6{,}000\,\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}$) and have low synthesized nickel masses ($M_\mathrm{Ni} \sim 0.005\text{--}0.08\,M_\odot$) and low ejecta velocities ($v_\mathrm{ej} \sim 2{,}000\text{--}6{,}000\,\mathrm{km/s}$). Their low luminosity ($M_V \sim -13\text{--}-17.5\,\mathrm{mag}$) and fast evolutionary timescales make them important contaminant candidates in shallow wide-field kilonova searches.

## Population Generator

The `SupernovaIaxPopulation` generates realistic Type Iax distributions:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rate` | float | `3000.0` | Volumetric rate in $\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}$ |
| `z_max` | float | `0.20` | Maximum redshift |
| `peak_abs_mag` | float | `-15.0` | Peak absolute magnitude (mean) |
| `m_ni` | float (sampled) | Uniform $[0.005, 0.08]\,M_\odot$ | Synthesized $^{56}\mathrm{Ni}$ mass |
| `v_ej` | float (sampled) | Uniform $[2000, 6000]\,\mathrm{km/s}$ | Ejecta velocity |
| `m_ej` | float (sampled) | Uniform $[0.10, 0.50]\,M_\odot$ | Total ejecta mass |
| `t_rise` | float (sampled) | Uniform $[6, 14]$ days | Rise time to peak |
| `t_fall` | float (sampled) | Uniform $[12, 25]$ days | Exponential decay time |

### Python Usage

```python
from survey_sim import SupernovaIaxPopulation, ParametricModel

pop = SupernovaIaxPopulation(rate=3000.0, z_max=0.20, peak_abs_mag=-15.0)
```
