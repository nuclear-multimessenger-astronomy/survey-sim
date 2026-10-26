# Dense Circumstellar Interaction Transients

Circumstellar interaction transients (e.g. SNe Ibn, fast Type IIn, interacting stripped-envelope supernovae) occur when high-velocity supernova ejecta collide with dense, nearby circumstellar material (CSM) lost by the progenitor star shortly before explosion (Chevalier & Irwin 2011; Chatzopoulos et al. 2012).

The conversion of kinetic energy into radiation via forward and reverse shocks produces luminous, fast-rising optical emission ($M_g \sim -16\text{--}-19\,\mathrm{mag}$) that rapidly peaks and declines, acting as a prominent kilonova imposter candidate in wide-field surveys.

## Population Generator

The `CsmInteractionPopulation` samples physical parameters matching the Chevalier / Chatzopoulos / fiestaEM parameterization:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rate` | float | `250.0` | Volumetric rate in $\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}$ |
| `z_max` | float | `0.25` | Maximum redshift |
| `peak_abs_mag` | float | `-17.5` | Peak absolute magnitude |
| `log10_mej` | float (sampled) | Uniform $[-0.3, 0.7]$ | Ejecta mass $\log_{10}(M_\mathrm{ej}/M_\odot)$ |
| `log10_csm_mass` | float (sampled) | Uniform $[-1.5, 0.2]$ | CSM mass $\log_{10}(M_\mathrm{csm}/M_\odot)$ |
| `log10_vej` | float (sampled) | Uniform $[3.7, 4.2]$ | Ejecta velocity $\log_{10}(v_\mathrm{ej}/\mathrm{km/s})$ |
| `eta` | float (sampled) | Normal $1.5 \pm 0.4$ | CSM density slope $\rho \propto r^{-\eta}$ |
| `log10_rho` | float (sampled) | Normal $-13.0 \pm 0.5$ | CSM reference density $\log_{10}(\rho_0/\mathrm{g\,cm}^{-3})$ |
| `log10_r0` | float (sampled) | Normal $1.0 \pm 0.3$ | CSM inner radius $\log_{10}(r_0/\mathrm{AU})$ |
| `log10_kappa` | float (sampled) | Normal $-0.47 \pm 0.1$ | Opacity $\log_{10}(\kappa/\mathrm{cm}^2\,\mathrm{g}^{-1})$ |

### Python Usage

```python
from survey_sim import CsmInteractionPopulation
from survey_sim.fiesta_csm_model import FiestaCsmModel

pop = CsmInteractionPopulation(rate=250.0, z_max=0.25, peak_abs_mag=-17.5)
model = FiestaCsmModel(filters=["g", "r", "i"])
```
