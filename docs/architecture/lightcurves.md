# Lightcurve Models

survey-sim supports three categories of lightcurve models, all implementing the `LightcurveModel` trait.

## LightcurveModel Trait

```rust
pub trait LightcurveModel: Send + Sync {
    fn evaluate(
        &self,
        instance: &TransientInstance,
        times_mjd: &[f64],
        bands: &[Band],
    ) -> Result<LightcurveEvaluation>;

    fn requires_gil(&self) -> bool { false }
}
```

Models receive a `TransientInstance` (redshift, distance, sky position, explosion time, physical parameters, extinction) and a list of observation times/bands. They return apparent magnitudes per band.

## Parametric Models (Rust-native)

The `ParametricModel` wraps the `lightcurve-fitting` crate's analytic templates:

1. Convert observer-frame MJD to rest-frame days since explosion
2. Evaluate normalized flux via `eval_model_flux()` (peak ~ 1)
3. Convert to absolute magnitude using `peak_abs_mag`
4. Apply distance modulus, K-correction, and extinction
5. Add per-band color offsets

Available templates: **MetzgerKN**, **Bazin**, **Villar**, **TDE**, **Afterglow**.

These models are fully parallel via Rayon (`requires_gil = false`).

## Blastwave Model (Rust-native)

The `BlastwaveModel` runs a full relativistic blast wave simulation for GRB afterglows using the [blastwave](https://github.com/nuclear-multimessenger-astronomy/blastwave) crate.

### Physics Pipeline

For each transient instance:

1. **Build jet configuration** --- top-hat angular profile with arcsinh-spaced grid (128 cells), ISM density, forward + reverse shock microphysics
2. **Solve hydrodynamics** --- finite-volume PDE with CFL-limited RK2, lateral spreading
3. **Solve reverse shock** --- coupled forward-reverse shock dynamics
4. **Configure afterglow radiation** --- synchrotron + self-absorption (`sync_ssa_smooth`)
5. **Compute luminosity** --- EATS integration at each observation time and frequency
6. **Convert to magnitudes** --- \(L_\nu \to F_\nu \to m_\mathrm{AB}\) with extinction

### Flux Conversion

Specific luminosity \(L_\nu\) (erg/s/Hz) from the EATS integral is converted to observed flux density:

\[
F_\nu = \frac{L_\nu \cdot (1 + z)}{4\pi d_L^2}
\]

Then to AB magnitude:

\[
m_\mathrm{AB} = -2.5 \log_{10}\left(\frac{F_\nu}{3631 \,\mathrm{Jy}}\right) = 23.9 - 2.5 \log_{10}(F_\mathrm{mJy})
\]

### Parameters

The blastwave model reads the following from `TransientInstance.model_params`:

| Key | Description | Units |
|-----|-------------|-------|
| `Eiso` | Isotropic equivalent energy | erg |
| `Gamma_0` | Initial bulk Lorentz factor | -- |
| `theta_v` | Viewing angle | rad |
| `logthc` | log10 of jet half-opening angle | rad |
| `logn0` | log10 of ISM number density | cm\(^{-3}\) |
| `logepse` | log10 of electron energy fraction | -- |
| `logepsB` | log10 of magnetic energy fraction | -- |
| `p` | Electron spectral index | -- |
| `av` | Host extinction \(A_V\) | mag |
| `p_rvs` | Reverse shock electron index | -- |
| `logepse_rvs` | log10 of RS electron energy fraction | -- |
| `logepsB_rvs` | log10 of RS magnetic energy fraction | -- |

### Band Frequencies

The model maps observation band names to frequencies (Hz). Default: `{"g": 6.3e14}` (Argus g-band).

```python
model = BlastwaveModel(
    radiation_model="sync_ssa_smooth",
    band_frequencies={"g": 6.3e14, "r": 4.8e14},
)
```

Available radiation models: `sync`, `sync_smooth`, `sync_ssa`, `sync_ssa_smooth`, `sync_dnp`, `sync_ssc`.

## Python Callback Models

Any Python object with a `.predict(params)` method can be used as a lightcurve model. The method receives a dictionary of parameters and must return `(times, {band: mags})`.

```python
class MyModel:
    def predict(self, params):
        z = params["redshift"]
        d_l = params["luminosity_distance"]
        times = params["_obs_times_mjd"]
        bands = params["_obs_bands"]
        # ... compute magnitudes ...
        return (times, {"g": mags_g, "r": mags_r})
```

!!! warning "Performance"
    Python callback models run sequentially because they require the GIL. For large simulations (>1000 transients), prefer Rust-native models.

## Extinction

All models apply extinction from two sources:

- **Host galaxy**: \(A_\lambda^\mathrm{host} = A_V^\mathrm{host} \times R_\lambda\) using Cardelli et al. (1989) ratios
- **Milky Way**: \(A_\lambda^\mathrm{MW} = A_V^\mathrm{MW} \times R_\lambda\)

The extinction ratios \(R_\lambda = A_\lambda / A_V\) for standard bands:

| Band | \(R_\lambda\) |
|------|--------------|
| u | 1.56 |
| g | 1.31 |
| r | 1.00 |
| i | 0.75 |
| z | 0.55 |
| y | 0.47 |
| J | 0.29 |
| H | 0.18 |
| K | 0.11 |

Custom extinction coefficients can be defined in instrument YAML configurations.
