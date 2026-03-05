# Populations

survey-sim generates synthetic transient populations by drawing physical parameters from astrophysically motivated distributions. Each population implements the `PopulationGenerator` trait.

## PopulationGenerator Trait

```rust
pub trait PopulationGenerator: Send + Sync {
    fn generate(&self, n: usize, rng: &mut dyn RngCore) -> Vec<TransientInstance>;
    fn volumetric_rate(&self) -> f64;       // Gpc^-3 yr^-1
    fn transient_type(&self) -> TransientType;
}
```

All populations share the same sampling strategy for extrinsic parameters:

- **Redshift**: rejection sampling from \(dN/dz \propto dV/dz\) (comoving volume element)
- **Sky position**: uniform on the sphere (isotropic)
- **Explosion time**: uniform within the survey's MJD range
- **MW extinction**: Gaussian \(A_V^\mathrm{MW} \sim \mathcal{N}(0.1, 0.1)\), clipped to \([0, 2]\)

## Built-in Populations

### KilonovaPopulation

Metzger (2017) parametric kilonova model.

| Parameter | Distribution | Range |
|-----------|-------------|-------|
| `mej` | Log-uniform | \([10^{-3}, 0.1]\) M\(_\odot\) |
| `vej` | Gaussian | \(\mathcal{N}(0.2, 0.05)\), clipped \([0.05, 0.5]\) c |
| `kappa` | Log-uniform | \([0.5, 30]\) cm\(^2\)/g |
| `peak_abs_mag` | Gaussian | \(\mathcal{N}(-16, 1)\), clipped \(\pm 3\) |
| Host \(A_V\) | Gaussian | \(\mathcal{N}(0.1, 0.2)\), clipped \([0, 3]\) |

```python
pop = KilonovaPopulation(rate=1000.0, z_max=0.3, peak_abs_mag=-16.0)
```

### Bu2026KilonovaPopulation

Two-component kilonova parameters matching the Bu2026 surrogate training bounds (for use with fiestaEM).

| Parameter | Distribution | Range |
|-----------|-------------|-------|
| `log10_mej_dyn` | Uniform | \([-4.0, -1.3]\) |
| `v_ej_dyn` | Uniform | \([0.12, 0.35]\) c |
| `Ye_dyn` | Uniform | \([0.15, 0.35]\) |
| `log10_mej_wind` | Uniform | \([-4.0, -0.56]\) |
| `v_ej_wind` | Uniform | \([0.05, 0.15]\) c |
| `Ye_wind` | Uniform | \([0.2, 0.4]\) |
| `inclination_EM` | \(\arccos(U(0,1))\) | \([0, \pi/2]\) rad |

```python
pop = Bu2026KilonovaPopulation(rate=1000.0, z_max=0.3)
```

### FixedBu2026KilonovaPopulation

Same as Bu2026 but with user-specified fixed physical parameters. Only redshift, sky position, explosion time, and extinction are randomized.

```python
pop = FixedBu2026KilonovaPopulation(
    log10_mej_dyn=-2.0, v_ej_dyn=0.2, ye_dyn=0.25,
    log10_mej_wind=-1.5, v_ej_wind=0.1, ye_wind=0.3,
    inclination_em=0.5,
    rate=1000.0, z_max=0.3,
)
```

### SupernovaIaPopulation

Type Ia supernovae with SALT2-like stretch and color parameters.

| Parameter | Distribution | Range |
|-----------|-------------|-------|
| `x1` (stretch) | Gaussian | \(\mathcal{N}(0, 1)\), clipped \([-3, 3]\) |
| `c` (color) | Gaussian | \(\mathcal{N}(0, 0.1)\), clipped \([-0.3, 0.3]\) |
| `peak_abs_mag` | Gaussian | \(\mathcal{N}(-19.3, 0.15)\), clipped \(\pm 1\) |
| Host \(A_V\) | Gaussian | \(\mathcal{N}(0.2, 0.3)\), clipped \([0, 3]\) |

```python
pop = SupernovaIaPopulation(rate=30000.0, z_max=1.0, peak_abs_mag=-19.3)
```

### SupernovaIIPopulation

Type II supernovae with plateau characteristics.

| Parameter | Distribution | Range |
|-----------|-------------|-------|
| `plateau_duration` | Gaussian | \(\mathcal{N}(80, 20)\), clipped \([30, 150]\) days |
| `plateau_slope` | Gaussian | \(\mathcal{N}(0.01, 0.005)\), clipped \([0, 0.05]\) mag/day |
| `peak_abs_mag` | Gaussian | \(\mathcal{N}(-17, 0.8)\), clipped \(\pm 3\) |

```python
pop = SupernovaIIPopulation(rate=70000.0, z_max=0.5, peak_abs_mag=-17.0)
```

### TdePopulation

Tidal disruption events.

| Parameter | Distribution | Range |
|-----------|-------------|-------|
| `m_bh` | Log-uniform | \([10^5, 10^8]\) M\(_\odot\) |
| `m_star` | Gaussian | \(\mathcal{N}(1, 0.5)\), clipped \([0.1, 10]\) M\(_\odot\) |
| `peak_abs_mag` | Gaussian | \(\mathcal{N}(-20, 1)\), clipped \(\pm 3\) |

```python
pop = TdePopulation(rate=100.0, z_max=0.5, peak_abs_mag=-20.0)
```

### GrbPopulation

GRB afterglows sampled from a pre-computed CSV catalog. Each row contains a full set of blastwave parameters (jet energy, Lorentz factor, microphysics, viewing angle, etc.) drawn from astrophysical distributions.

The population samples catalog rows with replacement and assigns random sky positions, explosion times, and MW extinction.

| CSV Column | model_params Key | Description |
|------------|-----------------|-------------|
| `Eiso` | `Eiso` | Isotropic equivalent energy (erg) |
| `Gamma_0` | `Gamma_0` | Initial Lorentz factor |
| `thv` | `theta_v` | Viewing angle (rad) |
| `logthc` | `logthc` | log10 jet half-opening angle |
| `logn0` | `logn0` | log10 ISM density (cm\(^{-3}\)) |
| `logepse` | `logepse` | log10 electron energy fraction |
| `logepsB` | `logepsB` | log10 magnetic energy fraction |
| `p` | `p` | Electron spectral index |
| `av` | `av` | Host extinction \(A_V\) |
| `p_rvs` | `p_rvs` | Reverse shock electron index |
| `logepse_rvs` | `logepse_rvs` | log10 RS electron fraction |
| `logepsB_rvs` | `logepsB_rvs` | log10 RS magnetic fraction |
| `peak_mag` | --- | Pre-computed peak apparent mag (informational) |

```python
pop = GrbPopulation(
    "GRB_afterglows_argus.csv",
    rate=1.0,      # Gpc^-3 yr^-1
    z_max=6.0,
)
```

## Custom Populations

Implement the `PopulationGenerator` trait in Rust:

```rust
pub struct MyPopulation { /* ... */ }

impl PopulationGenerator for MyPopulation {
    fn generate(&self, n: usize, rng: &mut dyn RngCore) -> Vec<TransientInstance> {
        // Draw parameters, build TransientInstance structs
    }

    fn volumetric_rate(&self) -> f64 { 100.0 }
    fn transient_type(&self) -> TransientType { TransientType::Custom }
}
```

Or use a Python callback model with any of the built-in populations.
