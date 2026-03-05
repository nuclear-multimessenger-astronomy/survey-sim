# Pipeline Overview

survey-sim runs a Monte Carlo simulation in three phases to estimate transient detection efficiencies. Each phase is designed for parallelism where possible.

## Three-Phase Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Phase 1    │     │     Phase 2      │     │     Phase 3      │
│   Spatial    │────>│   Lightcurve     │────>│   Detection      │
│   Matching   │     │   Evaluation     │     │   Criteria       │
│  (parallel)  │     │ (parallel/seq)   │     │   (parallel)     │
└──────────────┘     └──────────────────┘     └──────────────────┘
```

### Phase 1: Spatial-Temporal Matching

For each transient instance, find all survey observations that overlap in both sky position and time window:

- **Sky position**: HEALPix cone query centered on the transient's (RA, Dec) with the instrument's field-of-view radius
- **Time window**: \([t_\mathrm{exp},\; t_\mathrm{exp} + \Delta t \cdot (1 + z)]\) where \(\Delta t\) is a configurable window (default 100 days)

This phase runs fully parallel via Rayon. Transients with zero matching observations are immediately discarded.

### Phase 2: Lightcurve Evaluation

For each transient with matching observations, evaluate the lightcurve model at the observed times and bands to produce apparent magnitudes:

- **Rust-native models** (`MetzgerKNModel`, `BlastwaveModel`): run in parallel via Rayon
- **Python callback models** (fiestaEM, custom `.predict()`): run sequentially due to the Python GIL

The model returns a `LightcurveEvaluation` containing apparent magnitudes per band at each observation time.

### Phase 3: Detection Evaluation

Apply detection criteria to each transient's lightcurve+observation pairs:

- Compare apparent magnitude to observation limiting magnitude (SNR check)
- Count detections across bands
- Evaluate rise/fade rate constraints (fast-transient criteria)
- Check minimum band and timespan requirements

This phase runs fully parallel via Rayon.

## Population Generation

Before the pipeline runs, each `PopulationGenerator` draws \(N\) transient instances with:

- **Redshift** \(z\) sampled from the volumetric rate distribution \(dN/dz \propto dV/dz\) via rejection sampling
- **Luminosity distance** \(d_L(z)\) from flat \(\Lambda\)CDM cosmology
- **Sky position** (RA, Dec) uniform on the sphere
- **Explosion time** \(t_\mathrm{exp}\) uniform within the survey's MJD range
- **Physical parameters** drawn from population-specific distributions
- **Extinction** \(A_V^\mathrm{MW}\) and \(A_V^\mathrm{host}\) from Gaussian distributions

## Rate Recovery

After the pipeline completes, detection efficiency is computed as:

\[
\epsilon(z) = \frac{N_\mathrm{det}(z)}{N_\mathrm{sim}(z)}
\]

The all-sky detection rate is then:

\[
\dot{N}_\mathrm{det} = \mathcal{R} \cdot \frac{\Omega_\mathrm{survey}}{4\pi} \cdot \int_0^{z_\mathrm{max}} \epsilon(z) \cdot \frac{dV}{dz} \cdot \frac{1}{1+z} \, dz
\]

where \(\mathcal{R}\) is the volumetric rate (Gpc\(^{-3}\) yr\(^{-1}\)), \(\Omega_\mathrm{survey}\) is the survey solid angle, and the \(1/(1+z)\) factor accounts for cosmological time dilation.
