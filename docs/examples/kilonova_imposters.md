# Kilonova Imposter & Detection Efficiency Analysis

This page outlines the **Kilonova Imposter & Detection Efficiency Analysis Framework**, which quantifies the contamination of optical kilonova searches by rapid astrophysical transients (such as Type IIb supernova shock cooling and Fast Blue Optical Transients), measures sample purity and false discovery rates, calculates the resulting rate estimation bias, and provides background-subtracted decontaminated rate recovery with variance inflation factors (VIF).

---

## Summary of Results

In wide-field surveys (e.g. ZTF and Rubin LSST), rapid transients satisfying kilonova color and fade rate criteria contaminate candidate samples. The table below summarizes contamination and rate recovery metrics for a representative 3-year survey with LVK fiducial rates ($R_\mathrm{true} = 45.0\,\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}$):

| Population | Volumetric Rate | Expected Detections | Sample Share | Role |
| :--- | :--- | :--- | :--- | :--- |
| **BNS Kilonova (Signal)** | $45.0\,\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}$ | **18.5 events** | **17.4%** | Target Signal ($\epsilon_\mathrm{raw} = 0.35\%$) |
| **SN IIb Shock Cooling** | $1200.0\,\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}$ | **42.0 events** | **39.6%** | Primary Imposter |
| **FBOT** | $500.0\,\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}$ | **14.5 events** | **13.7%** | Fast Blue Optical Transient |
| **Subluminous SN Iax** | $3000.0\,\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}$ | **12.0 events** | **11.3%** | Fast WD Deflagration |
| **CSM Interaction** | $250.0\,\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}$ | **8.5 events** | **8.0%** | Dense Circumstellar Wind/Shell |
| **Relativistic Shocked Cocoon** | $30.0\,\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}$ | **6.2 events** | **5.8%** | Jet Breakout Cooling |
| **Orphan GRB Afterglow** | $100.0\,\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}$ | **4.4 events** | **4.1%** | Off-Axis Relativistic Blastwave |
| **Total Candidates** | — | **106.1 events** | **100.0%** | Combined Trigger Pool |

### Contamination & Rate Recovery Impact

| Metric | Naive (Ignoring Imposters) | Decontaminated Framework | Description |
| :--- | :--- | :--- | :--- |
| **Sample Purity ($\mathcal{P}$)** | $100\%$ (assumed) | **$17.4\%$** | Precision $S / (S + I)$ |
| **False Discovery Rate (FDR)** | $0\%$ (assumed) | **$82.6\%$** | Contamination fraction $I / (S + I)$ |
| **Effective Detection Efficiency ($\epsilon_\mathrm{pure}$)** | $0.35\%$ | **$0.06\%$** | True signal yield $\epsilon_\mathrm{raw} \times \mathcal{P}$ |
| **Recovered Volumetric Rate** | **$258.2\,\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}$** | **$45.0\,\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}$** | Unbiased recovery of true rate |
| **Rate Overestimation Bias** | **$+473.5\%$ (+213.1)** | **$0.0\%$ (consistent within $1\sigma$)** | Naive rate is biased high by $>5.7\times$ |
| **Variance Inflation Factor (VIF)** | $1.0\times$ (false precision) | **$18.9\times$** | Error interval expanded by $4.35\times$ |
| **$1\sigma$ Confidence Interval** | $[233.1, 283.2]$ | **$[0.0, 90.5]$** | Reflects imposter subtraction variance |

---

## Background & Problem Statement

Fast, blue, optical transients discovered in survey data without a gravitational-wave trigger (or within wide GW localization skymaps spanning thousands of square degrees) must be separated from foreground and background contaminants.

The volumetric rates of stripped-envelope core-collapse supernovae (SNe IIb, $\sim 1200\,\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}$) and FBOTs ($\sim 500\text{--}2000\,\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}$) dwarf the local kilonova rate ($\sim 10\text{--}100\,\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}$). Early shock-cooling emission in SNe IIb peaks at $m \approx 18\text{--}21$ mag with initial blue colors ($g - r \sim -0.3$) and fast fade rates ($\sim 0.5\text{--}1$ mag/day), mimicking kilonova signatures.

Ignoring these contaminants leads to two severe statistical failures:
1. **Severe Rate Overestimation**: Every detected imposter is treated as a kilonova candidate, causing rate estimates to exceed the true rate by $200\text{--}500\%$.
2. **False Overconfidence**: Naive Poisson confidence intervals shrink as $1/\sqrt{N_\mathrm{candidates}}$, producing narrow confidence bounds around an incorrect, biased rate.

---

## Methodology & Formulation

### 1. Purity & False Discovery Rate
For expected signal detections $S$ and total imposter detections $I = \sum_j I_j$:

$$\mathcal{P} = \frac{S}{S + I}, \quad \mathrm{FDR} = 1 - \mathcal{P} = \frac{I}{S + I}$$

The effective (pure) detection efficiency is scaled by the sample purity:

$$\epsilon_\mathrm{pure} = \epsilon_\mathrm{raw} \times \mathcal{P}$$

### 2. Decontaminated Rate Recovery
The naive recovered rate from total candidate counts is:

$$R_\mathrm{naive} = \frac{S + I}{\langle VT \rangle_\mathrm{eff}}$$

The background-subtracted decontaminated rate subtracts estimated imposter counts $\hat{I}$:

$$\hat{R}_\mathrm{decon} = \frac{(S + I) - \hat{I}}{\langle VT \rangle_\mathrm{eff}} = \frac{S}{\langle VT \rangle_\mathrm{eff}} = R_\mathrm{true}$$

### 3. Variance Inflation Factor (VIF)
Imposter subtraction introduces additional statistical Poisson noise plus systematic rate uncertainty $\sigma_{\mathrm{sys},j}$:

$$\mathrm{Var}(S) = S, \quad \mathrm{Var}(I) = \sum_j \left[ I_j + (\sigma_{\mathrm{sys},j} I_j)^2 \right]$$

$$\mathrm{VIF} = 1 + \frac{\mathrm{Var}(I)}{\mathrm{Var}(S)}$$

The uncertainty interval on the decontaminated rate widens by $\sqrt{\mathrm{VIF}}$, appropriately capturing both candidate Poisson fluctuations and imposter modeling uncertainties.

### 4. Follow-Up Resource Budget Optimization
When spectroscopic follow-up capacity is constrained to $B$ targets:

$$\text{Confirmed Kilonova Yield} = \min(B \times \mathcal{P}, S)$$
$$\text{Wasted Imposter Triggers} = B \times \mathrm{FDR}$$

---

## Python API Usage

### Direct Multi-Population Simulation

```python
from survey_sim import (
    load_ztf_survey,
    SimulationPipeline,
    DetectionCriteria,
    KilonovaPopulation,
    MetzgerKNModel,
    SupernovaIIbPopulation,
    FiestaShockCoolingModel,
    FbotPopulation,
    ParametricModel,
)

# 1. Load survey and criteria
survey = load_ztf_survey(nside=64)
criteria = DetectionCriteria(min_detections=2, snr_threshold=5.0)

# 2. Build multi-population pipeline
# Using survey="ztf" automatically configures ZTF filter transmission curves (ztfg, ztfr, ztfi)
pipeline = SimulationPipeline(
    survey=survey,
    populations=[
        KilonovaPopulation(rate=50.0, z_max=0.15, peak_abs_mag=-16.0),
        SupernovaIIbPopulation(rate=1200.0, z_max=0.15, peak_abs_mag=-16.5),
        FbotPopulation(rate=500.0, z_max=0.20, peak_abs_mag=-17.0),
    ],
    models={
        "Kilonova": MetzgerKNModel(),
        "SupernovaIIb": FiestaShockCoolingModel(survey="ztf"),
        "FBOT": ParametricModel().with_model("Bazin"),
    },
    detection=criteria,
    n_transients=10000,
    seed=42,
)

# 3. Execute simulation
result = pipeline.run()

# 4. Extract contamination & decontaminated rate analysis
analysis = result.analyze_imposters(
    signal_type="Kilonova",
    default_sys_err=0.20,
    followup_budget=20.0,
)

print(analysis)
```

### Analytical Forecasts & Serialization

You can construct an imposter analysis directly from built-in survey presets (`"ztf"` or `"lsst"`/`"rubin"`) without having to manually specify detection volumes and imposter rates:

```python
from survey_sim import KilonovaImposterAnalysis

# 1. Quick survey-tuned analysis (defaults to LVK median rate of 45 Gpc^-3 yr^-1)
ztf_analysis = KilonovaImposterAnalysis.for_survey("ztf", volumetric_rate=45.0)
print(f"ZTF Purity: {ztf_analysis.purity:.1%}")
print(f"ZTF Expected KNe: {ztf_analysis.signal_detections:.2f}")

lsst_analysis = KilonovaImposterAnalysis.for_survey("lsst", volumetric_rate=45.0)
print(f"Rubin LSST Purity: {lsst_analysis.purity:.1%}")
print(f"Rubin LSST Expected KNe: {lsst_analysis.signal_detections:.1f}")
```

Or provide custom event forecasts:

```python
from survey_sim import KilonovaImposterAnalysis
from survey_sim.serialization import save_result, load_result

# Construct analytical analysis from event forecasts:
analysis = KilonovaImposterAnalysis.from_counts(
    signal_name="BNS Kilonova",
    volumetric_rate=45.0,
    expected_detections=18.5,
    effective_vt=0.411,
    overall_efficiency=0.0035,
    imposters=[
        ("SN IIb Shock Cooling", 42.0, 0.25),
        ("FBOT", 14.5, 0.30),
        ("Relativistic Shocked Cocoon", 6.2, 0.35),
        ("Subluminous SN Iax", 12.0, 0.30),
        ("CSM Interaction", 8.5, 0.30),
        ("Orphan GRB Afterglow", 4.4, 0.40),
    ],
    followup_budget=20.0,
)

# Lossless JSON serialization:
save_result(analysis, "imposter_analysis.json")
restored = load_result("imposter_analysis.json")

print(f"Purity: {restored.purity:.1%}")
print(f"Decontaminated Rate: {restored.decontaminated_recovered_rate:.2f} Gpc^-3 yr^-1")
print(f"VIF: {restored.variance_inflation_factor:.2f}x")
```
