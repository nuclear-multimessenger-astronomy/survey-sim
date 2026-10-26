"""Type stubs for the survey_sim native module."""

from typing import Optional

class SurveyStore:
    n_observations: int
    mjd_range: tuple[float, float]
    duration_years: float
    bands: list[str]

    @staticmethod
    def from_rubin(db_path: str, nside: int = 64) -> "SurveyStore": ...
    @staticmethod
    def from_ztf(csv_path: str, nside: int = 64) -> "SurveyStore": ...
    @staticmethod
    def from_python(observations: list[dict], nside: int = 64) -> "SurveyStore": ...

class KilonovaPopulation:
    def __init__(self, rate: float = 1000.0, z_max: float = 0.3, peak_abs_mag: float = -16.0) -> None: ...

class SupernovaIaPopulation:
    def __init__(self, rate: float = 30000.0, z_max: float = 1.0, peak_abs_mag: float = -19.3) -> None: ...

class SupernovaIIPopulation:
    def __init__(self, rate: float = 70000.0, z_max: float = 0.5, peak_abs_mag: float = -17.0) -> None: ...

class SupernovaIIbPopulation:
    def __init__(self, rate: float = 1200.0, z_max: float = 0.15, peak_abs_mag: float = -16.5) -> None: ...

class TdePopulation:
    def __init__(self, rate: float = 100.0, z_max: float = 0.5, peak_abs_mag: float = -20.0) -> None: ...

class MetzgerKNModel:
    def __init__(self, peak_abs_mag: float = -16.0) -> None: ...

class DetectionCriteria:
    def __init__(
        self,
        min_detections: int = 2,
        min_bands: int = 1,
        min_per_band: int = 1,
        max_timespan_days: float = 30.0,
        snr_threshold: float = 5.0,
    ) -> None: ...

class DetectionResult:
    detected: bool
    n_detections: int
    n_bands_detected: int
    first_detection_mjd: Optional[float]
    last_detection_mjd: Optional[float]
    detections_per_band: dict[str, int]

class SimulationPipeline:
    def __init__(
        self,
        survey: SurveyStore,
        populations: list,
        models: dict[str, object],
        detection: DetectionCriteria,
        n_transients: int = 100000,
        seed: int = 42,
    ) -> None: ...
    def run(self) -> "SimulationResult": ...

class SimulationResult:
    n_simulated: int
    n_detected: int
    rate_summaries: list["RateSummary"]
    def analyze_imposters(
        self,
        signal_type: str = "Kilonova",
        default_sys_err: float = 0.20,
        followup_budget: Optional[float] = None,
    ) -> "KilonovaImposterAnalysis": ...

class RateSummary:
    transient_type: str
    volumetric_rate: float
    detections_per_year: float
    detections_total: float
    overall_efficiency: float
    def analyze_imposters(
        self,
        imposter_counts: Optional[list[tuple[str, float, Optional[float]]]] = None,
        imposter_summaries: Optional[list["RateSummary"]] = None,
        default_sys_err: float = 0.20,
        followup_budget: Optional[float] = None,
    ) -> "KilonovaImposterAnalysis": ...

class KilonovaImposterAnalysis:
    signal_name: str
    signal_detections: float
    total_imposters: float
    total_candidates: float
    imposter_to_signal_ratio: float
    purity: float
    false_discovery_rate: float
    pure_efficiency: float
    raw_efficiency: float
    f1_score: float
    significance_fom: float
    naive_recovered_rate: float
    rate_bias_absolute: float
    rate_bias_fractional: float
    decontaminated_recovered_rate: float
    decontaminated_1sig_interval: tuple[float, float]
    decontaminated_2sig_interval: tuple[float, float]
    variance_inflation_factor: float
    uncertainty_inflation_factor: float
    imposter_breakdown: list[tuple[str, float]]
    @staticmethod
    def from_counts(
        signal_name: str,
        volumetric_rate: float,
        expected_detections: float,
        effective_vt: float,
        overall_efficiency: float,
        imposters: list[tuple[str, float, Optional[float]]],
        followup_budget: Optional[float] = None,
    ) -> "KilonovaImposterAnalysis": ...
    @staticmethod
    def from_simulation_result(
        result: SimulationResult,
        signal_type: str = "Kilonova",
        default_sys_err: float = 0.20,
        followup_budget: Optional[float] = None,
    ) -> "KilonovaImposterAnalysis": ...
    @staticmethod
    def from_rate_summaries(
        rate_summaries: list[RateSummary],
        signal_type: str = "Kilonova",
        default_sys_err: float = 0.20,
        followup_budget: Optional[float] = None,
    ) -> "KilonovaImposterAnalysis": ...
    def f_beta(self, beta: float) -> float: ...
    def with_followup_budget(self, budget: float) -> "KilonovaImposterAnalysis": ...
    def to_dict(self) -> dict: ...
