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

class RateSummary:
    transient_type: str
    volumetric_rate: float
    detections_per_year: float
    detections_total: float
    overall_efficiency: float
