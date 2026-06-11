"""Serialization for survey-sim result objects."""
import json
from pathlib import Path
from typing import Any, Dict, Union
import survey_sim


def save_result(result: Any, filepath: Union[str, Path], overwrite: bool = False) -> None:
    """Save a survey-sim result object to JSON."""
    filepath = Path(filepath)
    if filepath.exists() and not overwrite:
        raise FileExistsError(f"File exists: {filepath}")
    
    if isinstance(result, survey_sim.SimulationResult):
        data = _serialize_simulation_result(result)
    elif isinstance(result, survey_sim.TooSimulationResult):
        data = _serialize_too_simulation_result(result)
    elif isinstance(result, survey_sim.CoverageResult3D):
        data = _serialize_coverage_result_3d(result)
    elif isinstance(result, survey_sim.CoverageResult):
        data = _serialize_coverage_result(result)
    else:
        raise ValueError(f"Unsupported: {type(result)}")
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_result(filepath: Union[str, Path]) -> Any:
    """Load a saved result from JSON."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Not found: {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    result_type = data.get("type")
    if result_type == "SimulationResult":
        return _deserialize_simulation_result(data)
    elif result_type == "TooSimulationResult":
        return _deserialize_too_simulation_result(data)
    elif result_type == "CoverageResult3D":
        return _deserialize_coverage_result_3d(data)
    elif result_type == "CoverageResult":
        return _deserialize_coverage_result(data)
    else:
        raise ValueError(f"Unknown: {result_type}")


def _serialize_float(val: float):
    if val != val: return "NaN"
    if val == float("inf"): return "Infinity"
    if val == float("-inf"): return "-Infinity"
    return val


def _deserialize_float(val):
    if isinstance(val, str):
        if val == "NaN": return float("nan")
        if val == "Infinity": return float("inf")
        if val == "-Infinity": return float("-inf")
    return float(val)


def _serialize_dict_floats(d):
    result = {}
    for k, v in d.items():
        if isinstance(v, float): result[k] = _serialize_float(v)
        elif isinstance(v, dict): result[k] = _serialize_dict_floats(v)
        elif isinstance(v, list): result[k] = [_serialize_float(x) if isinstance(x, float) else x for x in v]
        else: result[k] = v
    return result


def _deserialize_dict_floats(d):
    result = {}
    for k, v in d.items():
        if isinstance(v, (float, str)):
            try: result[k] = _deserialize_float(v)
            except: result[k] = v
        elif isinstance(v, dict): result[k] = _deserialize_dict_floats(v)
        elif isinstance(v, list): result[k] = [_deserialize_float(x) if isinstance(x, (float, str)) else x for x in v]
        else: result[k] = v
    return result


def _serialize_rate_summary(rs):
    return _serialize_dict_floats({
        "transient_type": rs.transient_type,
        "volumetric_rate": rs.volumetric_rate,
        "detections_per_year": rs.detections_per_year,
        "detections_total": rs.detections_total,
        "overall_efficiency": rs.overall_efficiency,
        "n_simulated": rs.n_simulated,
        "n_detected": rs.n_detected,
        "z_max": rs.z_max,
        "survey_omega_sr": rs.survey_omega_sr,
        "survey_duration_years": rs.survey_duration_years,
        "effective_vt_gpc3_yr": rs.effective_vt_gpc3_yr,
    })


def _serialize_detected_source(source):
    times, mags, errs, bands = source.photometry
    nd_times, nd_depths, nd_bands = source.non_detections
    return _serialize_dict_floats({
        "z": source.z,
        "peak_abs_mag": source.peak_abs_mag,
        "t_exp": source.t_exp,
        "transient_type": source.transient_type,
        "true_params": dict(source.true_params),
        "photometry_times": list(times),
        "photometry_mags": list(mags),
        "photometry_errs": list(errs),
        "photometry_bands": list(bands),
        "non_detections_times": list(nd_times),
        "non_detections_depths": list(nd_depths),
        "non_detections_bands": list(nd_bands),
    })


def _serialize_simulation_result(result):
    return _serialize_dict_floats({
        "type": "SimulationResult",
        "n_simulated": result.n_simulated,
        "n_detected": result.n_detected,
        "rate_summaries": [_serialize_rate_summary(rs) for rs in result.rate_summaries],
        "detected_sources": [_serialize_detected_source(s) for s in result.sources()],
    })


def _serialize_too_simulation_result(result):
    return _serialize_dict_floats({
        "type": "TooSimulationResult",
        "strategy_name": result.strategy_name,
        "n_events": result.n_events,
        "n_detected": result.n_detected,
        "efficiency": result.efficiency,
        "detected": list(result.detected),
        "distances": list(result.distances),
        "areas_90": list(result.areas_90),
        "n_detections_per_event": list(result.n_detections_per_event),
    })


def _serialize_coverage_result(result):
    return _serialize_dict_floats({
        "type": "CoverageResult",
        "prob_2d": result.prob_2d,
        "area_deg2": result.area_deg2,
        "n_pixels": result.n_pixels,
        "covered": list(result.covered),
    })


def _serialize_coverage_result_3d(result):
    return _serialize_dict_floats({
        "type": "CoverageResult3D",
        "prob_2d": result.prob_2d,
        "prob_3d": result.prob_3d,
        "area_deg2": result.area_deg2,
        "n_pixels": result.n_pixels,
        "covered": list(result.covered),
        "best_d_max": list(result.best_d_max),
    })


def _deserialize_simulation_result(data):
    rate_summaries = [_MockRateSummary(_deserialize_dict_floats(rs)) for rs in data["rate_summaries"]]
    detected_sources = [_MockDetectedSource(_deserialize_dict_floats(s)) for s in data["detected_sources"]]
    return _MockSimulationResult(data["n_simulated"], data["n_detected"], rate_summaries, detected_sources)


def _deserialize_too_simulation_result(data):
    data = _deserialize_dict_floats(data)
    return _MockTooSimulationResult(data["strategy_name"], data["n_events"], data["n_detected"], data["efficiency"], data["detected"], data["distances"], data["areas_90"], data["n_detections_per_event"])


def _deserialize_coverage_result(data):
    data = _deserialize_dict_floats(data)
    return _MockCoverageResult(data["prob_2d"], data["area_deg2"], data["n_pixels"], data["covered"])


def _deserialize_coverage_result_3d(data):
    data = _deserialize_dict_floats(data)
    return _MockCoverageResult3D(data["prob_2d"], data["prob_3d"], data["area_deg2"], data["n_pixels"], data["covered"], data["best_d_max"])


class _MockRateSummary:
    def __init__(self, data):
        for k, v in data.items(): setattr(self, k, v)
    def upper_limit(self, confidence_level=0.90, n_observed=0):
        from survey_sim.efficiency.rates import poisson_upper_limit
        n_upper = poisson_upper_limit(n_observed, confidence_level)
        rate_upper = n_upper / self.effective_vt_gpc3_yr if self.effective_vt_gpc3_yr > 0 else float("inf")
        return _MockRateUpperLimit(self.transient_type, n_observed, confidence_level, n_upper, self.effective_vt_gpc3_yr, rate_upper, self.survey_duration_years, self.survey_omega_sr)
    def __repr__(self): return f"RateSummary({self.transient_type}, {self.volumetric_rate:.3e})"


class _MockRateUpperLimit:
    def __init__(self, transient_type, n_observed, confidence_level, n_upper, effective_vt, rate_upper, duration, omega):
        self.transient_type = transient_type
        self.n_observed = n_observed
        self.confidence_level = confidence_level
        self.n_upper = n_upper
        self.effective_vt_gpc3_yr = effective_vt
        self.rate_upper = rate_upper
        self.survey_duration_years = duration
        self.survey_omega_sr = omega


class _MockDetectedSource:
    def __init__(self, data):
        self.n_obs = len(data["photometry_times"])
        self.n_non_detections = len(data["non_detections_times"])
        self._photometry = (data["photometry_times"], data["photometry_mags"], data["photometry_errs"], data["photometry_bands"])
        self._non_detections = (data["non_detections_times"], data["non_detections_depths"], data["non_detections_bands"])
        for k in ["z", "peak_abs_mag", "t_exp", "transient_type", "true_params"]:
            if k in data: setattr(self, k, data[k])
    @property
    def photometry(self): return self._photometry
    @property
    def non_detections(self): return self._non_detections
    def __repr__(self): return f"DetectedSource({self.transient_type})"


class _MockSimulationResult:
    def __init__(self, n_sim, n_det, rate_summaries, detected_sources):
        self.n_simulated = n_sim
        self.n_detected = n_det
        self.rate_summaries = rate_summaries
        self.detected_sources = detected_sources
    def get_source(self, idx):
        if 0 <= idx < len(self.detected_sources): return self.detected_sources[idx]
        raise IndexError(f"index {idx}")
    def sources(self): return self.detected_sources
    @property
    def n_sources(self): return len(self.detected_sources)
    def __repr__(self): return f"SimulationResult(sim={self.n_simulated}, det={self.n_detected})"
    def __str__(self):
        eff = self.n_detected / max(self.n_simulated, 1)
        s = f"SimulationResult\n  n_simulated: {self.n_simulated}\n  n_detected: {self.n_detected}\n  efficiency: {eff:.4f}\n  sources: {len(self.detected_sources)}\n"
        if self.rate_summaries:
            s += "  rate_summaries:\n"
            for rs in self.rate_summaries:
                s += f"    {rs.transient_type}: {rs.volumetric_rate:.3e} Gpc^-3/yr\n"
        return s


class _MockTooSimulationResult:
    def __init__(self, strategy_name, n_events, n_detected, efficiency, detected, distances, areas_90, n_detections_per_event):
        self.strategy_name = strategy_name
        self.n_events = n_events
        self.n_detected = n_detected
        self.efficiency = efficiency
        self.detected = detected
        self.distances = distances
        self.areas_90 = areas_90
        self.n_detections_per_event = n_detections_per_event
    def __repr__(self): return f"TooSimulationResult('{self.strategy_name}')"


class _MockCoverageResult:
    def __init__(self, prob_2d, area_deg2, n_pixels, covered):
        self.prob_2d = prob_2d
        self.area_deg2 = area_deg2
        self.n_pixels = n_pixels
        self.covered = covered
    def __repr__(self): return f"CoverageResult({self.prob_2d:.4f})"


class _MockCoverageResult3D:
    def __init__(self, prob_2d, prob_3d, area_deg2, n_pixels, covered, best_d_max):
        self.prob_2d = prob_2d
        self.prob_3d = prob_3d
        self.area_deg2 = area_deg2
        self.n_pixels = n_pixels
        self.covered = covered
        self.best_d_max = best_d_max
    def __repr__(self): return f"CoverageResult3D({self.prob_2d:.4f}, {self.prob_3d:.4f})"
