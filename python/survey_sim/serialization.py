"""Serialization for survey-sim result objects."""
import json
from pathlib import Path
from typing import Any, Dict, Union
import survey_sim

def _format_rust_e(val, precision=3):
    """Formats floats to match Rust's scientific display (e.g., 1.115e0 instead of 1.115e+00)"""
    if val == float('inf'):
        return "inf"
    base, exp = f"{val:.{precision}e}".split('e')
    return f"{base}e{int(exp)}"

def _truncate_payload(obj: Any, time_decimals: int = 3, mag_decimals: int = 3, params_decimals: int = 5) -> Any:
    """
    Recursively traverses the payload to truncate overly precise floats.
    Targets specific keys (e.g., times, mags, depths) for custom precision.
    """
    if isinstance(obj, float):
        # Fallback for generic floats
        return round(obj, params_decimals) 
    
    elif isinstance(obj, dict):
        truncated_dict = {}
        for k, v in obj.items():
            ## skip rate_summaries
            if k == "rate_summaries":
                truncated_dict[k] = v
            # Apply decimal places to time-related floats
            if isinstance(v, float) and ('time' in k or 't_exp' in k):
                truncated_dict[k] = round(v, time_decimals)
            # Apply decimal places to magnitude/depth-related floats
            elif isinstance(v, float) and ('mag' in k or 'depth' in k or 'photometry_errs' in k):
                truncated_dict[k] = round(v, mag_decimals)
            # Recursively process nested dictionaries or lists
            else:
                truncated_dict[k] = _truncate_payload(v, time_decimals, mag_decimals)
        return truncated_dict
        
    elif isinstance(obj, list):
        return [_truncate_payload(item, time_decimals, mag_decimals) for item in obj]
        
    return obj


def save_result(result: Any, filepath: Union[str, Path], overwrite: bool = False, truncate: bool = True) -> None:
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
    
    if truncate:
        data = _truncate_payload(data)

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
        if isinstance(v, float):
            result[k] = _deserialize_float(v)
        elif isinstance(v, str) and v in {"NaN", "Infinity", "-Infinity"}:
            result[k] = _deserialize_float(v)
        elif isinstance(v, dict): result[k] = _deserialize_dict_floats(v)
        elif isinstance(v, list):
            result[k] = [
                _deserialize_float(x)
                if isinstance(x, float) or (isinstance(x, str) and x in {"NaN", "Infinity", "-Infinity"})
                else x
                for x in v
            ]
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


def _get_source_data(source, field):
    value = getattr(source, field)
    if callable(value):
        return value()
    return value


def _serialize_detected_source(source):
    photometry = _get_source_data(source, "photometry")
    non_detections = _get_source_data(source, "non_detections")
    times, mags, errs, bands = photometry
    nd_times, nd_depths, nd_bands = non_detections
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
    detected_sources = list(result.sources())
    undetected_sources = list(getattr(result, "undetected_sources", []))
    n_undetected = getattr(result, "n_undetected", len(undetected_sources))

    payload = {
        "type": "SimulationResult",
        "n_simulated": result.n_simulated,
        "n_detected": result.n_detected,
        "rate_summaries": [_serialize_rate_summary(rs) for rs in result.rate_summaries],
        "detected_sources": [_serialize_detected_source(s) for s in detected_sources],
    }

    # Keep undetected outputs optional in JSON: only include when populated.
    if n_undetected > 0 or undetected_sources:
        payload["n_undetected"] = n_undetected
        payload["undetected_sources"] = [_serialize_detected_source(s) for s in undetected_sources]

    return _serialize_dict_floats(payload)


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
    detected_sources = [_MockDetectedSource(_deserialize_dict_floats(s)) for s in data.get("detected_sources", [])]
    undetected_sources = [_MockDetectedSource(_deserialize_dict_floats(s)) for s in data.get("undetected_sources", [])]
    n_undetected = data.get("n_undetected", len(undetected_sources))
    return _MockSimulationResult(
        data["n_simulated"],
        data["n_detected"],
        rate_summaries,
        detected_sources,
        n_undetected,
        undetected_sources,
    )


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
        # Calculate the Poisson upper limit directly using scipy
        from scipy.stats import chi2
        alpha = 1.0 - confidence_level
        n_upper = 0.5 * chi2.ppf(1.0 - alpha, 2 * (n_observed + 1))
        
        rate_upper = n_upper / self.effective_vt_gpc3_yr if self.effective_vt_gpc3_yr > 0 else float("inf")
        return _MockRateUpperLimit(
            self.transient_type, 
            n_observed, 
            confidence_level, 
            n_upper, 
            self.effective_vt_gpc3_yr, 
            rate_upper, 
            self.survey_duration_years, 
            self.survey_omega_sr
        )

    def __repr__(self):
        rate_str = _format_rust_e(self.volumetric_rate, 3)
        omega_str = _format_rust_e(self.survey_omega_sr, 3)
        vt_eff_str = _format_rust_e(self.effective_vt_gpc3_yr, 3)
        
        # Grab sim/det counts from the JSON payload
        n_sim = getattr(self, 'n_simulated', 0)
        n_det = getattr(self, 'n_detected', 0)
        
        # Calculate derived values natively like Rust does
        eff = getattr(self, 'efficiency', n_det / max(n_sim, 1))
        
        # det_total = rate * VT_eff
        calc_det_total = self.volumetric_rate * self.effective_vt_gpc3_yr
        det_tot = getattr(self, 'det_total', calc_det_total)
        det_tot_str = _format_rust_e(det_tot, 3)
        
        # det_per_yr = det_total / T
        calc_det_yr = det_tot / self.survey_duration_years if self.survey_duration_years > 0 else 0
        det_yr_str = _format_rust_e(getattr(self, 'det_per_yr', calc_det_yr), 3)
        
        z_max = getattr(self, 'z_max', 0.0)

        return (f"RateSummary(type={self.transient_type}, rate={rate_str} Gpc^-3/yr, "
                f"n_sim={n_sim}, n_det={n_det}, eff={eff:.4f}, det/yr={det_yr_str}, "
                f"det_total={det_tot_str}, z_max={z_max:.2f}, omega_sr={omega_str}, "
                f"T={self.survey_duration_years:.2f} yr, VT_eff={vt_eff_str} Gpc^3 yr)")


class _MockRateUpperLimit:
    def __init__(self, transient_type, n_observed, confidence_level, n_upper, effective_vt_gpc3_yr, rate_upper, survey_duration_years, survey_omega_sr):
        self.transient_type = transient_type
        self.n_observed = n_observed
        self.confidence_level = confidence_level
        self.n_upper = n_upper
        self.effective_vt_gpc3_yr = effective_vt_gpc3_yr
        self.rate_upper = rate_upper
        self.survey_duration_years = survey_duration_years
        self.survey_omega_sr = survey_omega_sr

    def __repr__(self):
        cl_percent = int(self.confidence_level * 100)
        r_upper_str = _format_rust_e(self.rate_upper, 3)
        vt_eff_str = _format_rust_e(self.effective_vt_gpc3_yr, 3)
        
        return (f"RateUpperLimit(type={self.transient_type}, "
                f"N={self.n_observed}, "
                f"CL={cl_percent}%, "
                f"R_upper={r_upper_str} Gpc^-3/yr, "
                f"N_upper={self.n_upper:.3f}, "
                f"VT_eff={vt_eff_str} Gpc^3 yr)")


class _MockDetectedSource:
    def __init__(self, data):
        self.n_obs = len(data["photometry_times"])
        self.n_non_detections = len(data["non_detections_times"])
        self._photometry = (data["photometry_times"], data["photometry_mags"], data["photometry_errs"], data["photometry_bands"])
        self._non_detections = (data["non_detections_times"], data["non_detections_depths"], data["non_detections_bands"])
        for k in ["z", "peak_abs_mag", "t_exp", "transient_type", "true_params"]:
            if k in data: setattr(self, k, data[k])
            
    def photometry(self): return self._photometry
    def non_detections(self): return self._non_detections
    def __repr__(self):
        # Extract unique bands using the callable method
        if callable(self.photometry):
            _, _, _, bands = self.photometry()
        else:
            _, _, _, bands = self.photometry
            
        unique_bands = ",".join(sorted(set(bands)))
        return f"DetectedSource(type={self.transient_type}, z={self.z:.3f}, n_obs={self.n_obs}, bands={unique_bands})"


class _MockSimulationResult:
    def __init__(self, n_sim, n_det, rate_summaries, detected_sources, n_undetected=0, undetected_sources=None):
        self.n_simulated = n_sim
        self.n_detected = n_det
        self.n_undetected = n_undetected
        self.rate_summaries = rate_summaries
        self.detected_sources = detected_sources
        self.undetected_sources = undetected_sources if undetected_sources is not None else []
    def get_source(self, idx):
        if 0 <= idx < len(self.detected_sources): return self.detected_sources[idx]
        raise IndexError(f"index {idx}")
    def sources(self): return self.detected_sources
    @property
    def n_sources(self): return len(self.detected_sources)
    def __str__(self):
        eff = self.n_detected / max(self.n_simulated, 1)
        include_undetected = self.n_undetected > 0 or bool(self.undetected_sources)
        lines = ["SimulationResult", f"  n_simulated: {self.n_simulated}", f"  n_detected:  {self.n_detected}"]
        if include_undetected:
            lines.append(f"  n_undetected: {self.n_undetected}")
        lines.extend([
            f"  efficiency:  {eff:.4f}",
            f"  sources:     {self.n_sources}",
            "  rate_summaries:"
        ])
        for rs in self.rate_summaries:
            rate_str = _format_rust_e(rs.volumetric_rate, 3)
            omega_str = _format_rust_e(rs.survey_omega_sr, 3)
            
            n_sim = getattr(rs, 'n_simulated', 0)
            n_det = getattr(rs, 'n_detected', 0)
            rs_eff = getattr(rs, 'efficiency', n_det / max(n_sim, 1))
            
            calc_det_total = rs.volumetric_rate * rs.effective_vt_gpc3_yr
            det_tot = getattr(rs, 'det_total', calc_det_total)
            det_tot_str = _format_rust_e(det_tot, 3)
            
            calc_det_yr = det_tot / rs.survey_duration_years if rs.survey_duration_years > 0 else 0
            det_yr_str = _format_rust_e(getattr(rs, 'det_per_yr', calc_det_yr), 3)
            
            lines.append(f"    {rs.transient_type} : rate={rate_str} Gpc^-3/yr, eff={rs_eff:.4f}, "
                         f"det/yr={det_yr_str}, det_total={det_tot_str} "
                         f"(T={rs.survey_duration_years:.2f} yr, omega={omega_str} sr)")
                         
        return "\n".join(lines) + "\n"

    def __repr__(self):
        eff = self.n_detected / max(self.n_simulated, 1)
        if self.n_undetected > 0 or self.undetected_sources:
            return (
                f"SimulationResult(n_simulated={self.n_simulated}, n_detected={self.n_detected}, "
                f"n_undetected={self.n_undetected}, efficiency={eff:.4f}, sources={self.n_sources})"
            )
        return (
            f"SimulationResult(n_simulated={self.n_simulated}, n_detected={self.n_detected}, "
            f"efficiency={eff:.4f}, sources={self.n_sources})"
        )


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
