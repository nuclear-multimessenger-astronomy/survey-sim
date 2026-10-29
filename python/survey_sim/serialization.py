"""Serialization for survey-sim result objects."""
import json
from pathlib import Path
from typing import Any, Dict, Union

try:
    import survey_sim
except ImportError:
    survey_sim = None

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
            # Apply decimal places to time-related floats or lists
            if 'time' in k or 't_exp' in k:
                if isinstance(v, float):
                    truncated_dict[k] = round(v, time_decimals)
                elif isinstance(v, (list, tuple)):
                    truncated_dict[k] = [round(x, time_decimals) if isinstance(x, float) else x for x in v]
                else:
                    truncated_dict[k] = _truncate_payload(v, time_decimals, mag_decimals, params_decimals)
            # Apply decimal places to magnitude/depth-related floats or lists
            elif 'mag' in k or 'depth' in k or 'photometry_errs' in k or 'err' in k:
                if isinstance(v, float):
                    truncated_dict[k] = round(v, mag_decimals)
                elif isinstance(v, (list, tuple)):
                    truncated_dict[k] = [round(x, mag_decimals) if isinstance(x, float) else x for x in v]
                else:
                    truncated_dict[k] = _truncate_payload(v, time_decimals, mag_decimals, params_decimals)
            # Recursively process nested dictionaries or lists
            else:
                truncated_dict[k] = _truncate_payload(v, time_decimals, mag_decimals, params_decimals)
        return truncated_dict
        
    elif isinstance(obj, list):
        return [_truncate_payload(item, time_decimals, mag_decimals, params_decimals) for item in obj]
        
    return obj


def save_result(result: Any, filepath: Union[str, Path], overwrite: bool = False, truncate: bool = True) -> None:
    """Save a survey-sim result object to JSON."""
    filepath = Path(filepath)
    if filepath.exists() and not overwrite:
        raise FileExistsError(f"File exists: {filepath}")
    
    cls_name = getattr(result, "__class__", None).__name__ if hasattr(result, "__class__") else ""
    if (survey_sim and hasattr(survey_sim, "SimulationResult") and isinstance(result, survey_sim.SimulationResult)) or isinstance(result, _MockSimulationResult) or cls_name in ("SimulationResult", "PySimulationResult", "_MockSimulationResult"):
        data = _serialize_simulation_result(result)
    elif (survey_sim and hasattr(survey_sim, "TooSimulationResult") and isinstance(result, survey_sim.TooSimulationResult)) or isinstance(result, _MockTooSimulationResult) or cls_name in ("TooSimulationResult", "PyTooSimulationResult", "_MockTooSimulationResult"):
        data = _serialize_too_simulation_result(result)
    elif (survey_sim and hasattr(survey_sim, "CoverageResult3D") and isinstance(result, survey_sim.CoverageResult3D)) or isinstance(result, _MockCoverageResult3D) or cls_name in ("CoverageResult3D", "PyCoverageResult3D", "_MockCoverageResult3D"):
        data = _serialize_coverage_result_3d(result)
    elif (survey_sim and hasattr(survey_sim, "CoverageResult") and isinstance(result, survey_sim.CoverageResult)) or isinstance(result, _MockCoverageResult) or cls_name in ("CoverageResult", "PyCoverageResult", "_MockCoverageResult"):
        data = _serialize_coverage_result(result)
    elif (
        (survey_sim and hasattr(survey_sim, "KilonovaImposterAnalysis") and isinstance(result, survey_sim.KilonovaImposterAnalysis))
        or isinstance(result, _MockKilonovaImposterAnalysis)
        or cls_name in ("PyKilonovaImposterAnalysis", "KilonovaImposterAnalysis", "_MockKilonovaImposterAnalysis")
    ):
        data = _serialize_imposter_analysis(result)
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
    elif result_type == "KilonovaImposterAnalysis":
        return _deserialize_imposter_analysis(data)
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

    if hasattr(result, "imposter_analysis") and result.imposter_analysis is not None:
        payload["imposter_analysis"] = _serialize_imposter_analysis(result.imposter_analysis)

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
    res = _MockSimulationResult(
        data["n_simulated"],
        data["n_detected"],
        rate_summaries,
        detected_sources,
        n_undetected,
        undetected_sources,
    )
    if "imposter_analysis" in data:
        res.imposter_analysis = _deserialize_imposter_analysis(data["imposter_analysis"])
    return res


def _deserialize_too_simulation_result(data):
    data = _deserialize_dict_floats(data)
    return _MockTooSimulationResult(data["strategy_name"], data["n_events"], data["n_detected"], data["efficiency"], data["detected"], data["distances"], data["areas_90"], data["n_detections_per_event"])


def _deserialize_coverage_result(data):
    data = _deserialize_dict_floats(data)
    return _MockCoverageResult(data["prob_2d"], data["area_deg2"], data["n_pixels"], data["covered"])


def _deserialize_coverage_result_3d(data):
    data = _deserialize_dict_floats(data)
    return _MockCoverageResult3D(data["prob_2d"], data["prob_3d"], data["area_deg2"], data["n_pixels"], data["covered"], data["best_d_max"])


def _serialize_imposter_analysis(analysis):
    followup_dict = None
    if hasattr(analysis, "followup_metrics"):
        fm = analysis.followup_metrics
        if callable(fm):
            fm = fm()
        if fm is not None:
            followup_dict = dict(fm)
    elif getattr(analysis, "followup_budget", None) is not None:
        followup_dict = dict(analysis.followup_budget)

    data = {
        "type": "KilonovaImposterAnalysis",
        "signal_name": getattr(analysis, "signal_name", "Kilonova"),
        "signal_rate": getattr(analysis, "signal_rate", getattr(analysis, "input_rate", 0.0)),
        "signal_detections": getattr(analysis, "signal_detections", 0.0),
        "effective_vt": getattr(analysis, "effective_vt", 0.0),
        "raw_efficiency": getattr(analysis, "raw_efficiency", 0.0),
        "pure_efficiency": getattr(analysis, "pure_efficiency", 0.0),
        "purity": getattr(analysis, "purity", 1.0),
        "false_discovery_rate": getattr(analysis, "false_discovery_rate", 0.0),
        "imposter_to_signal_ratio": getattr(analysis, "imposter_to_signal_ratio", 0.0),
        "total_imposters": getattr(analysis, "total_imposters", 0.0),
        "total_candidates": getattr(analysis, "total_candidates", getattr(analysis, "total_detections", 0.0)),
        "f1_score": getattr(analysis, "f1_score", 0.0),
        "significance_fom": getattr(analysis, "significance_fom", 0.0),
        "naive_recovered_rate": getattr(analysis, "naive_recovered_rate", 0.0),
        "rate_bias_absolute": getattr(analysis, "rate_bias_absolute", 0.0),
        "rate_bias_fractional": getattr(analysis, "rate_bias_fractional", 0.0),
        "naive_poisson_1sig_interval": list(getattr(analysis, "naive_poisson_1sig_interval", (0.0, 0.0))),
        "decontaminated_recovered_rate": getattr(analysis, "decontaminated_recovered_rate", 0.0),
        "decontaminated_1sig_interval": list(getattr(analysis, "decontaminated_1sig_interval", (0.0, 0.0))),
        "decontaminated_2sig_interval": list(getattr(analysis, "decontaminated_2sig_interval", (0.0, 0.0))),
        "variance_inflation_factor": getattr(analysis, "variance_inflation_factor", 1.0),
        "uncertainty_inflation_factor": getattr(analysis, "uncertainty_inflation_factor", 1.0),
        "consistent_1sig": bool(getattr(analysis, "consistent_1sig", True)),
        "consistent_2sig": bool(getattr(analysis, "consistent_2sig", True)),
        "imposter_breakdown": [list(x) for x in getattr(analysis, "imposter_breakdown", [])],
    }
    if followup_dict is not None:
        data["followup_budget"] = followup_dict
    return _serialize_dict_floats(data)


def _deserialize_imposter_analysis(data):
    return _MockKilonovaImposterAnalysis(_deserialize_dict_floats(data))


def _compute_imposter_analysis(signal_summary, imposter_counts=None, imposter_summaries=None, default_sys_err=0.20, followup_budget=None):
    n_kn = float(getattr(signal_summary, "detections_total", 0.0))
    raw_eff = float(getattr(signal_summary, "overall_efficiency", getattr(signal_summary, "efficiency", 0.0)))
    rate_kn = float(getattr(signal_summary, "volumetric_rate", 0.0))
    vt = float(getattr(signal_summary, "effective_vt_gpc3_yr", 0.0))
    if vt == 0.0 and rate_kn > 0.0:
        vt = n_kn / rate_kn

    imposter_breakdown = []
    n_imp_total = 0.0
    var_sys_total = 0.0

    if imposter_counts:
        for item in imposter_counts:
            name = str(item[0])
            cnt = float(item[1])
            err = float(item[2]) if len(item) > 2 and item[2] is not None else default_sys_err
            imposter_breakdown.append((name, cnt))
            n_imp_total += cnt
            sig_sys = err * cnt
            var_sys_total += sig_sys * sig_sys

    if imposter_summaries:
        for imp in imposter_summaries:
            name = str(getattr(imp, "transient_type", "Imposter"))
            cnt = float(getattr(imp, "detections_total", 0.0))
            imposter_breakdown.append((name, cnt))
            n_imp_total += cnt
            sig_sys = default_sys_err * cnt
            var_sys_total += sig_sys * sig_sys

    n_tot = n_kn + n_imp_total
    r_is = (n_imp_total / n_kn) if n_kn > 0.0 else float("inf")
    purity = (n_kn / n_tot) if n_tot > 0.0 else 1.0
    fdr = 1.0 - purity
    pure_eff = raw_eff * purity
    f1 = (2.0 * purity * raw_eff / (purity + raw_eff)) if (purity + raw_eff) > 0.0 else 0.0
    fom = (n_kn / (n_tot ** 0.5)) if n_tot > 0.0 else 0.0

    naive_rate = (n_tot / vt) if vt > 0.0 else 0.0
    rate_bias_abs = (n_imp_total / vt) if vt > 0.0 else 0.0
    rate_bias_frac = r_is
    naive_sigma = (n_tot ** 0.5 / vt) if (n_tot > 0.0 and vt > 0.0) else 0.0
    naive_interval = (max(0.0, naive_rate - naive_sigma), naive_rate + naive_sigma)

    decontam_rate = (max(0.0, n_kn) / vt) if vt > 0.0 else 0.0
    count_var = n_tot + n_imp_total + var_sys_total
    decontam_sigma = (count_var ** 0.5 / vt) if vt > 0.0 else 0.0
    decontam_1sig = (max(0.0, decontam_rate - decontam_sigma), decontam_rate + decontam_sigma)
    decontam_2sig = (max(0.0, decontam_rate - 2.0 * decontam_sigma), decontam_rate + 2.0 * decontam_sigma)

    vif = (count_var / n_kn) if n_kn > 0.0 else float("inf")
    uif = vif ** 0.5 if vif != float("inf") else float("inf")

    c1 = decontam_1sig[0] <= rate_kn <= decontam_1sig[1]
    c2 = decontam_2sig[0] <= rate_kn <= decontam_2sig[1]

    data = {
        "signal_name": getattr(signal_summary, "transient_type", "Kilonova"),
        "signal_rate": rate_kn,
        "signal_detections": n_kn,
        "effective_vt": vt,
        "raw_efficiency": raw_eff,
        "pure_efficiency": pure_eff,
        "purity": purity,
        "false_discovery_rate": fdr,
        "imposter_to_signal_ratio": r_is,
        "total_imposters": n_imp_total,
        "total_candidates": n_tot,
        "f1_score": f1,
        "significance_fom": fom,
        "naive_recovered_rate": naive_rate,
        "rate_bias_absolute": rate_bias_abs,
        "rate_bias_fractional": rate_bias_frac,
        "naive_poisson_1sig_interval": naive_interval,
        "decontaminated_recovered_rate": decontam_rate,
        "decontaminated_1sig_interval": decontam_1sig,
        "decontaminated_2sig_interval": decontam_2sig,
        "variance_inflation_factor": vif,
        "uncertainty_inflation_factor": uif,
        "consistent_1sig": c1,
        "consistent_2sig": c2,
        "imposter_breakdown": imposter_breakdown,
    }

    if followup_budget is not None:
        alloc = min(1.0, followup_budget / n_tot) if n_tot > 0.0 else 1.0
        data["followup_budget"] = {
            "budget": float(followup_budget),
            "total_candidates": n_tot,
            "allocation_fraction": alloc,
            "expected_confirmed_kn": n_kn * alloc,
            "confirmed_efficiency": raw_eff * alloc,
            "wasted_imposter_followups": n_imp_total * alloc,
            "is_budget_exceeded": n_tot > followup_budget,
        }

    return _MockKilonovaImposterAnalysis(data)


class _MockKilonovaImposterAnalysis:
    def __init__(self, data):
        for k, v in data.items():
            setattr(self, k, v)
        if "imposter_breakdown" in data:
            self.imposter_breakdown = [(x[0], float(x[1])) for x in data["imposter_breakdown"]]
        if "naive_poisson_1sig_interval" in data:
            self.naive_poisson_1sig_interval = tuple(data["naive_poisson_1sig_interval"])
        if "decontaminated_1sig_interval" in data:
            self.decontaminated_1sig_interval = tuple(data["decontaminated_1sig_interval"])
        if "decontaminated_2sig_interval" in data:
            self.decontaminated_2sig_interval = tuple(data["decontaminated_2sig_interval"])

    def f_beta(self, beta: float) -> float:
        beta_sq = beta * beta
        p = self.purity
        r = self.raw_efficiency
        denom = beta_sq * p + r
        return (1.0 + beta_sq) * (p * r) / denom if denom > 0.0 else 0.0

    def followup_metrics(self):
        return getattr(self, "followup_budget", None)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def with_followup_budget(self, budget: float):
        alloc = min(1.0, budget / self.total_candidates) if self.total_candidates > 0.0 else 1.0
        self.followup_budget = {
            "budget": float(budget),
            "total_candidates": self.total_candidates,
            "allocation_fraction": alloc,
            "expected_confirmed_kn": self.signal_detections * alloc,
            "confirmed_efficiency": self.raw_efficiency * alloc,
            "wasted_imposter_followups": self.total_imposters * alloc,
            "is_budget_exceeded": self.total_candidates > budget,
        }
        return self

    def __repr__(self):
        return (
            f"KilonovaImposterAnalysis(signal='{self.signal_name}', "
            f"N_kn={self.signal_detections:.1f}, N_imp={self.total_imposters:.1f}, "
            f"purity={self.purity * 100:.1f}%, eff_pure={self.pure_efficiency * 100:.2f}%, "
            f"F1={self.f1_score:.3f}, bias=+{self.rate_bias_fractional * 100:.1f}%)"
        )

    def __str__(self):
        return self.__repr__()


class _MockRateSummary:
    def __init__(self, data):
        for k, v in data.items(): setattr(self, k, v)

    def analyze_imposters(self, imposter_counts=None, imposter_summaries=None, default_sys_err=0.20, followup_budget=None):
        return _compute_imposter_analysis(
            self,
            imposter_counts=imposter_counts,
            imposter_summaries=imposter_summaries,
            default_sys_err=default_sys_err,
            followup_budget=followup_budget,
        )

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
    def analyze_imposters(self, signal_type="Kilonova", default_sys_err=0.20, followup_budget=None):
        signal_rs = next((rs for rs in self.rate_summaries if rs.transient_type == signal_type), None)
        if signal_rs is None:
            available = [rs.transient_type for rs in self.rate_summaries]
            raise ValueError(f"Signal transient type '{signal_type}' not found in simulation rate summaries. Available: {available}")
        imposter_summaries = [rs for rs in self.rate_summaries if rs.transient_type != signal_type]
        analysis = signal_rs.analyze_imposters(
            imposter_summaries=imposter_summaries,
            default_sys_err=default_sys_err,
            followup_budget=followup_budget,
        )
        self.imposter_analysis = analysis
        return analysis
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
