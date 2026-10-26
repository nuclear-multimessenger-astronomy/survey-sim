"""Survey band and instrument configurations for survey-sim.

Provides unified passband mappings and survey profiles (e.g. ZTF, Rubin/LSST)
so that all Fiesta surrogate models and imposter efficiency pipelines can be
configured by simply passing ``survey="ztf"`` or ``survey="lsst"``.
"""

from typing import Dict, List, Optional, Union

ZTF_BAND_MAP: Dict[str, str] = {
    "g": "ztfg",
    "r": "ztfr",
    "i": "ztfi",
}

LSST_BAND_MAP: Dict[str, str] = {
    "u": "lsstu",
    "g": "lsstg",
    "r": "lsstr",
    "i": "lssti",
    "z": "lsstz",
    "y": "lssty",
}

SURVEY_BAND_MAPS: Dict[str, Dict[str, str]] = {
    "ztf": ZTF_BAND_MAP,
    "lsst": LSST_BAND_MAP,
    "rubin": LSST_BAND_MAP,
}

SURVEY_PRESETS: Dict[str, dict] = {
    "ztf": {
        "name": "ZTF",
        "band_map": ZTF_BAND_MAP,
        "default_bands": ["g", "r", "i"],
        "duration_years": 3.09,
        "effective_vt_kn": 0.0032,  # Gpc^3 yr (ZTF Bu2026 ~150-200 Mpc)
    },
    "lsst": {
        "name": "Rubin LSST",
        "band_map": LSST_BAND_MAP,
        "default_bands": ["u", "g", "r", "i", "z", "y"],
        "duration_years": 10.0,
        "effective_vt_kn": 0.411,
    },
    "rubin": {
        "name": "Rubin LSST",
        "band_map": LSST_BAND_MAP,
        "default_bands": ["u", "g", "r", "i", "z", "y"],
        "duration_years": 10.0,
        "effective_vt_kn": 0.411,
    },
}


def resolve_band_map(
    survey: Optional[str] = None,
    band_map: Optional[Dict[str, str]] = None,
    filters: Optional[Union[List[str], Dict[str, str]]] = None,
    default_survey: str = "ztf",
) -> Dict[str, str]:
    """Resolve a unified {survey_band: fiesta_filter} mapping.

    Parameters
    ----------
    survey : str, optional
        Survey identifier (e.g. ``"ztf"`` or ``"lsst"``/``"rubin"``).
        Defaults to ``"ztf"`` if neither ``band_map`` nor ``filters`` is provided.
    band_map : dict[str, str], optional
        Explicit mapping from survey short band names to sncosmo/fiesta filter names.
    filters : list[str] or dict[str, str], optional
        Filter list or dictionary.
    default_survey : str, default "ztf"
        Fallback survey if none is explicitly specified.

    Returns
    -------
    dict[str, str]
        Mapping from short band names (e.g. ``"g"``) to fiesta filter names (e.g. ``"ztfg"``).
    """
    if band_map is not None:
        return dict(band_map)

    if isinstance(filters, dict):
        return dict(filters)

    if survey is not None:
        key = str(survey).lower().strip()
        if key in SURVEY_BAND_MAPS:
            return dict(SURVEY_BAND_MAPS[key])
        raise ValueError(
            f"Unknown survey '{survey}'. Available presets: {list(SURVEY_BAND_MAPS.keys())}"
        )

    if filters is not None and isinstance(filters, (list, tuple)):
        mapping = {}
        for f in filters:
            f_str = str(f)
            if f_str.startswith("ztf") and len(f_str) > 3:
                mapping[f_str[3:]] = f_str
            elif f_str.startswith("lsst") and len(f_str) > 4:
                mapping[f_str[4:]] = f_str
            else:
                mapping[f_str] = f_str
        return mapping

    # Default fallback
    key = default_survey.lower().strip()
    return dict(SURVEY_BAND_MAPS.get(key, ZTF_BAND_MAP))
