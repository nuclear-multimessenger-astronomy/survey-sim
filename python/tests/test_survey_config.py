"""Tests for survey band mapping and survey presets across fiesta models."""

import pytest
from survey_sim import (
    ZTF_BAND_MAP,
    LSST_BAND_MAP,
    SURVEY_BAND_MAPS,
    resolve_band_map,
    KilonovaImposterAnalysis,
)


def test_resolve_band_map_defaults():
    # Default is ZTF
    bm = resolve_band_map()
    assert bm == ZTF_BAND_MAP
    assert bm["g"] == "ztfg"
    assert bm["r"] == "ztfr"
    assert bm["i"] == "ztfi"


def test_resolve_band_map_ztf_and_lsst():
    ztf = resolve_band_map(survey="ztf")
    assert ztf == ZTF_BAND_MAP

    lsst = resolve_band_map(survey="lsst")
    assert lsst == LSST_BAND_MAP

    rubin = resolve_band_map(survey="rubin")
    assert rubin == LSST_BAND_MAP


def test_resolve_band_map_custom():
    custom = {"g": "my_g", "r": "my_r"}
    bm = resolve_band_map(band_map=custom)
    assert bm == custom


def test_resolve_band_map_invalid():
    with pytest.raises(ValueError, match="Unknown survey 'unknown'"):
        resolve_band_map(survey="unknown")


def test_imposter_analysis_for_survey_presets():
    ztf_analysis = KilonovaImposterAnalysis.for_survey("ztf", volumetric_rate=45.0)
    assert ztf_analysis.signal_name == "BNS Kilonova"
    assert ztf_analysis.signal_detections < 1.0  # ZTF 3yr expected ~0.14
    assert ztf_analysis.total_imposters > 0.0
    assert 0.0 < ztf_analysis.purity < 1.0

    lsst_analysis = KilonovaImposterAnalysis.for_survey("lsst", volumetric_rate=45.0)
    assert lsst_analysis.signal_detections > 10.0  # LSST 10yr expected ~18.5
    assert lsst_analysis.total_imposters > 50.0
    assert 0.0 < lsst_analysis.purity < 1.0
