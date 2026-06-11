import importlib.util
import math
import pathlib
import sys
import tempfile
import types
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SERIALIZATION_PATH = REPO_ROOT / "python" / "survey_sim" / "serialization.py"


class RateSummary:
    def __init__(self):
        self.transient_type = "KN"
        self.volumetric_rate = float("inf")
        self.detections_per_year = 2.0
        self.detections_total = 10.0
        self.overall_efficiency = 0.5
        self.n_simulated = 20
        self.n_detected = 10
        self.z_max = 0.3
        self.survey_omega_sr = 1.2
        self.survey_duration_years = 3.0
        self.effective_vt_gpc3_yr = 4.5


class DetectedSource:
    def __init__(self):
        self.z = 0.1
        self.peak_abs_mag = -16.2
        self.t_exp = 0.5
        self.transient_type = "KN"
        self.true_params = {"x": float("nan")}
        self.photometry = ([1.0, 2.0], [20.1, 20.2], [0.1, 0.2], ["g", "r"])
        self.non_detections = ([0.5], [21.0], ["g"])


class SimulationResult:
    def __init__(self):
        self.n_simulated = 20
        self.n_detected = 10
        self.rate_summaries = [RateSummary()]
        self._detected = [DetectedSource()]

    def sources(self):
        return self._detected


class TooSimulationResult:
    def __init__(self):
        self.strategy_name = "baseline"
        self.n_events = 3
        self.n_detected = 2
        self.efficiency = 2.0 / 3.0
        self.detected = [True, False, True]
        self.distances = [100.0, 200.0, 300.0]
        self.areas_90 = [10.0, 20.0, 30.0]
        self.n_detections_per_event = [1, 0, 2]


class CoverageResult:
    def __init__(self):
        self.prob_2d = 0.5
        self.area_deg2 = 42.0
        self.n_pixels = 7
        self.covered = [1, 2, 3]


class CoverageResult3D:
    def __init__(self):
        self.prob_2d = 0.4
        self.prob_3d = 0.2
        self.area_deg2 = 24.0
        self.n_pixels = 5
        self.covered = [4, 5]
        self.best_d_max = [150.0, 180.0]


def load_serialization_module():
    if not SERIALIZATION_PATH.exists():
        raise FileNotFoundError(f"Missing serialization module: {SERIALIZATION_PATH}")

    survey_sim_stub = types.ModuleType("survey_sim")
    survey_sim_stub.SimulationResult = SimulationResult
    survey_sim_stub.TooSimulationResult = TooSimulationResult
    survey_sim_stub.CoverageResult = CoverageResult
    survey_sim_stub.CoverageResult3D = CoverageResult3D
    sys.modules["survey_sim"] = survey_sim_stub

    spec = importlib.util.spec_from_file_location("survey_sim.serialization", str(SERIALIZATION_PATH))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {SERIALIZATION_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestSerialization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_serialization_module()

    def roundtrip(self, obj):
        with tempfile.TemporaryDirectory() as td:
            outfile = pathlib.Path(td) / "result.json"
            self.mod.save_result(obj, outfile)
            self.assertTrue(outfile.exists())
            return self.mod.load_result(outfile)

    def test_simulation_result_roundtrip(self):
        out = self.roundtrip(SimulationResult())
        self.assertEqual(out.n_simulated, 20)
        self.assertEqual(out.n_detected, 10)
        self.assertEqual(len(out.rate_summaries), 1)
        self.assertTrue(math.isinf(out.rate_summaries[0].volumetric_rate))
        src = out.sources()[0]
        self.assertEqual(src.transient_type, "KN")
        self.assertTrue(math.isnan(src.true_params["x"]))

    def test_too_simulation_result_roundtrip(self):
        out = self.roundtrip(TooSimulationResult())
        self.assertEqual(out.strategy_name, "baseline")
        self.assertEqual(out.n_events, 3)
        self.assertEqual(out.detected, [True, False, True])

    def test_coverage_result_roundtrip(self):
        out = self.roundtrip(CoverageResult())
        self.assertEqual(out.prob_2d, 0.5)
        self.assertEqual(out.n_pixels, 7)

    def test_coverage_result_3d_roundtrip(self):
        out = self.roundtrip(CoverageResult3D())
        self.assertEqual(out.prob_2d, 0.4)
        self.assertEqual(out.prob_3d, 0.2)
        self.assertEqual(out.best_d_max, [150.0, 180.0])

    def test_save_result_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as td:
            outfile = pathlib.Path(td) / "result.json"
            self.mod.save_result(CoverageResult(), outfile)
            with self.assertRaises(FileExistsError):
                self.mod.save_result(CoverageResult(), outfile, overwrite=False)


if __name__ == "__main__":
    unittest.main()
