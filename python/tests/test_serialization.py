import pathlib
import tempfile
import unittest
import json

import numpy as np

# Import real simulation components
from survey_sim import (
    load_ztf_survey, KilonovaPopulation, MetzgerKNModel,
    DetectionCriteria, SimulationPipeline
)
from survey_sim.serialization import save_result, load_result

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


class TestSimulationReproducibility(unittest.TestCase):
    """
    Runs a lightweight simulation using the real Rust backend twice to 
    guarantee that the pipeline is deterministic and that the serialized 
    JSON results behave identically to live objects.
    """
    @classmethod
    def setUpClass(cls):
        cls.survey = load_ztf_survey(nside=64)

    def _run_lightweight_pipeline(self, seed: int):
        """Helper to run a fast 1000-transient simulation."""
        kn_pop = KilonovaPopulation(rate=1000.0, z_max=0.3, peak_abs_mag=-16.0)
        model = MetzgerKNModel()
        det = DetectionCriteria(min_detections=2, snr_threshold=5.0)

        pipe = SimulationPipeline(
            self.survey,
            [kn_pop],
            {"Kilonova": model},
            det,
            n_transients=1000, 
            seed=seed,
        )
        
        return pipe.run()

    def test_pipeline_collects_undetected_sources_when_requested(self):
        kn_pop = KilonovaPopulation(rate=1000.0, z_max=0.2, peak_abs_mag=-16.0)
        model = MetzgerKNModel()
        det = DetectionCriteria(min_detections=2, snr_threshold=10.0)

        pipe = SimulationPipeline(
            self.survey,
            [kn_pop],
            {"Kilonova": model},
            det,
            n_transients=200,
            seed=7,
        )

        result = pipe.run(include_undetected=True)
        self.assertTrue(hasattr(result, "n_undetected"))
        self.assertTrue(hasattr(result, "undetected_sources"))
        self.assertEqual(result.n_undetected, len(result.undetected_sources))
        self.assertGreaterEqual(result.n_undetected, 0)

    def test_serialization_handles_optional_undetected_outputs(self):
        kn_pop = KilonovaPopulation(rate=1000.0, z_max=0.2, peak_abs_mag=-16.0)
        model = MetzgerKNModel()
        det = DetectionCriteria(min_detections=2, snr_threshold=10.0)
        pipe = SimulationPipeline(
            self.survey,
            [kn_pop],
            {"Kilonova": model},
            det,
            n_transients=200,
            seed=7,
        )

        default_result = pipe.run()
        undetected_result = pipe.run(include_undetected=True)

        with tempfile.TemporaryDirectory() as td:
            default_path = pathlib.Path(td) / "default.json"
            undetected_path = pathlib.Path(td) / "with_undetected.json"

            save_result(default_result, default_path)
            save_result(undetected_result, undetected_path)

            with open(default_path, "r") as f:
                default_payload = json.load(f)
            with open(undetected_path, "r") as f:
                undetected_payload = json.load(f)

            self.assertNotIn("n_undetected", default_payload)
            self.assertNotIn("undetected_sources", default_payload)
            self.assertIn("n_undetected", undetected_payload)
            self.assertIn("undetected_sources", undetected_payload)

            restored = load_result(undetected_path)
            self.assertEqual(restored.n_undetected, len(restored.undetected_sources))
            self.assertEqual(
                restored.n_undetected,
                undetected_result.n_undetected,
            )

    def test_reproducibility_and_parity(self):
        # 1. Run the pipeline twice with the identical seed
        live_result_1 = self._run_lightweight_pipeline(seed=42)
        live_result_2 = self._run_lightweight_pipeline(seed=42)

        # Guarantee the live Rust backend is fully deterministic first
        self.assertEqual(live_result_1.n_simulated, live_result_2.n_simulated)
        self.assertEqual(live_result_1.n_detected, live_result_2.n_detected)
        self.assertEqual(str(live_result_1), str(live_result_2))

        with tempfile.TemporaryDirectory() as td:
            out1 = pathlib.Path(td) / "sim_result_1.json"
            out2 = pathlib.Path(td) / "sim_result_2.json"
            
            # Save both live results and load them back
            save_result(live_result_1, out1)
            save_result(live_result_2, out2)
            
            self.assertTrue(out1.exists())
            self.assertTrue(out2.exists())
            
            restored_1 = load_result(out1)
            restored_2 = load_result(out2)

            # 2. Check Base Attributes & String Representations (Live vs Restored vs Restored)
            self.assertEqual(live_result_1.n_detected, restored_1.n_detected)
            self.assertEqual(restored_1.n_detected, restored_2.n_detected)
            
            self.assertEqual(str(live_result_1), str(restored_1))
            self.assertEqual(str(restored_1), str(restored_2))
            
            self.assertEqual(repr(live_result_1), repr(restored_1))
            self.assertEqual(repr(restored_1), repr(restored_2))

            # 3. Check Rate Summaries and Limits
            if len(live_result_1.rate_summaries) > 0:
                live_rs = live_result_1.rate_summaries[0]
                rest_rs_1 = restored_1.rate_summaries[0]
                rest_rs_2 = restored_2.rate_summaries[0]
                
                self.assertEqual(repr(live_rs), repr(rest_rs_1))
                self.assertEqual(repr(rest_rs_1), repr(rest_rs_2))
                self.assertEqual(repr(live_rs.upper_limit(0.95)), repr(rest_rs_1.upper_limit(0.95)))

            # 4. Deep check of Sources and Photometry Iterators
            live_sources = live_result_1.sources()
            rest_sources_1 = restored_1.sources()
            rest_sources_2 = restored_2.sources()
            
            self.assertEqual(len(live_sources), len(rest_sources_1))
            self.assertEqual(len(rest_sources_1), len(rest_sources_2))

            if len(live_sources) > 0:
                live_src = live_sources[0]
                rest_src_1 = rest_sources_1[0]
                rest_src_2 = rest_sources_2[0]
                
                self.assertEqual(repr(live_src), repr(rest_src_1))
                self.assertEqual(repr(rest_src_1), repr(rest_src_2))

                # Ensure photometry executes as a method and arrays match identically
                r_times, r_mags, r_errs, r_bands = live_src.photometry()
                m1_times, m1_mags, m1_errs, m1_bands = rest_src_1.photometry()
                m2_times, m2_mags, m2_errs, m2_bands = rest_src_2.photometry()
                
                # Check live against restored 1
                np.testing.assert_allclose(r_times, m1_times)
                np.testing.assert_allclose(r_mags, m1_mags)
                np.testing.assert_allclose(r_errs, m1_errs)
                self.assertEqual(list(r_bands), list(m1_bands))

                # Check restored 1 against restored 2
                np.testing.assert_allclose(m1_times, m2_times)
                np.testing.assert_allclose(m1_mags, m2_mags)


if __name__ == "__main__":
    unittest.main()