import pathlib
import tempfile
import unittest
import json

def _assert_allclose(a, b, rtol=1e-5, atol=1e-8):
    for x, y in zip(a, b):
        assert abs(x - y) <= atol + rtol * abs(y), f"{x} != {y}"

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
        try:
            cls.survey = load_ztf_survey(nside=64)
        except Exception as e:
            raise unittest.SkipTest(f"ZTF survey data or dependencies unavailable: {e}")

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
            save_result(live_result_1, out1, truncate=False)
            save_result(live_result_2, out2, truncate=False)
            
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
                
                # Check live against restored 1 (within truncation tolerance)
                _assert_allclose(r_times, m1_times, atol=1e-3)
                _assert_allclose(r_mags, m1_mags, atol=1e-3)
                _assert_allclose(r_errs, m1_errs, atol=1e-3)
                self.assertEqual(list(r_bands), list(m1_bands))

                # Check restored 1 against restored 2 (exact match)
                _assert_allclose(m1_times, m2_times)
                _assert_allclose(m1_mags, m2_mags)
                _assert_allclose(m1_errs, m2_errs)

class TestImposterAnalysisSerialization(unittest.TestCase):
    def test_imposter_analysis_serialization_roundtrip(self):
        from survey_sim.serialization import _MockRateSummary, _MockSimulationResult

        rs_kn = _MockRateSummary({
            "transient_type": "Kilonova",
            "volumetric_rate": 100.0,
            "detections_per_year": 10.0,
            "detections_total": 30.0,
            "overall_efficiency": 0.20,
            "survey_omega_sr": 1.5,
            "survey_duration_years": 3.0,
            "effective_vt_gpc3_yr": 0.30,
            "n_simulated": 1000,
            "n_detected": 200,
            "z_max": 0.25,
        })
        rs_fbot = _MockRateSummary({
            "transient_type": "FBOT",
            "volumetric_rate": 500.0,
            "detections_per_year": 20.0,
            "detections_total": 60.0,
            "overall_efficiency": 0.05,
            "survey_omega_sr": 1.5,
            "survey_duration_years": 3.0,
            "effective_vt_gpc3_yr": 0.12,
            "n_simulated": 1000,
            "n_detected": 50,
            "z_max": 0.40,
        })

        sim = _MockSimulationResult(2000, 250, [rs_kn, rs_fbot], [])
        analysis = sim.analyze_imposters(signal_type="Kilonova", default_sys_err=0.25, followup_budget=50.0)

        with tempfile.TemporaryDirectory() as td:
            out_an = pathlib.Path(td) / "imposter_analysis.json"
            save_result(analysis, out_an)
            restored_an = load_result(out_an)

            self.assertEqual(restored_an.signal_name, analysis.signal_name)
            self.assertAlmostEqual(restored_an.purity, analysis.purity, places=4)
            self.assertAlmostEqual(restored_an.f1_score, analysis.f1_score, places=4)
            self.assertAlmostEqual(restored_an.significance_fom, analysis.significance_fom, places=4)
            self.assertAlmostEqual(restored_an.pure_efficiency, analysis.pure_efficiency, places=4)
            self.assertAlmostEqual(restored_an.f_beta(0.5), analysis.f_beta(0.5), places=4)
            self.assertEqual(restored_an.imposter_breakdown, [("FBOT", 60.0)])

            # Test simulation result with imposter analysis attached
            out_sim = pathlib.Path(td) / "sim_with_imposters.json"
            save_result(sim, out_sim)
            restored_sim = load_result(out_sim)

            self.assertTrue(hasattr(restored_sim, "imposter_analysis"))
            self.assertAlmostEqual(restored_sim.imposter_analysis.purity, analysis.purity, places=4)
            self.assertAlmostEqual(restored_sim.imposter_analysis.naive_recovered_rate, analysis.naive_recovered_rate, places=4)

            # Test analyze_imposters method on restored simulation
            recalculated = restored_sim.analyze_imposters(signal_type="Kilonova", default_sys_err=0.25)
            self.assertAlmostEqual(recalculated.purity, analysis.purity, places=4)


if __name__ == "__main__":
    unittest.main()