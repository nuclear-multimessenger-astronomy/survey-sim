use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
	PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn python_cmd() -> Option<&'static str> {
	if Command::new("python3").arg("--version").output().is_ok() {
		Some("python3")
	} else if Command::new("python").arg("--version").output().is_ok() {
		Some("python")
	} else {
		None
	}
}

#[test]
fn serialization_py_exists() {
	let path = repo_root().join("python").join("survey_sim").join("serialization.py");
	assert!(
		path.exists(),
		"Expected serialization.py at {}",
		path.display()
	);
}

#[test]
fn serialization_py_import_and_roundtrip() {
	let Some(python) = python_cmd() else {
		// No python runtime available in this environment.
		return;
	};

	let script = r#"
import importlib.util
import math
import pathlib
import sys
import tempfile
import types

root = pathlib.Path(r'__ROOT__')
path = root / 'python' / 'survey_sim' / 'serialization.py'
if not path.exists():
	raise SystemExit(f'missing: {path}')

survey_sim_stub = types.ModuleType('survey_sim')

class RateSummary:
	def __init__(self):
		self.transient_type = 'KN'
		self.volumetric_rate = float('inf')
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
		self.transient_type = 'KN'
		self.true_params = {'x': float('nan')}
		self.photometry = ([1.0, 2.0], [20.1, 20.2], [0.1, 0.2], ['g', 'r'])
		self.non_detections = ([0.5], [21.0], ['g'])

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
		self.strategy_name = 'baseline'
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

survey_sim_stub.SimulationResult = SimulationResult
survey_sim_stub.TooSimulationResult = TooSimulationResult
survey_sim_stub.CoverageResult = CoverageResult
survey_sim_stub.CoverageResult3D = CoverageResult3D
sys.modules['survey_sim'] = survey_sim_stub

spec = importlib.util.spec_from_file_location('serialization', str(path))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def roundtrip(obj):
	with tempfile.TemporaryDirectory() as td:
		outfile = pathlib.Path(td) / 'result.json'
		mod.save_result(obj, outfile)
		assert outfile.exists(), f'missing output file: {outfile}'
		return mod.load_result(outfile)

sim_out = roundtrip(SimulationResult())
assert sim_out.n_simulated == 20
assert sim_out.n_detected == 10
assert len(sim_out.rate_summaries) == 1
assert math.isinf(sim_out.rate_summaries[0].volumetric_rate)
src = sim_out.sources()[0]
assert src.transient_type == 'KN'
assert math.isnan(src.true_params['x'])

too_out = roundtrip(TooSimulationResult())
assert too_out.strategy_name == 'baseline'
assert too_out.n_events == 3
assert too_out.detected == [True, False, True]

cov2d_out = roundtrip(CoverageResult())
assert cov2d_out.prob_2d == 0.5
assert cov2d_out.n_pixels == 7

cov3d_out = roundtrip(CoverageResult3D())
assert cov3d_out.prob_2d == 0.4
assert cov3d_out.prob_3d == 0.2
assert cov3d_out.best_d_max == [150.0, 180.0]
"#
	.replace("__ROOT__", &repo_root().display().to_string());

	let out = Command::new(python).arg("-c").arg(script).output().unwrap();
	assert!(
		out.status.success(),
		"Python test failed:\nstdout:\n{}\nstderr:\n{}",
		String::from_utf8_lossy(&out.stdout),
		String::from_utf8_lossy(&out.stderr)
	);
}
