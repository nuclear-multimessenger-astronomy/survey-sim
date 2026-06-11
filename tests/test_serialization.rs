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
	let path = repo_root().join("serialization.py");
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
import pathlib
import sys

root = pathlib.Path(r'__ROOT__')
path = root / 'serialization.py'
if not path.exists():
	raise SystemExit(f'missing: {path}')

spec = importlib.util.spec_from_file_location('serialization', str(path))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

sample = {'name': 'survey-sim', 'value': 42, 'items': [1, 2, 3]}

if hasattr(mod, 'serialize') and hasattr(mod, 'deserialize'):
	encoded = mod.serialize(sample)
	decoded = mod.deserialize(encoded)
	assert decoded == sample, f'roundtrip mismatch: {decoded} != {sample}'
elif hasattr(mod, 'dumps') and hasattr(mod, 'loads'):
	encoded = mod.dumps(sample)
	decoded = mod.loads(encoded)
	assert decoded == sample, f'roundtrip mismatch: {decoded} != {sample}'
elif hasattr(mod, 'to_json') and hasattr(mod, 'from_json'):
	encoded = mod.to_json(sample)
	decoded = mod.from_json(encoded)
	assert decoded == sample, f'roundtrip mismatch: {decoded} != {sample}'
else:
	# Fallback smoke test: module loaded and has at least one public symbol.
	public = [n for n in dir(mod) if not n.startswith('_')]
	assert public, 'serialization.py has no public symbols to test'
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
