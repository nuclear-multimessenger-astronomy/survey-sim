"""Auto-configure CUDA libraries for JAX GPU support.

Import this module before importing JAX to ensure pip-installed nvidia
libraries (cuDNN, cuBLAS, cuSPARSE, etc.) are discoverable via ctypes preload.

Usage::

    import survey_sim.gpu_setup  # noqa: F401  (side-effect import)
    import jax
"""

import ctypes
import os
import sys


def _find_nvidia_libs():
    """Find nvidia shared libraries from pip packages in the active venv."""
    libs = {}
    for path in sys.path:
        nvidia_dir = os.path.join(path, "nvidia")
        if not os.path.isdir(nvidia_dir):
            continue
        for pkg in os.listdir(nvidia_dir):
            lib_dir = os.path.join(nvidia_dir, pkg, "lib")
            if not os.path.isdir(lib_dir):
                continue
            for f in os.listdir(lib_dir):
                if f.endswith(".so") or ".so." in f:
                    libs[f] = os.path.join(lib_dir, f)
        break  # only need the first site-packages hit
    return libs


def setup():
    """Preload nvidia CUDA libraries so JAX can find them via dlopen."""
    libs = _find_nvidia_libs()
    if not libs:
        return

    # Also set LD_LIBRARY_PATH for any child processes
    lib_dirs = set()
    for path in libs.values():
        lib_dirs.add(os.path.dirname(path))
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    existing = set(ld_path.split(":")) if ld_path else set()
    new_dirs = [d for d in lib_dirs if d not in existing]
    if new_dirs:
        os.environ["LD_LIBRARY_PATH"] = ":".join(new_dirs) + (":" + ld_path if ld_path else "")

    # Preload key libraries in dependency order using ctypes
    # This makes them available for subsequent dlopen calls by JAX
    load_order = [
        "libnvJitLink.so",
        "libcublas.so",
        "libcublasLt.so",
        "libcusparse.so",
        "libcusolver.so",
        "libcufft.so",
        "libcudnn.so",
    ]
    for prefix in load_order:
        for name, path in libs.items():
            if name.startswith(prefix.replace(".so", "")) and ".so" in name:
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
                break


setup()
