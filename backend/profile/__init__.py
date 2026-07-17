"""Compatibility shim for the stdlib ``profile`` module.

This package name collides with Python's standard-library ``profile`` module
when scripts are executed from ``backend/``. Some PyTorch/TensorRT export paths
import ``cProfile``, which in turn expects ``profile`` to expose the stdlib
API. Re-export that API here so local benchmark packages do not break those
imports.
"""

from __future__ import annotations

import importlib.util
import os
import sysconfig


_STDLIB_PROFILE_PATH = os.path.join(sysconfig.get_path("stdlib"), "profile.py")
_STDLIB_SPEC = importlib.util.spec_from_file_location(
    "_stdlib_profile",
    _STDLIB_PROFILE_PATH,
)

if _STDLIB_SPEC is None or _STDLIB_SPEC.loader is None:
    raise ImportError(f"Failed to locate stdlib profile module at {_STDLIB_PROFILE_PATH}")

_stdlib_profile = importlib.util.module_from_spec(_STDLIB_SPEC)
_STDLIB_SPEC.loader.exec_module(_stdlib_profile)

Profile = _stdlib_profile.Profile
run = _stdlib_profile.run
runctx = _stdlib_profile.runctx

if hasattr(_stdlib_profile, "main"):
    main = _stdlib_profile.main

__all__ = ["Profile", "run", "runctx"]
if "main" in globals():
    __all__.append("main")
