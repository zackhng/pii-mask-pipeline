"""Load a class from a 'package.module:ClassName' string (the NER swap point)."""
from __future__ import annotations

import importlib


def load_class(path: str):
    if ":" not in path:
        raise ValueError(f"class path must look like 'package.module:ClassName', got {path!r}")
    module_name, class_name = path.split(":", 1)
    return getattr(importlib.import_module(module_name), class_name)
