"""File-based model loading also supports hyphens in GitHub usernames."""

import importlib.util
import inspect
import sys


def load_adapter(root, model_key):
    path = root / "models" / f"{model_key}.py"
    module_name = "submission_" + model_key.replace("-", "_")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    examples = {
        "build_model": (({},), {}),
        "load_weights": ((None, "weights.pt"), {"device": "cpu"}),
        "predict": ((None, None), {}),
    }
    for name, (args, kwargs) in examples.items():
        function = getattr(module, name, None)
        if not callable(function):
            raise ValueError(f"{path.name} must provide {name}")
        try:
            inspect.signature(function).bind(*args, **kwargs)
        except TypeError as exc:
            raise ValueError(f"Invalid {name} signature in {path.name}") from exc
    return module

