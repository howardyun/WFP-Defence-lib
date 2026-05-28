from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Callable

import torch
from torch import nn


def _load_module_from_file(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import Python module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_symbol(spec: str) -> Callable:
    """Load `module:function` or `C:/path/file.py:function`."""
    if ":" not in spec:
        raise ValueError("Expected symbol spec in the form 'module:function' or 'file.py:function'")

    module_name, symbol_name = spec.rsplit(":", 1)
    module_path = Path(module_name)
    if module_path.suffix == ".py" or module_path.exists():
        module = _load_module_from_file(module_path)
    else:
        module = importlib.import_module(module_name)

    try:
        symbol = getattr(module, symbol_name)
    except AttributeError as exc:
        raise AttributeError(f"{module_name!r} does not define {symbol_name!r}") from exc
    if not callable(symbol):
        raise TypeError(f"{spec!r} is not callable")
    return symbol


def extract_state_dict(checkpoint, key: str | None = None):
    if key:
        return checkpoint[key]
    if isinstance(checkpoint, dict):
        for candidate in ("state_dict", "model_state_dict", "net", "model"):
            value = checkpoint.get(candidate)
            if isinstance(value, dict):
                return value
        if all(isinstance(k, str) for k in checkpoint.keys()):
            return checkpoint
    return checkpoint


def build_model(builder: Callable, num_classes: int) -> nn.Module:
    try:
        model = builder(num_classes=num_classes)
    except TypeError:
        try:
            model = builder(num_classes)
        except TypeError:
            model = builder()
    if not isinstance(model, nn.Module):
        raise TypeError("Detector builder must return a torch.nn.Module")
    return model


def load_state_dict_flex(model: nn.Module, state_dict) -> None:
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError:
        pass

    if isinstance(state_dict, dict) and all(isinstance(k, str) for k in state_dict.keys()):
        stripped = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }
        model.load_state_dict(stripped)
        return

    model.load_state_dict(state_dict)


def load_detector(
    *,
    builder_spec: str,
    checkpoint_path: str | Path,
    num_classes: int,
    device: torch.device,
    state_dict_key: str | None = None,
) -> nn.Module:
    builder = load_symbol(builder_spec)
    model = build_model(builder, num_classes)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_state_dict(checkpoint, state_dict_key)
    load_state_dict_flex(model, state_dict)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def format_detector_input(x: torch.Tensor, kind: str = "burst", layout: str = "ncl") -> torch.Tensor:
    """Format a 2-D batch `[N, L]` for common PyTorch detector layouts."""
    if kind not in {"burst", "direction"}:
        raise ValueError(f"Unknown detector input kind: {kind}")

    if layout == "nl":
        return x
    if layout == "ncl":
        return x.unsqueeze(1)
    if layout == "nchw":
        return x.unsqueeze(1).unsqueeze(-1)
    raise ValueError(f"Unknown detector input layout: {layout}")
