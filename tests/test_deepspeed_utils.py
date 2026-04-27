import argparse
import sys
from importlib.machinery import ModuleSpec
from types import ModuleType, SimpleNamespace

import torch


def install_deepspeed_utils_import_stubs():
    accelerate_module = sys.modules.get("accelerate")
    if accelerate_module is None:
        accelerate_module = ModuleType("accelerate")
        accelerate_module.__spec__ = ModuleSpec("accelerate", loader=None)
        sys.modules["accelerate"] = accelerate_module
    accelerate_module.DeepSpeedPlugin = getattr(accelerate_module, "DeepSpeedPlugin", type("DeepSpeedPlugin", (), {}))
    accelerate_module.Accelerator = getattr(accelerate_module, "Accelerator", type("Accelerator", (), {}))

    utils_module = ModuleType("library.utils")
    utils_module.__spec__ = ModuleSpec("library.utils", loader=None)
    utils_module.setup_logging = lambda *args, **kwargs: None
    sys.modules.setdefault("library.utils", utils_module)

    device_utils_module = sys.modules.get("library.device_utils")
    if device_utils_module is None:
        device_utils_module = ModuleType("library.device_utils")
        device_utils_module.__spec__ = ModuleSpec("library.device_utils", loader=None)
        sys.modules["library.device_utils"] = device_utils_module
    device_utils_module.get_preferred_device = lambda: SimpleNamespace(type="cuda")


install_deepspeed_utils_import_stubs()

from library import deepspeed_utils


def test_prepare_deepspeed_args_enables_zero3_full_model_save_by_default():
    args = argparse.Namespace(
        deepspeed=True,
        zero_stage=3,
        zero3_save_16bit_model=False,
        optimizer_type="AdamW",
        use_8bit_adam=False,
        mixed_precision="bf16",
        max_data_loader_n_workers=8,
    )

    deepspeed_utils.prepare_deepspeed_args(args)

    assert args.zero3_save_16bit_model is True
    assert args.max_data_loader_n_workers == 1


def test_prepare_deepspeed_args_keeps_zero3_save_flag_when_already_enabled():
    args = argparse.Namespace(
        deepspeed=True,
        zero_stage=3,
        zero3_save_16bit_model=True,
        optimizer_type="AdamW",
        use_8bit_adam=False,
        mixed_precision="bf16",
        max_data_loader_n_workers=2,
    )

    deepspeed_utils.prepare_deepspeed_args(args)

    assert args.zero3_save_16bit_model is True


def test_prepare_deepspeed_model_uses_requested_autocast_dtype(monkeypatch):
    calls = []

    class AutocastSpy:
        def __init__(self, device_type, dtype=None):
            calls.append((device_type, dtype))

        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyModel(torch.nn.Module):
        @property
        def device(self):
            return SimpleNamespace(type="cuda")

        def forward(self, x):
            return x

    monkeypatch.setattr(deepspeed_utils.torch, "autocast", AutocastSpy)

    wrapper = deepspeed_utils.prepare_deepspeed_model(
        argparse.Namespace(mixed_precision="bf16"),
        model=DummyModel(),
    )
    wrapper.get_models()["model"](torch.tensor(1.0))

    assert calls == [("cuda", torch.bfloat16)]
