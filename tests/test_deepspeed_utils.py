import argparse
import sys
from importlib.machinery import ModuleSpec
from types import ModuleType, SimpleNamespace


def install_deepspeed_utils_import_stubs():
    if "accelerate" not in sys.modules:
        accelerate_module = ModuleType("accelerate")
        accelerate_module.__spec__ = ModuleSpec("accelerate", loader=None)
        accelerate_module.DeepSpeedPlugin = type("DeepSpeedPlugin", (), {})
        accelerate_module.Accelerator = type("Accelerator", (), {})
        sys.modules["accelerate"] = accelerate_module

    utils_module = ModuleType("library.utils")
    utils_module.__spec__ = ModuleSpec("library.utils", loader=None)
    utils_module.setup_logging = lambda *args, **kwargs: None
    sys.modules.setdefault("library.utils", utils_module)

    device_utils_module = ModuleType("library.device_utils")
    device_utils_module.__spec__ = ModuleSpec("library.device_utils", loader=None)
    device_utils_module.get_preferred_device = lambda: SimpleNamespace(type="cuda")
    sys.modules.setdefault("library.device_utils", device_utils_module)


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
