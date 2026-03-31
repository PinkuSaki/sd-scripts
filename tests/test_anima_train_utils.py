import argparse
import sys
from importlib.machinery import ModuleSpec
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest


def install_anima_train_utils_import_stubs():
    if "accelerate" not in sys.modules:
        accelerate_module = ModuleType("accelerate")
        accelerate_module.__spec__ = ModuleSpec("accelerate", loader=None)
        accelerate_module.Accelerator = type("Accelerator", (), {})
        accelerate_module.PartialState = type("PartialState", (), {})
        sys.modules["accelerate"] = accelerate_module

    device_utils_module = ModuleType("library.device_utils")
    device_utils_module.__spec__ = ModuleSpec("library.device_utils", loader=None)
    device_utils_module.init_ipex = lambda: None
    device_utils_module.clean_memory_on_device = lambda *args, **kwargs: None
    device_utils_module.synchronize_device = lambda *args, **kwargs: None
    sys.modules.setdefault("library.device_utils", device_utils_module)

    utils_module = ModuleType("library.utils")
    utils_module.__spec__ = ModuleSpec("library.utils", loader=None)
    utils_module.setup_logging = lambda *args, **kwargs: None
    sys.modules.setdefault("library.utils", utils_module)

    anima_models_module = ModuleType("library.anima_models")
    anima_models_module.__spec__ = ModuleSpec("library.anima_models", loader=None)
    anima_models_module.Anima = type("Anima", (), {})
    sys.modules.setdefault("library.anima_models", anima_models_module)

    qwen_module = ModuleType("library.qwen_image_autoencoder_kl")
    qwen_module.__spec__ = ModuleSpec("library.qwen_image_autoencoder_kl", loader=None)
    qwen_module.AutoencoderKLQwenImage = type("AutoencoderKLQwenImage", (), {})
    sys.modules.setdefault("library.qwen_image_autoencoder_kl", qwen_module)

    for module_name in ("library.anima_utils", "library.train_util"):
        module = ModuleType(module_name)
        module.__spec__ = ModuleSpec(module_name, loader=None)
        sys.modules.setdefault(module_name, module)


install_anima_train_utils_import_stubs()

from library import anima_train_utils


def test_use_zero3_sampling_all_ranks_only_for_multigpu_deepspeed_zero3():
    multi_gpu_state = SimpleNamespace(num_processes=4)
    single_gpu_state = SimpleNamespace(num_processes=1)

    assert anima_train_utils._use_zero3_sampling_all_ranks(
        argparse.Namespace(deepspeed=True, zero_stage=3), multi_gpu_state
    )
    assert not anima_train_utils._use_zero3_sampling_all_ranks(
        argparse.Namespace(deepspeed=True, zero_stage=2), multi_gpu_state
    )
    assert not anima_train_utils._use_zero3_sampling_all_ranks(
        argparse.Namespace(deepspeed=False, zero_stage=3), multi_gpu_state
    )
    assert not anima_train_utils._use_zero3_sampling_all_ranks(
        argparse.Namespace(deepspeed=True, zero_stage=3), single_gpu_state
    )


def test_use_all_ranks_for_anima_weight_save_only_for_multigpu_deepspeed_zero3():
    multi_gpu_accelerator = SimpleNamespace(num_processes=4)
    single_gpu_accelerator = SimpleNamespace(num_processes=1)

    assert anima_train_utils.use_all_ranks_for_anima_weight_save(
        argparse.Namespace(deepspeed=True, zero_stage=3), multi_gpu_accelerator
    )
    assert not anima_train_utils.use_all_ranks_for_anima_weight_save(
        argparse.Namespace(deepspeed=True, zero_stage=2), multi_gpu_accelerator
    )
    assert not anima_train_utils.use_all_ranks_for_anima_weight_save(
        argparse.Namespace(deepspeed=False, zero_stage=3), multi_gpu_accelerator
    )
    assert not anima_train_utils.use_all_ranks_for_anima_weight_save(
        argparse.Namespace(deepspeed=True, zero_stage=3), single_gpu_accelerator
    )


def test_get_anima_state_dict_for_save_uses_accelerator_api():
    accelerator = MagicMock()
    accelerator.get_state_dict.return_value = {"net.weight": object()}
    args = argparse.Namespace(deepspeed=False, zero_stage=0, zero3_save_16bit_model=False)
    dit = object()

    state_dict = anima_train_utils._get_anima_state_dict_for_save(accelerator, args, dit)

    assert state_dict == {"net.weight": accelerator.get_state_dict.return_value["net.weight"]}
    accelerator.get_state_dict.assert_called_once_with(dit)


def test_get_anima_state_dict_for_save_raises_clear_error_for_zero3_without_16bit_gather():
    accelerator = MagicMock()
    accelerator.get_state_dict.side_effect = ValueError("Cannot get 16bit model weights")
    args = argparse.Namespace(deepspeed=True, zero_stage=3, zero3_save_16bit_model=False)

    with pytest.raises(ValueError, match="zero3_save_16bit_model"):
        anima_train_utils._get_anima_state_dict_for_save(accelerator, args, object())


def test_save_anima_model_stepwise_non_main_zero3_only_runs_state_save_once(monkeypatch):
    common_saver = MagicMock()
    state_saver = MagicMock()

    monkeypatch.setattr(anima_train_utils.train_util, "save_sd_model_on_epoch_end_or_stepwise_common", common_saver, raising=False)
    monkeypatch.setattr(anima_train_utils.train_util, "save_and_remove_state_stepwise", state_saver, raising=False)

    accelerator = SimpleNamespace(
        num_processes=4,
        is_main_process=False,
        get_state_dict=MagicMock(return_value={"net.weight": object()}),
        wait_for_everyone=MagicMock(),
    )
    args = argparse.Namespace(deepspeed=True, zero_stage=3, save_state=True)

    anima_train_utils.save_anima_model_on_epoch_end_or_stepwise(
        args,
        False,
        accelerator,
        save_dtype=None,
        epoch=0,
        num_train_epochs=8,
        global_step=250,
        dit=object(),
    )

    common_saver.assert_not_called()
    state_saver.assert_called_once_with(args, accelerator, 250)
    accelerator.get_state_dict.assert_called_once()
    accelerator.wait_for_everyone.assert_called_once()


def test_save_anima_model_stepwise_main_zero3_delegates_state_save_to_common_helper(monkeypatch):
    common_saver = MagicMock()
    state_saver = MagicMock()

    monkeypatch.setattr(anima_train_utils.train_util, "save_sd_model_on_epoch_end_or_stepwise_common", common_saver, raising=False)
    monkeypatch.setattr(anima_train_utils.train_util, "save_and_remove_state_stepwise", state_saver, raising=False)

    accelerator = SimpleNamespace(
        num_processes=4,
        is_main_process=True,
        get_state_dict=MagicMock(return_value={"net.weight": object()}),
        wait_for_everyone=MagicMock(),
    )
    args = argparse.Namespace(deepspeed=True, zero_stage=3, save_state=True)

    anima_train_utils.save_anima_model_on_epoch_end_or_stepwise(
        args,
        False,
        accelerator,
        save_dtype=None,
        epoch=0,
        num_train_epochs=8,
        global_step=250,
        dit=object(),
    )

    common_saver.assert_called_once()
    state_saver.assert_not_called()
    accelerator.get_state_dict.assert_called_once()
    accelerator.wait_for_everyone.assert_called_once()
