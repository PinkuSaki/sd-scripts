import argparse
import sys
from importlib.machinery import ModuleSpec
from types import ModuleType
from unittest.mock import MagicMock

import pytest


def install_anima_train_network_import_stubs():
    if "accelerate" not in sys.modules:
        accelerate_module = ModuleType("accelerate")
        accelerate_module.__spec__ = ModuleSpec("accelerate", loader=None)
        accelerate_module.Accelerator = type("Accelerator", (), {})
        sys.modules["accelerate"] = accelerate_module

    device_utils_module = ModuleType("library.device_utils")
    device_utils_module.__spec__ = ModuleSpec("library.device_utils", loader=None)
    device_utils_module.init_ipex = lambda: None
    device_utils_module.clean_memory_on_device = lambda *args, **kwargs: None
    sys.modules.setdefault("library.device_utils", device_utils_module)

    utils_module = ModuleType("library.utils")
    utils_module.__spec__ = ModuleSpec("library.utils", loader=None)
    utils_module.setup_logging = lambda *args, **kwargs: None
    sys.modules.setdefault("library.utils", utils_module)

    train_network_module = ModuleType("train_network")
    train_network_module.__spec__ = ModuleSpec("train_network", loader=None)

    class NetworkTrainer:
        def __init__(self):
            pass

    train_network_module.NetworkTrainer = NetworkTrainer
    sys.modules.setdefault("train_network", train_network_module)

    train_util_module = ModuleType("library.train_util")
    train_util_module.__spec__ = ModuleSpec("library.train_util", loader=None)
    train_util_module.DatasetGroup = type("DatasetGroup", (), {})
    train_util_module.MinimalDataset = type("MinimalDataset", (), {})
    sys.modules.setdefault("library.train_util", train_util_module)

    plain_modules = (
        "library.anima_models",
        "library.anima_train_utils",
        "library.anima_utils",
        "library.flux_train_utils",
        "library.qwen_image_autoencoder_kl",
        "library.sd3_train_utils",
        "library.strategy_base",
    )
    for module_name in plain_modules:
        module = ModuleType(module_name)
        module.__spec__ = ModuleSpec(module_name, loader=None)
        sys.modules.setdefault(module_name, module)

    strategy_anima_module = ModuleType("library.strategy_anima")
    strategy_anima_module.__spec__ = ModuleSpec("library.strategy_anima", loader=None)
    strategy_anima_module.AnimaTokenizeStrategy = type("AnimaTokenizeStrategy", (), {})
    strategy_anima_module.AnimaLatentsCachingStrategy = type("AnimaLatentsCachingStrategy", (), {})
    strategy_anima_module.AnimaTextEncodingStrategy = type("AnimaTextEncodingStrategy", (), {})
    strategy_anima_module.AnimaTextEncoderOutputsCachingStrategy = type("AnimaTextEncoderOutputsCachingStrategy", (), {})
    sys.modules.setdefault("library.strategy_anima", strategy_anima_module)


install_anima_train_network_import_stubs()

import anima_train_network


@pytest.fixture
def trainer():
    return anima_train_network.AnimaNetworkTrainer()


@pytest.fixture
def mock_args():
    return argparse.Namespace(
        fp8_base=False,
        fp8_base_unet=False,
        cache_text_encoder_outputs_to_disk=False,
        cache_text_encoder_outputs=True,
        network_train_unet_only=True,
        blocks_to_swap=None,
        cpu_offload_checkpointing=False,
        unsloth_offload_checkpointing=False,
        gradient_checkpointing=False,
    )


def test_assert_extra_args_rejects_markdown_section_dropout_with_text_encoder_cache(trainer, mock_args):
    train_dataset_group = MagicMock()
    train_dataset_group.enable_anima_markdown_section_dropout.return_value = True
    train_dataset_group.is_text_encoder_output_cacheable.return_value = False
    train_dataset_group.verify_bucket_reso_steps = MagicMock()

    val_dataset_group = MagicMock()
    val_dataset_group.enable_anima_markdown_section_dropout.return_value = False
    val_dataset_group.verify_bucket_reso_steps = MagicMock()

    with pytest.raises(AssertionError, match="markdown_section_dropout"):
        trainer.assert_extra_args(mock_args, train_dataset_group, val_dataset_group)

    train_dataset_group.enable_anima_markdown_section_dropout.assert_called_once()
    val_dataset_group.enable_anima_markdown_section_dropout.assert_called_once()


def test_assert_extra_args_allows_text_encoder_cache_without_markdown_section_dropout(trainer, mock_args):
    train_dataset_group = MagicMock()
    train_dataset_group.enable_anima_markdown_section_dropout.return_value = False
    train_dataset_group.is_text_encoder_output_cacheable.return_value = True
    train_dataset_group.verify_bucket_reso_steps = MagicMock()

    val_dataset_group = MagicMock()
    val_dataset_group.enable_anima_markdown_section_dropout.return_value = False
    val_dataset_group.verify_bucket_reso_steps = MagicMock()

    trainer.assert_extra_args(mock_args, train_dataset_group, val_dataset_group)

    train_dataset_group.enable_anima_markdown_section_dropout.assert_called_once()
    val_dataset_group.enable_anima_markdown_section_dropout.assert_called_once()
    train_dataset_group.verify_bucket_reso_steps.assert_called()
    val_dataset_group.verify_bucket_reso_steps.assert_called()
