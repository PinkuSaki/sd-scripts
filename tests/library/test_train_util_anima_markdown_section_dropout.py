import pickle
import sys
from importlib.machinery import ModuleSpec
from types import ModuleType
from unittest.mock import patch

import pytest


def install_train_util_import_stubs():
    if "accelerate" not in sys.modules:
        accelerate_module = ModuleType("accelerate")
        accelerate_module.__spec__ = ModuleSpec("accelerate", loader=None)
        accelerate_module.__version__ = "1.0.0"

        class DummyAccelerator:
            pass

        accelerate_module.Accelerator = DummyAccelerator
        accelerate_module.InitProcessGroupKwargs = type("InitProcessGroupKwargs", (), {})
        accelerate_module.DistributedDataParallelKwargs = type("DistributedDataParallelKwargs", (), {})
        accelerate_module.PartialState = type("PartialState", (), {})
        sys.modules["accelerate"] = accelerate_module

    if "diffusers.optimization" not in sys.modules:
        optimization_module = ModuleType("diffusers.optimization")
        optimization_module.__spec__ = ModuleSpec("diffusers.optimization", loader=None)
        optimization_module.SchedulerType = type("SchedulerType", (), {})
        optimization_module.TYPE_TO_SCHEDULER_FUNCTION = {}
        sys.modules["diffusers.optimization"] = optimization_module

    if "diffusers" not in sys.modules:
        diffusers_module = ModuleType("diffusers")
        diffusers_module.__spec__ = ModuleSpec("diffusers", loader=None)
        for name in (
            "StableDiffusionPipeline",
            "DDPMScheduler",
            "EulerAncestralDiscreteScheduler",
            "DPMSolverMultistepScheduler",
            "DPMSolverSinglestepScheduler",
            "LMSDiscreteScheduler",
            "PNDMScheduler",
            "DDIMScheduler",
            "EulerDiscreteScheduler",
            "HeunDiscreteScheduler",
            "KDPM2DiscreteScheduler",
            "KDPM2AncestralDiscreteScheduler",
            "AutoencoderKL",
        ):
            setattr(diffusers_module, name, type(name, (), {}))
        sys.modules["diffusers"] = diffusers_module

    if "torchvision.transforms" not in sys.modules:
        transforms_module = ModuleType("torchvision.transforms")
        transforms_module.__spec__ = ModuleSpec("torchvision.transforms", loader=None)

        class Compose:
            def __init__(self, transforms):
                self.transforms = transforms

            def __call__(self, value):
                for transform in self.transforms:
                    value = transform(value)
                return value

        class ToTensor:
            def __call__(self, value):
                return value

        class Normalize:
            def __init__(self, *args, **kwargs):
                pass

            def __call__(self, value):
                return value

        transforms_module.Compose = Compose
        transforms_module.ToTensor = ToTensor
        transforms_module.Normalize = Normalize
        sys.modules["torchvision.transforms"] = transforms_module

        torchvision_module = ModuleType("torchvision")
        torchvision_module.__spec__ = ModuleSpec("torchvision", loader=None)
        torchvision_module.transforms = transforms_module
        sys.modules["torchvision"] = torchvision_module

    if "transformers.optimization" not in sys.modules:
        transformers_optimization_module = ModuleType("transformers.optimization")
        transformers_optimization_module.__spec__ = ModuleSpec("transformers.optimization", loader=None)
        transformers_optimization_module.SchedulerType = type("SchedulerType", (), {})
        transformers_optimization_module.TYPE_TO_SCHEDULER_FUNCTION = {}
        sys.modules["transformers.optimization"] = transformers_optimization_module

    if "transformers" not in sys.modules:
        transformers_module = ModuleType("transformers")
        transformers_module.__spec__ = ModuleSpec("transformers", loader=None)
        transformers_module.CLIPTokenizer = type("CLIPTokenizer", (), {})
        transformers_module.CLIPTextModel = type("CLIPTextModel", (), {})
        transformers_module.CLIPTextModelWithProjection = type("CLIPTextModelWithProjection", (), {})
        transformers_module.optimization = sys.modules["transformers.optimization"]
        sys.modules["transformers"] = transformers_module

    stubbed_modules = {
        "library.custom_train_functions": ModuleType("library.custom_train_functions"),
        "library.sd3_utils": ModuleType("library.sd3_utils"),
        "library.model_util": ModuleType("library.model_util"),
        "library.huggingface_util": ModuleType("library.huggingface_util"),
        "library.deepspeed_utils": ModuleType("library.deepspeed_utils"),
    }

    sai_model_spec_module = ModuleType("library.sai_model_spec")
    sai_model_spec_module.ModelSpecMetadata = type("ModelSpecMetadata", (), {})
    sai_model_spec_module.build_metadata = lambda *args, **kwargs: {}
    sai_model_spec_module.build_metadata_dataclass = lambda *args, **kwargs: {}
    stubbed_modules["library.sai_model_spec"] = sai_model_spec_module

    lpw_module = ModuleType("library.lpw_stable_diffusion")
    lpw_module.StableDiffusionLongPromptWeightingPipeline = type("StableDiffusionLongPromptWeightingPipeline", (), {})
    stubbed_modules["library.lpw_stable_diffusion"] = lpw_module

    sdxl_lpw_module = ModuleType("library.sdxl_lpw_stable_diffusion")
    sdxl_lpw_module.SdxlStableDiffusionLongPromptWeightingPipeline = type(
        "SdxlStableDiffusionLongPromptWeightingPipeline", (), {}
    )
    stubbed_modules["library.sdxl_lpw_stable_diffusion"] = sdxl_lpw_module

    original_unet_module = ModuleType("library.original_unet")
    original_unet_module.UNet2DConditionModel = type("UNet2DConditionModel", (), {})
    stubbed_modules["library.original_unet"] = original_unet_module

    utils_module = ModuleType("library.utils")
    utils_module.setup_logging = lambda *args, **kwargs: None
    utils_module.resize_image = lambda image, *args, **kwargs: image
    utils_module.validate_interpolation_fn = lambda *args, **kwargs: True
    stubbed_modules["library.utils"] = utils_module

    strategy_base_module = ModuleType("library.strategy_base")

    class StrategyBase:
        _strategy = None

        @classmethod
        def get_strategy(cls):
            return cls._strategy

        @classmethod
        def set_strategy(cls, strategy):
            cls._strategy = strategy

    strategy_base_module.LatentsCachingStrategy = type("LatentsCachingStrategy", (StrategyBase,), {})
    strategy_base_module.TokenizeStrategy = type("TokenizeStrategy", (StrategyBase,), {})
    strategy_base_module.TextEncoderOutputsCachingStrategy = type("TextEncoderOutputsCachingStrategy", (StrategyBase,), {})
    strategy_base_module.TextEncodingStrategy = type("TextEncodingStrategy", (StrategyBase,), {})
    stubbed_modules["library.strategy_base"] = strategy_base_module

    for module_name, module in stubbed_modules.items():
        module.__spec__ = ModuleSpec(module_name, loader=None)
        sys.modules.setdefault(module_name, module)


install_train_util_import_stubs()

from library import train_util


def make_subset(custom_attributes=None):
    return train_util.FineTuningSubset(
        image_dir="D:/dataset",
        metadata_file="D:/dataset/metadata.json",
        alpha_mask=False,
        num_repeats=1,
        shuffle_caption=False,
        caption_separator=",",
        keep_tokens=0,
        keep_tokens_separator="",
        secondary_separator=None,
        enable_wildcard=False,
        color_aug=False,
        flip_aug=False,
        face_crop_aug_range=None,
        random_crop=False,
        caption_dropout_rate=0.0,
        caption_dropout_every_n_epochs=0,
        caption_tag_dropout_rate=0.0,
        caption_prefix=None,
        caption_suffix=None,
        token_warmup_min=1,
        token_warmup_step=0,
        custom_attributes=custom_attributes,
    )


def make_dataset(subset):
    dataset = train_util.MinimalDataset((1024, 1024), 1.0)
    dataset.subsets = [subset]
    return dataset


def test_process_caption_anima_markdown_section_dropout_deletes_whole_section():
    caption = (
        "## Character\n"
        "Alice\n\n"
        "## Atmosphere\n"
        "Moody fog\n\n"
        "## Artist\n"
        "@foo"
    )
    subset = make_subset({"markdown_section_dropout": {"Atmosphere": 1.0}})
    dataset = make_dataset(subset)
    dataset.enable_anima_markdown_section_dropout()

    with patch("library.train_util.random.random", return_value=0.0):
        actual = dataset.process_caption(subset, caption)

    assert actual == "## Character\nAlice\n\n## Artist\n@foo"


def test_process_caption_anima_markdown_section_dropout_prob_zero_keeps_caption():
    caption = (
        "## Character\n"
        "Alice\n\n"
        "## Atmosphere\n"
        "Moody fog"
    )
    subset = make_subset({"markdown_section_dropout": {"Atmosphere": 0.0}})
    dataset = make_dataset(subset)
    dataset.enable_anima_markdown_section_dropout()

    actual = dataset.process_caption(subset, caption)

    assert actual == "## Character"


def test_process_caption_anima_markdown_section_dropout_multiple_sections_are_independent():
    caption = (
        "## Character\n"
        "Alice\n\n"
        "## Atmosphere\n"
        "Moody fog\n\n"
        "## Artist\n"
        "@foo\n\n"
        "## Image effects\n"
        "Bloom"
    )
    subset = make_subset({"markdown_section_dropout": {"Atmosphere": 0.5, "Artist": 0.5}})
    dataset = make_dataset(subset)
    dataset.enable_anima_markdown_section_dropout()

    with patch("library.train_util.random.random", side_effect=[0.1, 0.8]):
        actual = dataset.process_caption(subset, caption)

    assert actual == "## Character\nAlice\n\n## Artist\n@foo\n\n## Image effects\nBloom"


def test_process_caption_anima_markdown_section_dropout_unknown_title_is_ignored():
    caption = (
        "## Character\n"
        "Alice\n\n"
        "## Atmosphere\n"
        "Moody fog"
    )
    subset = make_subset({"markdown_section_dropout": {"Background": 1.0}})
    dataset = make_dataset(subset)
    dataset.enable_anima_markdown_section_dropout()

    actual = dataset.process_caption(subset, caption)

    assert actual == caption


def test_process_caption_anima_markdown_section_dropout_repeated_titles_are_sampled_independently():
    caption = (
        "## Atmosphere\n"
        "Fog\n\n"
        "## Character\n"
        "Alice\n\n"
        "## Atmosphere\n"
        "Sunlight"
    )
    subset = make_subset({"markdown_section_dropout": {"Atmosphere": 0.5}})
    dataset = make_dataset(subset)
    dataset.enable_anima_markdown_section_dropout()

    with patch("library.train_util.random.random", side_effect=[0.1, 0.9]):
        actual = dataset.process_caption(subset, caption)

    assert actual == "## Character\nAlice\n\n## Atmosphere\nSunlight"


def test_process_caption_anima_markdown_section_dropout_collapses_extra_blank_lines():
    caption = (
        "## Character\n"
        "Alice\n\n\n"
        "## Atmosphere\n"
        "Fog\n\n"
        "## Artist\n"
        "@foo"
    )
    subset = make_subset({"markdown_section_dropout": {"Atmosphere": 1.0}})
    dataset = make_dataset(subset)
    dataset.enable_anima_markdown_section_dropout()

    with patch("library.train_util.random.random", return_value=0.0):
        actual = dataset.process_caption(subset, caption)

    assert actual == "## Character\nAlice\n\n## Artist\n@foo"


def test_process_caption_anima_markdown_section_dropout_is_not_enabled_by_default():
    caption = (
        "## Atmosphere\n"
        "Fog\n\n"
        "## Artist\n"
        "@foo"
    )
    subset = make_subset({"markdown_section_dropout": {"Atmosphere": 1.0}})
    dataset = make_dataset(subset)

    actual = dataset.process_caption(subset, caption)

    assert actual == "## Atmosphere"


def test_enable_anima_markdown_section_dropout_rejects_invalid_config():
    subset = make_subset({"markdown_section_dropout": ["Atmosphere"]})
    dataset = make_dataset(subset)

    with pytest.raises(ValueError, match="markdown_section_dropout"):
        dataset.enable_anima_markdown_section_dropout()


def test_anima_markdown_section_dropout_disables_text_encoder_output_cache():
    subset = make_subset({"markdown_section_dropout": {"Atmosphere": 0.5}})
    dataset = make_dataset(subset)
    dataset.enable_anima_markdown_section_dropout()

    assert dataset.is_text_encoder_output_cacheable(cache_supports_dropout=True) is False


def test_subset_custom_attributes_are_normalized_to_picklable_builtin_containers():
    class InlineTableDict(dict):
        pass

    subset = make_subset(
        InlineTableDict(
            {
                "markdown_section_dropout": InlineTableDict({"Character 1": 1.0, "Atmosphere": 0.5}),
            }
        )
    )

    assert type(subset.custom_attributes) is dict
    assert type(subset.custom_attributes["markdown_section_dropout"]) is dict
    assert subset.custom_attributes["markdown_section_dropout"]["Character 1"] == 1.0

    pickle.dumps(subset.custom_attributes)
