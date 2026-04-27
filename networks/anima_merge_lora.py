# Merge LoRA weights into an Anima DiT model.

import argparse
import logging
import os
import sys
import time

import torch
from safetensors.torch import load_file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from library.utils import setup_logging
except ModuleNotFoundError:

    def setup_logging():
        logging.basicConfig(level=logging.INFO)


setup_logging()
logger = logging.getLogger(__name__)


def str_to_dtype(p):
    if p in (None, "None"):
        return None
    if p in ("float", "float32"):
        return torch.float
    if p == "fp16":
        return torch.float16
    if p == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {p}")


def load_lora_for_dit(file_name: str):
    if os.path.splitext(file_name)[1] == ".safetensors":
        sd = load_file(file_name)
    else:
        try:
            sd = torch.load(file_name, map_location="cpu", weights_only=True)
        except TypeError:
            sd = torch.load(file_name, map_location="cpu")

    text_encoder_keys = [k for k in sd.keys() if k.startswith("lora_te_")]
    if text_encoder_keys:
        logger.warning(
            f"{file_name} contains {len(text_encoder_keys)} Qwen3 text encoder LoRA tensors. "
            "Only lora_unet/DiT tensors are merged into the Anima DiT model."
        )

    # Keep unprefixed keys as well; the shared LoRA merge helper supports both
    # lora_unet_* and raw module-name prefixes.
    dit_sd = {k: v for k, v in sd.items() if not k.startswith("lora_te_")}
    if len(dit_sd) == 0:
        raise ValueError(f"No DiT LoRA weights found in {file_name}")

    return dit_sd


def merge(args):
    from library import anima_utils, sai_model_spec

    if args.models is None or len(args.models) == 0:
        raise ValueError("--models must contain at least one LoRA file / --models 至少需要指定一个 LoRA 文件")

    if args.ratios is None:
        args.ratios = [1.0] * len(args.models)
    while len(args.ratios) < len(args.models):
        args.ratios.append(1.0)
    if len(args.ratios) > len(args.models):
        args.ratios = args.ratios[: len(args.models)]

    if args.save_to is None:
        raise ValueError("--save_to is required / 必须指定 --save_to")

    dest_dir = os.path.dirname(args.save_to)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)

    merge_dtype = str_to_dtype(args.precision)
    save_dtype = str_to_dtype(args.save_precision)
    if save_dtype is None:
        save_dtype = merge_dtype

    lora_weights_list = []
    for model in args.models:
        logger.info(f"loading LoRA: {model}")
        lora_weights_list.append(load_lora_for_dit(model))

    logger.info(f"loading Anima DiT and merging LoRA weights: {args.dit}")
    dit = anima_utils.load_anima_model(
        args.working_device,
        args.dit,
        "torch",
        False,
        args.loading_device,
        merge_dtype,
        False,
        lora_weights_list=lora_weights_list,
        lora_multipliers=args.ratios,
    )
    dit.eval().requires_grad_(False)

    metadata = None
    if not args.no_metadata:
        merged_from = sai_model_spec.build_merged_from([args.dit] + args.models)
        title = os.path.splitext(os.path.basename(args.save_to))[0]
        metadata = sai_model_spec.build_metadata(
            None,
            False,
            False,
            False,
            False,
            False,
            time.time(),
            title=title,
            merged_from=merged_from,
            model_config={"anima": "preview"},
        )

    logger.info(f"saving merged Anima DiT model to: {args.save_to}")
    anima_utils.save_anima_model(args.save_to, dit.state_dict(), metadata, save_dtype)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dit",
        type=str,
        required=True,
        help="Anima DiT model to load: safetensors file / 要加载的 Anima DiT 模型，safetensors 文件",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        required=True,
        help="LoRA models to merge: safetensors files / 要合并的 LoRA 模型，safetensors 文件",
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="*",
        default=None,
        help="ratios for each LoRA model, default 1.0 for each / 每个 LoRA 的合并倍率，默认都是 1.0",
    )
    parser.add_argument(
        "--save_to",
        type=str,
        required=True,
        help="destination file name: safetensors file / 保存目标文件名，safetensors 文件",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="float",
        choices=["float", "float32", "fp16", "bf16"],
        help="precision in merging, float is recommended / 合并计算精度，推荐 float",
    )
    parser.add_argument(
        "--save_precision",
        type=str,
        default=None,
        choices=["float", "float32", "fp16", "bf16"],
        help="precision in saving, same as merging if omitted / 保存精度，省略时与合并计算精度相同",
    )
    parser.add_argument(
        "--loading_device",
        type=str,
        default="cpu",
        help="device to load the Anima DiT model / 加载 Anima DiT 模型的设备",
    )
    parser.add_argument(
        "--working_device",
        type=str,
        default="cpu",
        help="device for LoRA merge calculation / LoRA 合并计算使用的设备",
    )
    parser.add_argument(
        "--no_metadata",
        action="store_true",
        help="do not save sai modelspec metadata / 不保存 sai modelspec 元数据",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    merge(args)
