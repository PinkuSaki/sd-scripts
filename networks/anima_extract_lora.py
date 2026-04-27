# extract approximating LoRA by svd from two Anima models
# The code is based on https://github.com/cloneofsimo/lora/blob/develop/lora_diffusion/cli_svd.py
# Thanks to cloneofsimo!

import argparse
import gc
import json
import os
import sys
import time
import logging

import torch
from tqdm import tqdm

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
    if p == "float":
        return torch.float
    if p == "fp16":
        return torch.float16
    if p == "bf16":
        return torch.bfloat16
    return None


def load_anima_dit(model_path: str, load_dtype: torch.dtype | None):
    from library import anima_utils

    logger.info(f"loading Anima DiT model: {model_path}")
    return anima_utils.load_anima_model("cpu", model_path, "torch", False, "cpu", load_dtype, False)


def load_qwen3_text_encoder(model_path: str, load_dtype: torch.dtype | None):
    from library import anima_utils

    qwen3_dtype = load_dtype if load_dtype is not None else torch.float
    logger.info(f"loading Qwen3 text encoder: {model_path} (dtype={qwen3_dtype})")
    text_encoder, _ = anima_utils.load_qwen3_text_encoder(model_path, dtype=qwen3_dtype, device="cpu")
    return text_encoder


def build_network_kwargs(
    train_llm_adapter: bool,
    include_patterns: str | None,
    exclude_patterns: str | None,
    network_reg_dims: str | None,
    verbose: bool,
):
    kwargs = {}
    if train_llm_adapter:
        kwargs["train_llm_adapter"] = "true"
    if include_patterns is not None:
        kwargs["include_patterns"] = include_patterns
    if exclude_patterns is not None:
        kwargs["exclude_patterns"] = exclude_patterns
    if network_reg_dims is not None:
        kwargs["network_reg_dims"] = network_reg_dims
    if verbose:
        kwargs["verbose"] = "true"
    return kwargs


def extract_diff_modules(loras_o, loras_t, work_device: str):
    diffs = {}
    max_abs_diffs = {}

    for lora_o, lora_t in zip(loras_o, loras_t):
        lora_name = lora_o.lora_name
        module_o = lora_o.org_module
        module_t = lora_t.org_module

        diff = module_t.weight.to(work_device) - module_o.weight.to(work_device)
        max_abs_diffs[lora_name] = torch.max(torch.abs(diff)).item()
        diffs[lora_name] = diff

        # release references early to save memory
        module_o.weight = None
        module_t.weight = None

    return diffs, max_abs_diffs


def svd(
    model_org=None,
    model_tuned=None,
    save_to=None,
    dim=4,
    device=None,
    save_precision=None,
    clamp_quantile=0.99,
    min_diff=0.01,
    no_metadata=False,
    load_precision=None,
    qwen3_org=None,
    qwen3_tuned=None,
    train_llm_adapter=False,
    include_patterns=None,
    exclude_patterns=None,
    network_reg_dims=None,
    verbose=False,
):
    from library import sai_model_spec
    from networks import lora_anima

    if (qwen3_org is None) != (qwen3_tuned is None):
        raise ValueError("qwen3_org and qwen3_tuned must be specified together / qwen3_org 和 qwen3_tuned 需要同时指定")

    load_dtype = str_to_dtype(load_precision) if load_precision else None
    save_dtype = str_to_dtype(save_precision)
    work_device = "cpu"

    text_encoders_o = None
    text_encoders_t = None
    if qwen3_org is not None:
        text_encoders_o = [load_qwen3_text_encoder(qwen3_org, load_dtype)]
        text_encoders_t = [load_qwen3_text_encoder(qwen3_tuned, load_dtype)]

    unet_o = load_anima_dit(model_org, load_dtype)
    unet_t = load_anima_dit(model_tuned, load_dtype)

    net_kwargs = build_network_kwargs(train_llm_adapter, include_patterns, exclude_patterns, network_reg_dims, verbose)

    # create LoRA network to discover target modules: use dim as alpha for extracted weights
    lora_network_o = lora_anima.create_network(1.0, dim, dim, None, text_encoders_o, unet_o, **net_kwargs)
    lora_network_t = lora_anima.create_network(1.0, dim, dim, None, text_encoders_t, unet_t, **net_kwargs)

    diffs = {}

    if len(lora_network_o.text_encoder_loras) != len(lora_network_t.text_encoder_loras):
        raise ValueError("Qwen3 model structures are different between original and tuned models / 原始和目标 Qwen3 模型结构不一致")

    if len(lora_network_o.unet_loras) != len(lora_network_t.unet_loras):
        raise ValueError("Anima model structures are different between original and tuned models / 原始和目标 Anima 模型结构不一致")

    logger.info(
        f"selected LoRA targets: text_encoder={len(lora_network_o.text_encoder_loras)}, "
        f"dit={len(lora_network_o.unet_loras)}, train_llm_adapter={train_llm_adapter}"
    )

    text_encoder_different = qwen3_org is None
    if qwen3_org is not None:
        te_diffs, te_max_abs_diffs = extract_diff_modules(lora_network_o.text_encoder_loras, lora_network_t.text_encoder_loras, work_device)
        if te_max_abs_diffs:
            max_te_diff = max(te_max_abs_diffs.values())
            text_encoder_different = max_te_diff > min_diff
            if text_encoder_different:
                logger.info(f"Qwen3 text encoder is different. {max_te_diff} > {min_diff}")
                diffs.update(te_diffs)
            else:
                logger.warning(f"Qwen3 text encoder is same or difference is <= min_diff ({max_te_diff} <= {min_diff}). Extract DiT only.")
        del te_diffs, te_max_abs_diffs
        gc.collect()

    dit_diffs, dit_max_abs_diffs = extract_diff_modules(lora_network_o.unet_loras, lora_network_t.unet_loras, work_device)
    diffs.update(dit_diffs)
    if dit_max_abs_diffs:
        logger.info(f"max DiT diff: {max(dit_max_abs_diffs.values())}")
    del dit_diffs, dit_max_abs_diffs

    # release tuned-side models early
    del lora_network_t
    del unet_t
    if text_encoders_t is not None:
        del text_encoders_t
    gc.collect()

    logger.info("calculating SVD")
    lora_weights = {}
    with torch.no_grad():
        for lora_name, mat in tqdm(list(diffs.items())):
            mat = mat.to(torch.float)

            conv2d = len(mat.size()) == 4
            kernel_size = None if not conv2d else mat.size()[2:4]

            out_dim, in_dim = mat.size()[0:2]

            if device:
                mat = mat.to(device)

            rank = min(dim, in_dim, out_dim)

            if conv2d:
                if kernel_size != (1, 1):
                    mat = mat.flatten(start_dim=1)
                else:
                    mat = mat.squeeze()

            U, S, Vh = torch.linalg.svd(mat)

            U = U[:, :rank]
            S = S[:rank]
            U = U @ torch.diag(S)
            Vh = Vh[:rank, :]

            dist = torch.cat([U.flatten(), Vh.flatten()])
            hi_val = torch.quantile(dist, clamp_quantile)
            low_val = -hi_val

            U = U.clamp(low_val, hi_val)
            Vh = Vh.clamp(low_val, hi_val)

            if conv2d:
                U = U.reshape(out_dim, rank, 1, 1)
                Vh = Vh.reshape(rank, in_dim, kernel_size[0], kernel_size[1])

            U = U.to(work_device, dtype=save_dtype).contiguous()
            Vh = Vh.to(work_device, dtype=save_dtype).contiguous()

            lora_weights[lora_name] = (U, Vh)

    if len(lora_weights) == 0:
        raise ValueError("No LoRA weights were extracted. Please check the selected modules / 没有提取到任何 LoRA 权重，请检查目标模块选择")

    # make state dict for LoRA
    lora_sd = {}
    for lora_name, (up_weight, down_weight) in lora_weights.items():
        lora_sd[lora_name + ".lora_up.weight"] = up_weight
        lora_sd[lora_name + ".lora_down.weight"] = down_weight
        lora_sd[lora_name + ".alpha"] = torch.tensor(down_weight.size()[0])  # same as rank

    # validate extracted state dict against the Anima LoRA module and save it with the existing helper
    lora_network_save, lora_sd = lora_anima.create_network_from_weights(
        1.0, None, None, text_encoders_o, unet_o, weights_sd=lora_sd
    )
    lora_network_save.apply_to(text_encoders_o, unet_o)
    info = lora_network_save.load_state_dict(lora_sd)
    logger.info(f"Loading extracted LoRA weights: {info}")

    dir_name = os.path.dirname(save_to)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    metadata = {
        "ss_v2": str(False),
        "ss_base_model_version": sai_model_spec.ARCH_ANIMA_PREVIEW,
        "ss_network_module": "networks.lora_anima",
        "ss_network_dim": str(dim),
        "ss_network_alpha": str(float(dim)),
        "ss_network_args": json.dumps(net_kwargs),
    }

    if not no_metadata:
        title = os.path.splitext(os.path.basename(save_to))[0]
        sai_metadata = sai_model_spec.build_metadata(
            None, False, False, False, True, False, time.time(), title=title, model_config={"anima": "preview"}
        )
        metadata.update(sai_metadata)

    lora_network_save.save_weights(save_to, save_dtype, metadata)
    logger.info(f"LoRA weights saved to {save_to}")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load_precision",
        type=str,
        default=None,
        choices=[None, "float", "fp16", "bf16"],
        help="precision in loading, model default if omitted / 读取精度，省略时 DiT 使用模型原精度，Qwen3 使用 float",
    )
    parser.add_argument(
        "--save_precision",
        type=str,
        default=None,
        choices=[None, "float", "fp16", "bf16"],
        help="precision in saving, same to merging if omitted / 保存精度，省略时保存为 float",
    )
    parser.add_argument(
        "--model_org",
        type=str,
        default=None,
        required=True,
        help="Original Anima DiT model: safetensors file / 原始 Anima DiT 模型，safetensors 文件",
    )
    parser.add_argument(
        "--model_tuned",
        type=str,
        default=None,
        required=True,
        help="Tuned Anima DiT model, LoRA is difference of `original to tuned`: safetensors file / 目标 Anima DiT 模型，生成的 LoRA 表示原始模型到目标模型的差分",
    )
    parser.add_argument(
        "--qwen3_org",
        type=str,
        default=None,
        help="Original Qwen3 text encoder, optional. Specify together with --qwen3_tuned to also extract text encoder LoRA / 原始 Qwen3 文本编码器，可选；与 --qwen3_tuned 一起指定时会同时提取文本编码器 LoRA",
    )
    parser.add_argument(
        "--qwen3_tuned",
        type=str,
        default=None,
        help="Tuned Qwen3 text encoder, optional. Specify together with --qwen3_org / 目标 Qwen3 文本编码器，可选；需要和 --qwen3_org 一起指定",
    )
    parser.add_argument(
        "--save_to",
        type=str,
        default=None,
        required=True,
        help="destination file name: safetensors file / 保存目标文件名，safetensors",
    )
    parser.add_argument("--dim", type=int, default=4, help="dimension (rank) of LoRA (default 4) / LoRA rank，默认 4")
    parser.add_argument(
        "--device", type=str, default=None, help="device to use for SVD, cuda for GPU / 执行 SVD 的设备，cuda 表示使用 GPU"
    )
    parser.add_argument(
        "--clamp_quantile",
        type=float,
        default=0.99,
        help="Quantile clamping value, float, (0-1). Default = 0.99 / 用于裁剪奇异值分解结果的分位数，默认 0.99",
    )
    parser.add_argument(
        "--min_diff",
        type=float,
        default=0.01,
        help="Minimum difference used to decide whether optional Qwen3 extraction is meaningful. Default = 0.01 / 判断可选 Qwen3 差分是否足够明显的阈值，默认 0.01",
    )
    parser.add_argument(
        "--train_llm_adapter",
        action="store_true",
        help="also extract LoRA for LLM Adapter blocks / 同时为 LLM Adapter 模块提取 LoRA",
    )
    parser.add_argument(
        "--include_patterns",
        type=str,
        default=None,
        help="python-style list string for include_patterns, same as networks.lora_anima / include_patterns，格式与 networks.lora_anima 一致，例如 \"['.*final_layer.*']\"",
    )
    parser.add_argument(
        "--exclude_patterns",
        type=str,
        default=None,
        help="python-style list string for exclude_patterns, same as networks.lora_anima / exclude_patterns，格式与 networks.lora_anima 一致",
    )
    parser.add_argument(
        "--network_reg_dims",
        type=str,
        default=None,
        help="regex-based per-module rank settings, same as networks.lora_anima / 基于正则的模块级 rank 设置，格式如 \".*self_attn.*=8,.*cross_attn.*=4\"",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print selected module names from networks.lora_anima / 打印 networks.lora_anima 选中的模块名",
    )
    parser.add_argument(
        "--no_metadata",
        action="store_true",
        help="do not save sai modelspec metadata (minimum ss_metadata for LoRA is saved) / 不保存 sai modelspec 元数据（仍会保留 LoRA 所需的最小 ss_metadata）",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    svd(**vars(args))
