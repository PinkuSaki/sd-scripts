# 使用 `anima_train_network.py` 进行 Anima 模型的 LoRA 训练指南

本文档介绍如何使用 `sd-scripts` 仓库中的 `anima_train_network.py` 脚本，对 Anima 模型训练 LoRA（Low-Rank Adaptation）。

## 1. 简介

`anima_train_network.py` 用于为 Anima 模型训练 LoRA 等附加网络。Anima 采用基于 MiniTrainDIT 设计的 DiT（Diffusion Transformer）架构，使用 Rectified Flow 训练方法。模型由以下组件构成：

- **Qwen3-0.6B 文本编码器**：负责文本编码
- **LLM Adapter**：6 层 Transformer 桥接模块，将 Qwen3 嵌入转换到 T5 兼容的交叉注意力空间
- **Qwen-Image VAE**：16 通道潜空间、8 倍空间下采样

本指南假设你已了解 LoRA 训练的基本知识。通用参数请参考 [train_network.py 指南](train_network.md)。部分参数与 [`sd3_train_network.py`](sd3_train_network.md) 和 [`flux_train_network.py`](flux_train_network.md) 类似。

**前提条件：**

* 已克隆 `sd-scripts` 仓库并完成 Python 环境配置
* 已准备好训练数据集（参见[数据集配置指南](./config_README-en.md)）
* 已准备好 Anima 模型文件

## 2. 与 `train_network.py` 的差异

`anima_train_network.py` 基于 `train_network.py`，针对 Anima 模型进行了适配。主要差异如下：

* **目标模型：** Anima DiT 模型
* **模型结构：** 使用 MiniTrainDIT（Transformer 架构）替代 U-Net。使用单个文本编码器（Qwen3-0.6B）、LLM Adapter 以及 Qwen-Image VAE（16 通道潜空间，8 倍空间下采样）
* **参数变化：** DiT 模型路径使用 `--pretrained_model_name_or_path`，Qwen3 文本编码器使用 `--qwen3`，VAE 使用 `--vae`。LLM Adapter 和 T5 分词器可分别通过 `--llm_adapter_path` 和 `--t5_tokenizer_path` 指定
* **不兼容的参数：** SD v1/v2 专用参数（如 `--v2`、`--v_parameterization`、`--clip_skip`）不适用。`--fp8_base` 不被支持
* **时间步采样：** 使用与 FLUX 训练相同的 `--timestep_sampling` 选项（`sigma`、`uniform`、`sigmoid`、`shift`、`flux_shift`）
* **LoRA 模块选择：** 使用正则表达式进行模块选择，支持按模块设置 rank/学习率（`network_reg_dims`、`network_reg_lrs`），而非按组件指定参数。通过 `exclude_patterns` 和 `include_patterns` 控制模块的排除/包含

## 3. 准备工作

训练前需要准备以下文件：

1. **训练脚本：** `anima_train_network.py`
2. **Anima DiT 模型文件：** `.safetensors` 格式的基础 DiT 模型
3. **Qwen3-0.6B 文本编码器：** HuggingFace 模型目录或单个 `.safetensors` 文件（使用内置的 `configs/qwen3_06b/` 配置文件）
4. **Qwen-Image VAE 模型文件：** `.safetensors` 或 `.pth` 格式
5. **LLM Adapter 模型文件（可选）：** `.safetensors` 文件。如未单独提供，当 DiT 文件中包含 `llm_adapter.out_proj.weight` 键时会自动从中加载
6. **T5 分词器（可选）：** 如未指定则使用内置的 `configs/t5_old/` 分词器
7. **数据集定义文件（.toml）：** TOML 格式的数据集配置文件（参见[数据集配置指南](./config_README-en.md)）

模型文件可从 [Anima HuggingFace 仓库](https://huggingface.co/circlestone-labs/Anima) 获取。

**说明：**
* T5 分词器只需要分词器文件，不需要 T5 模型权重。使用 `google/t5-v1_1-xxl` 的词表。

## 4. 运行训练

从终端执行 `anima_train_network.py` 开始训练。命令行整体格式与 `train_network.py` 相同，但需要指定 Anima 特有的参数。

### 训练命令示例

```bash
accelerate launch --num_cpu_threads_per_process 1 anima_train_network.py \
  --pretrained_model_name_or_path="<Anima DiT 模型路径>" \
  --qwen3="<Qwen3-0.6B 模型路径或目录>" \
  --vae="<Qwen-Image VAE 模型路径>" \
  --dataset_config="my_anima_dataset_config.toml" \
  --output_dir="<输出目录>" \
  --output_name="my_anima_lora" \
  --save_model_as=safetensors \
  --network_module=networks.lora_anima \
  --network_dim=8 \
  --learning_rate=1e-4 \
  --optimizer_type="AdamW8bit" \
  --lr_scheduler="constant" \
  --timestep_sampling="sigmoid" \
  --discrete_flow_shift=1.0 \
  --max_train_epochs=10 \
  --save_every_n_epochs=1 \
  --mixed_precision="bf16" \
  --gradient_checkpointing \
  --cache_latents \
  --cache_text_encoder_outputs \
  --vae_chunk_size=64 \
  --vae_disable_cache
```

*（请将命令写成一行，或使用 `\`（Linux/macOS）或 `^`（Windows）进行换行。）*

学习率 `1e-4` 仅为示例值，请根据数据集和训练目标适当调整。此值对应 `alpha=1.0`（默认值）。如果增大 `--network_alpha`，请考虑降低学习率。

如果 loss 出现 NaN，请确保 PyTorch 版本为 2.5 或更高。

**注意：** `--vae_chunk_size` 和 `--vae_disable_cache` 是本仓库特有的选项，用于降低 Qwen-Image VAE 的显存占用。

### 4.1. 主要命令行参数说明

除 [train_network.py 指南](train_network.md) 中的通用参数外，需指定以下 Anima 特有参数。

#### 模型相关 [必需]

* `--pretrained_model_name_or_path="<Anima DiT 模型路径>"` **[必需]**
  - Anima DiT 模型 `.safetensors` 文件路径。模型配置（通道数、块数、注意力头数）从 state dict 自动检测。支持带 `net.` 前缀的 ComfyUI 格式。
* `--qwen3="<Qwen3-0.6B 模型路径>"` **[必需]**
  - Qwen3-0.6B 文本编码器路径。可以是 HuggingFace 模型目录或单个 `.safetensors` 文件。训练期间文本编码器始终处于冻结状态。
* `--vae="<Qwen-Image VAE 模型路径>"` **[必需]**
  - Qwen-Image VAE 模型文件路径（`.safetensors` 或 `.pth`）。固定配置：`dim=96, z_dim=16`。

#### 模型相关 [可选]

* `--llm_adapter_path="<LLM Adapter 路径>"` *[可选]*
  - 单独的 LLM Adapter 权重文件路径。如未指定，当 DiT 文件中存在 `llm_adapter.out_proj.weight` 键时，会自动从中加载。
* `--t5_tokenizer_path="<T5 分词器路径>"` *[可选]*
  - T5 分词器目录路径。如未指定，使用内置的 `configs/t5_old/` 分词器。

#### Anima 训练参数

* `--timestep_sampling=<方法>`
  - 时间步采样方法。可选 `sigma`、`uniform`、`sigmoid`（默认）、`shift`、`flux_shift`。与 FLUX 训练相同，各方法详见 [flux_train_network.py 指南](flux_train_network.md)。
* `--discrete_flow_shift=<浮点数>`
  - Rectified Flow 训练的时间步分布偏移量，默认 `1.0`。仅在 `--timestep_sampling` 设为 `shift` 时生效。偏移公式为：`t_shifted = (t * shift) / (1 + (shift - 1) * t)`。
* `--sigmoid_scale=<浮点数>`
  - `--timestep_sampling` 设为 `sigmoid`、`shift` 或 `flux_shift` 时的缩放因子，默认 `1.0`。
* `--qwen3_max_token_length=<整数>`
  - Qwen3 分词器的最大 token 长度，默认 `512`。
* `--t5_max_token_length=<整数>`
  - T5 分词器的最大 token 长度，默认 `512`。
* `--attn_mode=<模式>`
  - 注意力实现方式。可选 `torch`（默认）、`xformers`、`flash`、`sageattn`。`xformers` 需同时指定 `--split_attn`。`sageattn` 仅支持推理，不支持训练。此参数会覆盖 `--xformers`。
* `--split_attn`
  - 拆分注意力计算以降低显存占用。使用 `--attn_mode xformers` 时必需。

#### 组件级学习率

这些参数为 Anima 模型各组件设置独立的学习率，主要用于全量微调。设为 `0` 可冻结对应组件：

* `--self_attn_lr=<浮点数>` — 自注意力层学习率。默认与 `--learning_rate` 相同。
* `--cross_attn_lr=<浮点数>` — 交叉注意力层学习率。默认与 `--learning_rate` 相同。
* `--mlp_lr=<浮点数>` — MLP 层学习率。默认与 `--learning_rate` 相同。
* `--mod_lr=<浮点数>` — AdaLN 调制层学习率。默认与 `--learning_rate` 相同。注意：调制层默认不包含在 LoRA 中。
* `--llm_adapter_lr=<浮点数>` — LLM Adapter 层学习率。默认与 `--learning_rate` 相同。

LoRA 训练时请使用 `--network_args` 中的 `network_reg_lrs`，参见[第 5.2 节](#52-基于正则表达式的-rank-和学习率控制)。

#### 显存与速度相关

* `--blocks_to_swap=<整数>`
  - 在 CPU 和 GPU 之间交换的 Transformer 块数量。数值越大越省显存，但会降低训练速度。最大值取决于模型规模：
    - 28 块模型（Anima-Preview）：最大 **26**
    - 36 块模型：最大 **34**
    - 20 块模型：最大 **18**
  - 不可与 `--cpu_offload_checkpointing`、`--unsloth_offload_checkpointing` 或 `--deepspeed` 同时使用。
* `--unsloth_offload_checkpointing`
  - 使用异步非阻塞传输将激活值卸载到 CPU 内存（比 `--cpu_offload_checkpointing` 更快）。不可与 `--cpu_offload_checkpointing`、`--blocks_to_swap` 或 `--deepspeed` 同时使用。
* `--cache_text_encoder_outputs`
  - 缓存 Qwen3 文本编码器输出以降低显存占用。不训练文本编码器 LoRA 时推荐使用。
* `--cache_text_encoder_outputs_to_disk`
  - 将文本编码器输出缓存到磁盘。自动启用 `--cache_text_encoder_outputs`。
* `--cache_latents` / `--cache_latents_to_disk`
  - 缓存 Qwen-Image VAE 潜空间输出。
* `--vae_chunk_size=<整数>`
  - Qwen-Image VAE 的分块处理大小。降低显存占用但会降低速度。默认不分块。
* `--vae_disable_cache`
  - 禁用 Qwen-Image VAE 的内部缓存以降低显存占用。

#### DeepSpeed 注意事项

* Anima LoRA 训练可以使用 DeepSpeed。使用 ZeRO-3 时，LoRA 权重会从 DeepSpeed 聚合后的 state dict 中保存，避免只保存本地分片。
* `--mixed_precision=bf16` 搭配 8bit optimizer 时，`library/deepspeed_utils.py` 会自动回退到兼容的非 8bit optimizer。
* 不要将 DeepSpeed 与 `--blocks_to_swap` 或 `--unsloth_offload_checkpointing` 混用；这两条显存路径没有接入 Anima LoRA 的 DeepSpeed engine。

#### 不兼容/不支持的参数

* `--v2`、`--v_parameterization`、`--clip_skip` — Stable Diffusion v1/v2 专用参数，Anima 训练中不使用。
* `--fp8_base` — Anima 不支持，指定后会被自动禁用并输出警告。

## 5. LoRA 目标模块

使用 `anima_train_network.py` 训练 LoRA 时，默认对以下模块进行训练：

* **DiT 块（`Block`）**：每个 Transformer 块中的自注意力（`self_attn`）、交叉注意力（`cross_attn`）和 MLP（`mlp`）层。调制层（`adaln_modulation`）、归一化层、嵌入层和最终层默认被排除。
* **嵌入层（`PatchEmbed`、`TimestepEmbedding`）和最终层（`FinalLayer`）**：默认排除，可通过 `include_patterns` 包含。
* **LLM Adapter 块（`LLMAdapterTransformerBlock`）**：仅在指定 `--network_args "train_llm_adapter=True"` 时训练。
* **文本编码器（Qwen3）**：仅在未指定 `--network_train_unet_only` 且未使用 `--cache_text_encoder_outputs` 时训练。

LoRA 网络模块为 `networks.lora_anima`。

### 5.1. 使用正则表达式选择模块

默认通过以下内置排除模式将部分模块从 LoRA 中排除：
```
.*(_modulation|_norm|_embedder|final_layer).*
```

可通过 `--network_args` 使用正则表达式自定义包含/排除的模块：

* `exclude_patterns` — 排除匹配这些模式的模块（在默认排除基础上追加）
* `include_patterns` — 强制包含匹配这些模式的模块（覆盖排除规则）

模式使用 `re.fullmatch()` 对完整模块名进行匹配。

包含最终层的示例：
```
--network_args "include_patterns=['.*final_layer.*']"
```

额外排除 MLP 层的示例：
```
--network_args "exclude_patterns=['.*mlp.*']"
```

### 5.2. 基于正则表达式的 Rank 和学习率控制

可以为匹配特定正则表达式的模块指定不同的 rank（network_dim）和学习率：

* `network_reg_dims`：格式为逗号分隔的 `pattern=rank` 字符串。
    * 示例：`--network_args "network_reg_dims=.*self_attn.*=8,.*cross_attn.*=4,.*mlp.*=8"`
    * 将自注意力模块的 rank 设为 8，交叉注意力设为 4，MLP 设为 8。
* `network_reg_lrs`：格式为逗号分隔的 `pattern=lr` 字符串。
    * 示例：`--network_args "network_reg_lrs=.*self_attn.*=1e-4,.*cross_attn.*=5e-5"`
    * 将自注意力模块的学习率设为 `1e-4`，交叉注意力设为 `5e-5`。

**注意：**
* `network_reg_dims` 和 `network_reg_lrs` 的设置优先于全局的 `--network_dim` 和 `--learning_rate`。
* 模式使用 `re.fullmatch()` 对模块原始名称（如 `blocks.0.self_attn.q_proj`）进行匹配。

### 5.3. LLM Adapter LoRA

要对 LLM Adapter 块应用 LoRA：

```
--network_args "train_llm_adapter=True"
```

初步测试表明，适当降低 LLM Adapter 的学习率可提高稳定性。建议使用 `"network_reg_lrs=.*llm_adapter.*=5e-5"` 等方式进行调整。

### 5.4. 其他网络参数

* `--network_args "verbose=True"` — 打印所有 LoRA 模块名称及其维度
* `--network_args "rank_dropout=0.1"` — Rank Dropout 比率
* `--network_args "module_dropout=0.1"` — Module Dropout 比率
* `--network_args "loraplus_lr_ratio=2.0"` — LoRA+ 学习率比率
* `--network_args "loraplus_unet_lr_ratio=2.0"` — 仅 DiT 的 LoRA+ 学习率比率
* `--network_args "loraplus_text_encoder_lr_ratio=2.0"` — 仅文本编码器的 LoRA+ 学习率比率

## 6. 使用训练好的模型

训练完成后，LoRA 模型文件（如 `my_anima_lora.safetensors`）会保存到 `output_dir` 指定的目录中。该文件可在支持 Anima 的推理环境（如配合相应节点的 ComfyUI）中使用。

## 7. 高级设置

### 7.1. 显存优化

Anima 模型较大，显存有限的 GPU 可能需要以下优化措施：

#### 主要显存优化选项

- **`--blocks_to_swap <数量>`**：在 CPU 与 GPU 之间交换 Transformer 块以降低显存占用。数值越大越省显存，但训练速度越慢。各模型的最大值见第 4.1 节。不可与 `--deepspeed` 同时使用。

- **`--unsloth_offload_checkpointing`**：使用异步非阻塞传输将梯度检查点卸载到 CPU，比 `--cpu_offload_checkpointing` 更快。不可与 `--blocks_to_swap` 或 `--deepspeed` 同时使用。

- **`--gradient_checkpointing`**：标准梯度检查点，以计算换显存。

- **`--cache_text_encoder_outputs`**：缓存 Qwen3 输出，训练期间释放文本编码器占用的显存。

- **`--cache_latents`**：缓存 Qwen-Image VAE 输出，训练期间释放 VAE 占用的显存。

- **使用 Adafactor 优化器**：进一步降低显存占用：
  ```
  --optimizer_type adafactor --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" --lr_scheduler constant_with_warmup --max_grad_norm 0.0
  ```

### 7.2. 训练设置

#### 时间步采样

`--timestep_sampling` 指定时间步的采样方式，可用方法与 FLUX 训练相同：

- `sigma`：类似 SD3 的基于 sigma 的采样
- `uniform`：从 [0, 1] 均匀随机采样
- `sigmoid`（默认）：从 Normal(0,1) 采样后乘以 `sigmoid_scale`，再应用 sigmoid。通用性较好
- `shift`：类似 `sigmoid`，但额外应用离散流偏移公式：`t_shifted = (t * shift) / (1 + (shift - 1) * t)`
- `flux_shift`：FLUX 训练中使用的分辨率依赖偏移

详见 [flux_train_network.py 指南](flux_train_network.md)。

#### 离散流偏移

`--discrete_flow_shift`（默认 `1.0`）仅在 `--timestep_sampling` 设为 `shift` 时生效。公式为：

```
t_shifted = (t * shift) / (1 + (shift - 1) * t)
```

#### 损失权重

`--weighting_scheme` 指定按时间步的损失加权方式：

- `uniform`（默认）：所有时间步等权重
- `sigma_sqrt`：按 `sigma^(-2)` 加权
- `cosmap`：按 `2 / (pi * (1 - 2*sigma + 2*sigma^2))` 加权
- `none`：与 uniform 相同
- `logit_normal`、`mode`：来自 SD3 训练的额外方案，详见 [`sd3_train_network.md` 指南](sd3_train_network.md)

#### Caption Dropout

Caption Dropout 使用数据集配置中的 `caption_dropout_rate` 设置（TOML 中按子集配置）。使用 `--cache_text_encoder_outputs` 时，Dropout 比率会随每个缓存条目保存，并在训练时应用，因此 Caption Dropout 与文本编码器输出缓存兼容。

**如果更改了 `caption_dropout_rate` 设置，必须删除并重新生成缓存。**

注意：目前仅 Anima 支持将 `caption_dropout_rate` 与文本编码器输出缓存结合使用。

### 7.3. 文本编码器 LoRA 支持

Anima LoRA 训练支持训练 Qwen3 文本编码器的 LoRA：

- 仅训练 DiT：指定 `--network_train_unet_only`
- 同时训练 DiT 和 Qwen3：不指定 `--network_train_unet_only`，且不使用 `--cache_text_encoder_outputs`

可通过 `--text_encoder_lr` 为 Qwen3 指定独立的学习率。未指定时使用默认的 `--learning_rate`。

注意：使用 `--cache_text_encoder_outputs` 时，文本编码器输出会被预计算且文本编码器会从 GPU 释放，因此无法训练文本编码器 LoRA。

## 8. 其他训练选项

- **`--loss_type`**：训练损失函数，默认 `l2`。
  - `l1`：L1 损失
  - `l2`：L2 损失（均方误差）
  - `huber`：Huber 损失
  - `smooth_l1`：Smooth L1 损失

- **`--huber_schedule`**、**`--huber_c`**、**`--huber_scale`**：`--loss_type` 为 `huber` 或 `smooth_l1` 时的参数。

- **`--ip_noise_gamma`**、**`--ip_noise_gamma_random_strength`**：Input Perturbation 噪声 gamma 值。

- **`--fused_backward_pass`**：将反向传播和优化器步骤融合以降低显存占用。仅适用于 Adafactor。详见 [`sdxl_train_network.py` 指南](sdxl_train_network.md)。

- **`--weighting_scheme`**、**`--logit_mean`**、**`--logit_std`**、**`--mode_scale`**：时间步损失加权选项。详见 [`sd3_train_network.md` 指南](sd3_train_network.md)。

## 9. 相关工具

### `networks/anima_extract_lora.py`

从两个 Anima 模型的差分中近似提取 LoRA 的脚本。默认提取 DiT 部分；如果同时指定 `--qwen3_org` 和 `--qwen3_tuned`，还可以一起提取 Qwen3 文本编码器的 LoRA。示例：

```bash
python networks/anima_extract_lora.py \
  --model_org path/to/anima_base.safetensors \
  --model_tuned path/to/anima_tuned.safetensors \
  --save_to path/to/output_anima_lora.safetensors \
  --dim 16 \
  --device cuda
```

如果还需要提取 LLM Adapter 的 LoRA，可以加上：

```bash
--train_llm_adapter
```

### `networks/anima_merge_lora.py`

将一个或多个 Anima LoRA 合并进 Anima DiT 模型并保存为新的 `.safetensors`。示例：

```bash
python networks/anima_merge_lora.py \
  --dit path/to/anima_base.safetensors \
  --models path/to/anima_lora.safetensors \
  --ratios 1.0 \
  --save_to path/to/anima_merged.safetensors \
  --save_precision bf16
```

多个 LoRA 可以按顺序传入 `--models` 和对应的 `--ratios`。如果 LoRA 文件中包含 Qwen3 文本编码器权重，脚本会跳过这些权重，只把 DiT 部分合并进 Anima DiT。

### `networks/convert_anima_lora_to_comfy.py`

将 LoRA 模型转换为 ComfyUI 兼容格式的脚本。ComfyUI 不直接支持 sd-scripts 格式的 Qwen3 LoRA，因此需要转换（仅 DiT 的 LoRA 可能不需要转换）。转换命令：

```bash
python networks/convert_anima_lora_to_comfy.py path/to/source.safetensors path/to/destination.safetensors
```

使用 `--reverse` 选项可进行反向转换（ComfyUI 格式转 sd-scripts 格式）。但反向转换仅适用于由此脚本转换的 LoRA，其他训练工具创建的 LoRA 无法转换。

## 10. 其他

### LoRA 模型中保存的元数据

以下元数据会保存在 LoRA 模型文件中：

* `ss_weighting_scheme`
* `ss_logit_mean`
* `ss_logit_std`
* `ss_mode_scale`
* `ss_timestep_sampling`
* `ss_sigmoid_scale`
* `ss_discrete_flow_shift`

`anima_train_network.py` 还支持许多与 `train_network.py` 共通的功能，如训练中的样本图像生成（`--sample_prompts` 等）和详细的优化器配置。这些内容请参考 [`train_network.py` 指南](train_network.md) 或脚本帮助信息（`python anima_train_network.py --help`）。
