# DeepSpeed 异常排查总结

## 现象

- 任务类型：Anima 全参数训练
- 环境：双卡 RTX 4080 SUPER
- 单卡训练：正常
- 双卡训练：
  - 不使用 `DeepSpeed`：正常
  - 使用 `DeepSpeed`：在第一个 step 附近退出

## 典型报错

```text
File ".../deepspeed/runtime/bf16_optimizer.py", line 290, in step
    assert all_groups_norm > 0.
AssertionError
```

补充现象：

- 训练进度通常停在 `steps: 0%`
- 有时会先看到 `epoch is incremented. current_epoch: 0, epoch: 1`
- 不是常见的 `avr_loss=nan`，而是 DeepSpeed 在 optimizer step 时直接断言失败

## 已验证条件

以下条件已经验证过，仍然会在双卡 + DeepSpeed 下报错：

- `--learning_rate=1e-7`
- `batch_size = 1`
- 不使用 `flash attn`
- `--zero_stage=0`

以下条件下训练正常：

- 单卡训练
- 双卡训练但去掉 `--deepspeed`

## 当前结论

这次异常的根因不在：

- 学习率过高
- `flash attn`
- 数据集 bucket 配置
- 单纯的 batch size 过大

更可能是：

- `Anima 全参数训练 + 双卡 + DeepSpeed` 这条路径本身不稳定
- 具体触发点在 `DeepSpeed bf16 optimizer`
- `zero_stage=0` 也会触发，说明不只是 ZeRO 分片问题，而是 DeepSpeed 优化器路径本身就可能不兼容当前配置

## 代码排查结论

结合仓库代码，问题已经可以进一步收敛：

- 实际触发组合不是泛泛的“DeepSpeed 不稳定”，而是：
  - `--deepspeed`
  - `--mixed_precision=bf16`
  - `--optimizer_type=AdamW8bit`
  - `--zero_stage=0`
- 在 `library/deepspeed_utils.py` / `library/train_util.py` 这条接线里，`accelerate` 会把外部传入的 optimizer 原样交给 DeepSpeed。
- 对于外部 optimizer，`accelerate` 还会自动设置 `zero_allow_untested_optimizer = true`，也就是“放行未验证 optimizer”，不会提前阻止这类组合。
- DeepSpeed 在 bf16 模式下会额外包一层 `BF16_Optimizer`；即使 `zero_stage=0`，这条 bf16 优化器路径仍然会生效，不是只有 ZeRO-1/2/3 才会走到这里。
- `AdamW8bit` 来自 `bitsandbytes`，并不是 DeepSpeed 自身支持/验证的标准 optimizer 组合；当前报错正好发生在 `BF16_Optimizer.step()` 对全局梯度范数做断言的位置。

更直接地说：

- 这次最可疑的根因是 `DeepSpeed bf16` 与 `bitsandbytes AdamW8bit` 组合不兼容。
- `--zero_stage=0` 不能绕开这个问题，因为异常点不在 ZeRO 分片，而在 bf16 optimizer 包装层。

## 附带风险

- 仓库当前的 DeepSpeed 接线方式并不是通过 DeepSpeed wrapper 本身执行前向，而是仍然直接调用原始模型做 forward/backward。
- 这不一定是本次断言失败的唯一原因，但会让 DeepSpeed 路径比普通 DDP 更脆弱，更难定位问题。

## 已做处理

- 已在 `library/deepspeed_utils.py` 增加兼容处理：
  - 如果检测到 `DeepSpeed + bf16 + 8bit optimizer`，会自动回退到兼容的非 8bit optimizer。
  - 对当前命令来说，`AdamW8bit` 会自动改成 `AdamW`。
- 已在 `anima_train.py` 中把 DeepSpeed 路径改为“直接 prepare 单模型 `dit`”，不再走多模型 wrapper 的旁路写法。
- 这样修完以后，目标不是继续支持 `AdamW8bit + DeepSpeed`，而是把这条训练链路切回 DeepSpeed 更标准、更可用的执行路径，优先争取 `zero2` 可用。

## 远程配置状态

远程文件：`/root/sd-scripts/train_hires/command.txt`

双卡命令中当时仍包含：

- `--deepspeed`
- `--zero_stage=0`
- `--optimizer_type="AdamW8bit"`
- `--llm_adapter_lr=0`
- `--attn_mode="flash"`

其中最关键的是 `--deepspeed`。根据实测，去掉后双卡即可正常训练。

## 对后续排查的建议

优先级从高到低：

1. 先不要用 `DeepSpeed`
   - 双卡直接走普通 DDP
   - 这是目前唯一已验证稳定的双卡方案

2. 如果必须继续试 `DeepSpeed`
   - 保持其它参数不变，只改一个变量做对照
   - 推荐顺序：
     - 现在代码会自动把 `AdamW8bit` 回退成 `AdamW`
     - 先直接试 `--zero_stage=2`
     - 再试 `mixed_precision=no`
     - 再试 `Adafactor`

3. 不要再优先怀疑学习率
   - 因为 `1e-7` 仍然会触发相同异常
   - 这已经说明问题不是普通的梯度发散

## 当前推荐方案

如果目标是先把训练稳定跑起来：

- 双卡
- 不用 `DeepSpeed`
- 保留 `batch_size = 1`
- 可再加：
  - `--lr_scheduler="constant_with_warmup"`
  - `--lr_warmup_steps=200`

## 一句话结论

这次报错的核心不是 `nan`，而是 `DeepSpeed bf16 optimizer` 在 `bf16 + AdamW8bit` 这条未验证组合下拿到了 `all_groups_norm == 0`，从而在 `step()` 时直接触发断言；当前最稳妥的处理方式是改用 `AdamW` 或直接暂时不要在这套 Anima 全参数配置上使用 `DeepSpeed`。
