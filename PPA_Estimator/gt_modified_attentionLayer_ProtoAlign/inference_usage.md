# ProtoAlign 推理脚本使用指南

`inference.py` 提供了加载训练好的 `GraphTransformerModel`、读取嵌入/目标 JSON 并批量输出 PPA 预测的工具。本说明覆盖环境准备、输入要求、命令示例与关键参数。

## 准备工作
1. **环境**：安装 `FastPPA_Estimator/std_environment.yml` 中一致的环境配置即可
2. **图嵌入目录**：通常为 `embed_gen/proc/<split>/` 产出的 `*_embedding_*.json`。脚本会顺序读取同名 `*.json`。
3. **目标目录**（可选）：`*_lib_*.json`，其命名需与嵌入文件遵循 `foo_embedding_001.json ↔ foo_lib_001.json` 规则，用于在推理后对比真值并计算指标。
4. **Checkpoint**：训练结束生成的 `checkpoint.pt`（或 `checkpoint_epoch_xxxx.pt`），内含模型权重。
5. **目标归一化统计**（可选）：若训练时对目标做过 normalize/log1p，推理时应提供相同的统计 JSON（例如 `sel_stats2log1p_tgt.json`），并根据需要设置 `--denormalize-output`。

Note: 因为我们此处统一进行了归一化因此目标归一化一定要在推理中启用。

## 快速开始
```bash
python -m FastPPA_Estimator.gt_modified_attentionLayer_ProtoAlign.inference \
  --emb_dir data/proc/test_embeddings \
  --target_dir data/liberty/test_targets \
  --checkpoint runs/protoalign_exp1/checkpoint.pt \
  --target-stats FastPPA_Estimator/gt_modified_attentionLayer_ProtoAlign/sel_stats2log1p_tgt.json \
  --device cuda \
  --output_dir predictions/test_exp1 \
  --denormalize-output \
  --save_metrics
```
- 如果仅需生成预测，可省略 `--target_dir` 与 `--save_metrics`，输出将只包含模型推理结果。
- 纯 CPU 环境下去掉 `--device cuda`；若传入 `cuda`/`mps` 但不可用，脚本会自动回落到 CPU。

## 参数说明
| 参数 | 作用 |
| --- | --- |
| `--emb_dir` | 必选，嵌入 JSON 目录（例如 `proc/<split>`）。|
| `--target_dir` | 与嵌入匹配的目标 JSON 目录；若提供则可输出真值并计算指标。|
| `--checkpoint` | 训练所得权重文件。|
| `--device` | 推理设备，`cpu`/`cuda`/`mps`。|
| `--output_dir` | 保存单个预测、汇总预测及指标的目录，默认 `predictions/`。|
| `--target-stats` | 目标归一化统计 JSON；需要与训练使用的文件一致。|
| `--no-target-normalization` | 有统计文件但想禁用归一化时使用。|
| `--denormalize-output` | 当目标已归一化时，将预测与真值还原到原始量纲。|
| `--save_metrics` | 若提供真值，计算 MSE/MAE/MAPE/STD/R² 并写入 `metrics.json`。|

## 输出内容
- `all_predictions.json`：聚合所有样本的预测，包含泄漏、功耗向量及延迟矩阵；如有真值会附带 `ground_truth` 部分。
- `<sample>_pred.json`：逐样本的推理结果，位于 `output_dir` 中。
- `metrics.json`（仅 `--save_metrics` 且存在真值时）：每个任务的 MSE/MAE/MAPE/误差标准差/R²。
