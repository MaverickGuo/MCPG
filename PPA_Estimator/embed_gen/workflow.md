# FastPPA `embed_gen` 集成流水线使用说明

`integrated_pipeline.py` 会依次调用 `process_raw.py`、`path_extractor.py` 以及 `generate_graph_embeddings_pe.py`，完成原始版图解析、拓扑提取与图嵌入生成三步处理，适合批量构建 PPA 评估所需的特征文件。

## 前置准备
- Python 3.8 及以上，建议使用虚拟环境确保依赖一致。
- 确保三个脚本位于 `FastPPA_Estimator/embed_gen/` 中，或通过参数指定自定义路径。
- 原始输入文件位于 `embed_gen/raw/`，处理结果建议输出到 `embed_gen/proc/`，方便后续复用。
- 至少需要以下四类输入：
  - `*.net`：电路网表。
  - `*_fets_*.json`：晶体管信息。
  - `*_layout_*.json`：布局信息。
  - `*_routing_*.json`：布线信息。

## 快速开始
```bash
python /Users/jonathan/Workspace/StdPPAEstimation/FastPPA_Estimator/embed_gen/integrated_pipeline.py \
  --net /Users/jonathan/Workspace/StdPPAEstimation/FastPPA_Estimator/embed_gen/raw/A2O1A1Ixp5_ASAP7_6t_L.net \
  --fets /Users/jonathan/Workspace/StdPPAEstimation/FastPPA_Estimator/embed_gen/raw/A2O1A1Ixp5_ASAP7_6t_L_fets_3.json \
  --layout /Users/jonathan/Workspace/StdPPAEstimation/FastPPA_Estimator/embed_gen/raw/A2O1A1Ixp5_ASAP7_6t_L_layout_3.json \
  --routing /Users/jonathan/Workspace/StdPPAEstimation/FastPPA_Estimator/embed_gen/raw/A2O1A1Ixp5_ASAP7_6t_L_routing_3.json \
  --processed-output /Users/jonathan/Workspace/StdPPAEstimation/FastPPA_Estimator/embed_gen/proc/A2O1A1Ixp5_ASAP7_6t_L_processed_3.json \
  --topology-output /Users/jonathan/Workspace/StdPPAEstimation/FastPPA_Estimator/embed_gen/proc/A2O1A1Ixp5_ASAP7_6t_L_topology_3.json \
  --embeddings-output /Users/jonathan/Workspace/StdPPAEstimation/FastPPA_Estimator/embed_gen/proc/A2O1A1Ixp5_ASAP7_6t_L_embedding_3.json
```
若追加 `--report-timing`，执行完成后会打印三个阶段的耗时统计，方便评估瓶颈；默认不输出总结。

## 参数说明
| 参数 | 作用 |
| --- | --- |
| `--net` | 指定 `.net` 网表文件。|
| `--fets` | 指定 FET JSON 文件或所在目录。|
| `--layout` | 指定布局 JSON 文件或所在目录。|
| `--routing` | 指定布线 JSON 文件或所在目录。|
| `--processed-output` | `process_raw.py` 产出的 enriched 布局 JSON 路径。|
| `--topology-output` | `path_extractor.py` 生成的拓扑 JSON 路径。|
| `--embeddings-output` | `generate_graph_embeddings_pe.py` 生成的嵌入 JSON 路径。|
| `--process-raw-script` | 如需自定义 `process_raw.py` 路径，使用此参数。|
| `--path-extractor-script` | 自定义 `path_extractor.py` 路径。|
| `--embedding-script` | 自定义 `generate_graph_embeddings_pe.py` 路径。|
| `--report-timing` | 打印每个阶段的耗时及总耗时。|

## 输出文件
- `*_processed_*.json`：带有标准化布局、器件和连线属性，供后续拓扑提取使用。
- `*_topology_*.json`：记录点到点路径、层次关系等拓扑结构，通常是嵌入生成的直接输入。
- `*_embedding_*.json`：最终图嵌入结果，供 PPA 模型或分析脚本读取。




