# 环境适配 PATCH 记录

本文记录为了让 CRUD-RAG 项目在当前服务器环境（摩尔线程 MUSA GPU + 无 CUDA + 受限网络）下跑起来所做的修改与操作。按时间顺序整理，供以后迁移/复现参考。

## 环境概览

| 项 | 值 |
|---|---|
| 服务器 | 摩尔线程 MTT S4000 ×6（MUSA 架构，无 NVIDIA CUDA） |
| Python | 3.10.12 |
| torch | 2.14.1+cu130（CUDA 版，`torch.cuda.is_available()=False`） |
| transformers | 5.19.0 |
| pymilvus | 2.3.3（客户端，连 localhost:19530） |
| 网络 | 无法直连 docker.io / huggingface.co；可直连 github.com（慢）、dl.minio.org.cn（快）；走 mihomo 代理(127.0.0.1:7897)可访问 docker.io / huggingface |

---

## 1. 依赖补充

`requirements.txt` 末尾新增：

```
einops
```

说明：
- 第一代 Qwen（`Qwen-7B-Chat`）的 `modeling_qwen.py` 使用 `einops.rearrange`，缺了它模型加载失败。

曾尝试加入 `transformers_stream_generator`，但因其与 transformers 5.x 不兼容（见第 3 条），最终**移除**。

---

## 2. 依赖移除

从 `requirements.txt` 移除了 `transformers_stream_generator`。

原因：
- 该包 `from transformers import DisjunctiveConstraint`，而 `DisjunctiveConstraint` 在 transformers 4.46 已被删除，当前 5.19 无法导入，导致 import 报错。

---

## 3. 注释掉 Qwen 模型文件中的流式依赖 import

**文件**：
`~/.cache/huggingface/hub/models--Qwen--Qwen-7B-Chat/snapshots/<hash>/modeling_qwen.py`

**修改**：注释掉 `stream_chat` 方法内的这一行：

```python
# from transformers_stream_generator.main import NewGenerationMixin, StreamGenerationConfig
```

**原因**：
1. transformers 5.x 加载远程模型代码时会静态扫描所有 import（`check_imports`），检测到缺失的 `transformers_stream_generator` 会抛 `ImportError`，导致模型无法加载。
2. 该包依赖的 `DisjunctiveConstraint` 在 transformers 5.19 中已删除，即使装上也无法 import。
3. 本项目 `Qwen_7B_Chat.request()` 使用**非流式** `model.generate()`，不调用 `stream_chat`，因此该流式专用依赖完全用不到。

> ⚠️ 注意：改的是 HF 缓存文件。若以后 `~/.cache/huggingface` 被清除重新下载，此改动会丢失，需重新注释。更持久的做法是下载模型到本地目录、改本地文件，并在 `config.py` 中指向本地路径。

---

## 4. GPT / LLM 相关配置

- `src/configs/config.py` 中设置 `GPT_api_key = 'padding_key'`（非空占位），使 `src/llms/__init__.py` 走 `from .api_model import GPT`，保证 `src.llms.GPT` 可 import，从而让 `src/metric/quest_eval.py`（`class QuestEval(GPT)`）及各 task 的 import 链不断。
- `quick_start.py` 中：注释掉评测循环（tasks/datasets/evaluator 部分），只保留「加载模型 + 建索引」逻辑，即当前阶段只用于建库。

---

## 5. 代理支持

`quick_start.py` 新增 `--proxy_on` 开关，用于在代码内设置代理（供下载模型使用）：

```python
parser.add_argument('--proxy_on', action='store_true', help="Whether to use proxy")
...
if args.proxy_on:
    import os
    os.environ["https_proxy"] = "http://127.0.0.1:7897"
    os.environ["http_proxy"] = "http://127.0.0.1:7897"
```

---

## 6. Milvus 部署（Docker Standalone，minio 用二进制替代）

- Milvus 采用 Docker 部署：`milvusdb/milvus:v2.3.3` + `quay.io/coreos/etcd:v3.5.5` 两个容器。
- 因 `minio/minio` 官方镜像已无法匿名拉取，改用 **minio 二进制**（`docker-images/minio/minio`，从 dl.minio.org.cn 直连下载）在宿主机直接运行，Milvus 通过 `host.docker.internal:9000` 连接。
- 编排与启动见 `docker-images/docker-compose.yml`（参考）与根目录 `start_service.sh`（实际启动脚本，用 `docker run`，因本机无 docker compose 插件）。
- 相关工具（crane、mihomo、clash 配置）放在 `third_party_tools/`。

---

## 7. 数据集与建库

- 检索文档库路径为 `data/80000_docs`（默认 `data/tmp` 不存在，必须显式指定）。
- 建库参数（collection 名编码了配方）：
  ```
  --docs_path data/80000_docs
  --embedding_name BAAI/bge-base-zh-v1.5
  --embedding_dim 768
  --chunk_size 128
  --chunk_overlap 0
  --collection_name docs_80k_chunksize_128_overlap_0_bge
  --construct_index
  ```

---

## 待解决 / 已知障碍

- **CUDA**：`Qwen_7B_Chat.request()` 使用 `.cuda()`，而本机是 MUSA 架构、无 CUDA。若需真正跑模型推理，需安装摩尔线程的 `torch_musa` 并适配，或改代码用 CPU。
- 建库阶段不涉及模型推理，因此上述 CUDA 障碍不影响「只建库」。
