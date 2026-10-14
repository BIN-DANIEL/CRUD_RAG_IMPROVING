# third_party_tools

本目录存放项目运行所需的第三方独立工具（无需 pip 安装、直接执行的二进制）。

## 目录内容

| 文件 | 版本 | 用途 |
|---|---|---|
| `crane` | v0.22.1 | 从 Docker registry 下载镜像并导出为 tar（绕开无法直连 docker.io 的问题） |
| `mihomo` | v1.19.32 | 代理客户端（Mihomo/Meta 内核），提供 HTTP 代理，用于访问被墙的镜像源 |
| `clash/` | — | mihomo 的配置目录（含 `config.yaml` 节点信息、`cache.db`、`clash.log`） |

## 为什么需要这两个工具

服务器（摩尔线程 GPU 机器）的网络环境存在以下限制：

- **无法直连** `docker.io` / `quay.io`（拉 Docker 镜像会超时）
- **无法直连** `huggingface.co`（下载模型会超时）
- **可以直连** `github.com`（但速度较慢）
- 服务器没有 root/sudo 免密权限，无法给 Docker daemon 配代理

因此用这两个工具绕过限制：

- `mihomo` 提供本地 HTTP 代理（默认 `127.0.0.1:7897`），走代理可访问 docker.io 和 huggingface.co
- `crane` 是单二进制、无需 sudo 的镜像下载器，读 `HTTPS_PROXY` 环境变量，走 mihomo 代理下载镜像为 tar，再用 `docker load` 加载

## mihomo 使用说明

**配置文件位置**：`third_party_tools/clash/config.yaml`（含节点信息和 `mixed-port: 7897`）

启动方式（在本目录 `third_party_tools/` 下运行）：

```bash
./mihomo -d ./clash
```

验证代理是否生效：

```bash
curl -x http://127.0.0.1:7897 -I https://huggingface.co   # 返回 200 即通
```

> 注意：mihomo 正在运行时，移动本二进制文件不影响已运行的进程；但重新启动需用上面的命令。

## crane 使用说明

### 下载镜像为 tar（走代理）

```bash
export HTTPS_PROXY=http://127.0.0.1:7897
export HTTP_PROXY=http://127.0.0.1:7897
./third_party_tools/crane pull milvusdb/milvus:v2.3.3 docker-images/milvus-v2.3.3.tar
```

### 加载 tar 到 Docker

```bash
docker load -i docker-images/milvus-v2.3.3.tar
```

### 项目需要的三个镜像

| 镜像 | 用途 |
|---|---|
| `milvusdb/milvus:v2.3.3` | Milvus 向量数据库核心服务 |
| `quay.io/coreos/etcd:v3.5.5` | Milvus 元数据存储 |
| `minio/minio:RELEASE.2023-03-20T20-16-18Z` | Milvus 对象存储 |

## 相关文件

- `../milvus-docker-compose.yml` — Milvus Standalone 的 Docker 编排文件（引用上述三个镜像）
