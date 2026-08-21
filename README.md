# SimpleRAG：精细与快速共存的中文 Hybrid RAG

本项目使用稠密语义召回、中文字符 n-gram 关键词召回和加权 RRF 融合，并按请求选择快速、平衡或精确重排。仓库不包含模型权重，首次运行时模型会下载到本机缓存。

## 1. 三种运行模式

| 模式 | 候选数 | 重排 | 使用场景 |
|---|---:|---|---|
| `fast` | 12 | 无 | 高并发、低延迟 |
| `balanced` | 30 | 轻量重排 | 默认企业问答 |
| `precise` | 60 | Cross-Encoder（配置后） | 合同、制度、复杂问题 |

## 2. 在 Windows 台式机安装

建议使用 Python 3.11 或 3.12。打开 PowerShell：

```powershell
git clone https://github.com/Suhang656/simpleRAG.git
cd simpleRAG
python -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

默认依赖兼容 CPU。如果使用 RTX 3090，请按 [PyTorch 官网](https://pytorch.org/get-started/locally/) 给出的命令安装与驱动匹配的 CUDA 版 `torch`，随后检查：

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

输出应包含 `True` 和显卡名称。

## 3. API Key 放在哪里

推荐使用项目级 `.env`，它和 `simpleRAG.py` 位于同一目录：

```text
simpleRAG/
├── .env              ← 真实 API Key 放这里，不上传 GitHub
├── .env.example      ← 可公开的配置模板
├── simpleRAG.py
└── requirements.txt
```

在项目目录执行：

```powershell
Copy-Item .env.example .env
notepad .env
```

阿里云百炼公共端点示例：

```dotenv
RAG_API_KEY=sk-替换成真实密钥
RAG_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
RAG_LLM_MODEL=qwen-plus
RAG_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
# RAG_RERANKER_MODEL=BAAI/bge-reranker-v2-m3
LOG_LEVEL=INFO
```

注意事项：

- 不要给值添加中文引号，不要把真实 Key 写入 `.env.example`。
- `.env` 已列入 `.gitignore`，不会被正常 Git 提交上传。
- 百炼的 Key、地域和 Base URL 必须匹配；业务空间应使用控制台给出的 Workspace URL。
- 使用其他 OpenAI 兼容服务时，只需替换 Key、Base URL 和模型名称。

程序自动读取 `.env`。操作系统环境变量优先于 `.env`，生产服务器仍可使用密钥管理系统注入。

### 可选：Windows 用户级全局变量

不想使用 `.env` 时可执行：

```powershell
[Environment]::SetEnvironmentVariable('RAG_API_KEY', '你的真实Key', 'User')
[Environment]::SetEnvironmentVariable('RAG_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1', 'User')
[Environment]::SetEnvironmentVariable('RAG_LLM_MODEL', 'qwen-plus', 'User')
```

设置后关闭并重新打开 PowerShell。项目级 `.env` 更安全、容易迁移；用户级变量会影响该 Windows 账号下的所有程序。

## 4. 检查配置

以下命令不加载模型，也不发起 API 请求：

```powershell
python simpleRAG.py --check-config
```

程序只显示 Key 前四位，不打印完整密钥。确认 CUDA、Base URL 和模型名称正确。

## 5. 准备知识库文档

当前版本读取 UTF-8 纯文本。建议把企业文档放在仓库之外，例如：

```text
D:\RAG_DATA\公司制度.txt
```

首次验证可新建一个测试文本，写入几段能够人工核对的事实内容。

## 6. 启动问答

```powershell
# 快速模式
python simpleRAG.py "D:\RAG_DATA\公司制度.txt" --mode fast

# 默认平衡模式
python simpleRAG.py "D:\RAG_DATA\公司制度.txt" --mode balanced
```

精确模式需要先在 `.env` 取消 Reranker 的注释：

```dotenv
RAG_RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

然后运行：

```powershell
python simpleRAG.py "D:\RAG_DATA\公司制度.txt" --mode precise
```

首次运行会下载 Embedding/Reranker 模型，需要能够访问 Hugging Face。看到 `知识库已就绪` 后输入问题，输入 `quit` 退出。

如果没有配置 `RAG_API_KEY`，程序进入离线检索调试模式：仍执行召回和重排，但直接返回带编号的知识片段，不调用 LLM。

## 7. RTX 3090 推荐配置

先用轻量组合验证：

```dotenv
RAG_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
# 暂不启用 Reranker
```

流程稳定后切换高质量组合：

```dotenv
RAG_EMBEDDING_MODEL=BAAI/bge-m3
RAG_RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

当前 LLM 通过 OpenAI 兼容 API 调用。如果以后使用 Ollama、vLLM 等 3090 本地服务，把 `RAG_BASE_URL` 改为本地 OpenAI 兼容地址，并填写对应模型名即可。

## 8. 测试与常见问题

```powershell
python -m unittest -v
```

- `CUDA: 不可用`：安装了 CPU 版 PyTorch，或 NVIDIA 驱动/CUDA 版本不匹配。
- `401 Unauthorized`：API Key 错误、失效，或 Key 与地域不匹配。
- `model not found`：`RAG_LLM_MODEL` 不是当前账号可调用的模型代码。
- 无法下载模型：检查 Hugging Face 网络，或下载模型后将变量设置为本地目录。
- 只返回知识片段：没有读取到 `RAG_API_KEY`，运行 `--check-config` 检查。

## 9. 模型选型

| 层次 | 快速/低成本 | 精细/高质量 |
|---|---|---|
| Embedding | `BAAI/bge-small-zh-v1.5` | `BAAI/bge-m3` |
| Reranker | 不启用或 `bge-reranker-base` | `BAAI/bge-reranker-v2-m3` |
| LLM | Qwen 7B/14B 级 | API 大模型或多卡部署 |
| 向量库 | FAISS 单机 | Milvus/Qdrant/pgvector |

企业选型应以自有黄金问答集评估 Recall@20、MRR@10、引用正确率、拒答准确率、P95 延迟和单问成本。

## 安全原则

- 永远不要提交 `.env`、API Key、企业知识库或生成索引。
- 生产环境使用密钥管理系统、租户/ACL 过滤、审计、限流、超时和熔断。
- 文档内容和用户问题都视为不可信输入，防范提示注入与敏感信息泄露。
