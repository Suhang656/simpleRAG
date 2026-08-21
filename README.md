# 精细与快速共存的 Hybrid RAG

系统采用稠密语义召回 + 中文字符 n-gram 关键词召回，以加权 RRF 融合；再按场景选择跳过重排、轻量重排或 Cross-Encoder 精排。回答携带来源编号，低相关问题直接拒答。仓库不包含模型权重，首次运行时按配置从 Hugging Face 下载到本机缓存。

| 模式 | 候选数 | 重排 | 场景 |
|---|---:|---|---|
| `fast` | 12 | 无 | 高并发客服、联想提示 |
| `balanced` | 30 | 轻量融合分数 | 默认企业问答 |
| `precise` | 60 | Cross-Encoder（配置后） | 合同、制度、复杂问题 |

## 模型比较与推荐

| 层次 | 快速/低成本 | 精细/高质量 | 结论 |
|---|---|---|---|
| Embedding | `BAAI/bge-small-zh-v1.5` | `BAAI/bge-m3` | 当前 MiniLM 偏英文，仅适合演示；中文生产优先 bge-m3 |
| Reranker | `BAAI/bge-reranker-base` | `BAAI/bge-reranker-v2-m3` | 只在 precise 路径启用，避免拖慢所有请求 |
| LLM | Qwen 7B/14B 级 | Qwen 32B/72B 级 | 按问题复杂度路由，大模型不替代检索 |
| 向量库 | FAISS 单机 | Milvus/Qdrant/pgvector | 多租户、在线增量和横向扩展时迁移 |

最终模型须用企业黄金问答集，以 Recall@20、MRR@10、Faithfulness、拒答准确率、P95 延迟和单问成本联合选型。

## 运行

```powershell
pip install -r requirements.txt
$env:RAG_API_KEY = "..."
$env:RAG_BASE_URL = "https://your-endpoint/v1"
$env:RAG_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5" # 先验证低资源方案
# 精确模式再设置：$env:RAG_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
python simpleRAG.py 三体.txt --mode balanced
```

未配置 API Key 时返回带编号的检索片段。密钥必须从企业密钥管理系统注入，禁止进入源码。

台式机首次测试建议先使用默认的 `bge-small-zh-v1.5 + balanced`，确认链路后再切换 `bge-m3 + precise`。如果使用 NVIDIA GPU，请根据显卡驱动在 PyTorch 官网选择对应 CUDA 版本安装 `torch`；`requirements.txt` 默认兼容 CPU 环境。

## 企业落地补全项

1. 入库保留标题、页码、权限、部门、版本和生效日期，按结构分块；建立增量索引、版本、灰度及回滚。
2. 检索前强制租户与 ACL 过滤；向量库生产化后提供副本、备份和容量治理。
3. API 加入 SSO/鉴权、限流、缓存、流式输出、超时、熔断；三类模型独立扩缩容。
4. 监控召回率、引用正确率、拒答率、P50/P95、token 成本和漂移，建立用户反馈与评测闭环。
5. 做入库脱敏、审计、数据保留和提示注入防护；索引内容一律视为不可信输入。
