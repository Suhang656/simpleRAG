"""Enterprise-oriented hybrid RAG: dense + lexical recall and optional reranking."""
from __future__ import annotations

import argparse, hashlib, json, logging, os, pickle, re, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import faiss
import numpy as np
import torch
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

Mode = Literal["fast", "balanced", "precise"]
log = logging.getLogger("rag")


@dataclass(slots=True)
class RAGConfig:
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    reranker_model: str | None = None
    chunk_size: int = 700
    chunk_overlap: int = 100
    batch_size: int = 32
    fast_candidates: int = 12
    balanced_candidates: int = 30
    precise_candidates: int = 60
    final_top_k: int = 5
    min_relevance: float = 0.12
    dense_weight: float = 0.65
    lexical_weight: float = 0.35
    max_context_chars: int = 9000


@dataclass(slots=True)
class Chunk:
    id: str
    text: str
    source: str
    position: int


@dataclass(slots=True)
class Hit:
    chunk: Chunk
    score: float
    dense_score: float = 0.0
    lexical_score: float = 0.0


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class EmbeddingModel:
    """Batched masked-mean pooling; normalized vectors enable cosine search."""
    def __init__(self, name: str, batch_size: int = 32):
        name = str(Path(name).resolve()) if Path(name).exists() else name
        self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(name, trust_remote_code=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> np.ndarray:
        outputs = []
        for i in range(0, len(texts), self.batch_size):
            encoded = self.tokenizer(texts[i:i+self.batch_size], padding=True, truncation=True,
                                     max_length=512, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                hidden = self.model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).expand_as(hidden).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            outputs.append(pooled.cpu().numpy().astype("float32"))
        return np.ascontiguousarray(np.vstack(outputs))


class CrossEncoderReranker:
    def __init__(self, name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(name, trust_remote_code=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        batch = self.tokenizer([query] * len(texts), texts, padding=True, truncation=True,
                               max_length=512, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            logits = self.model(**batch).logits
        return (logits[:, -1] if logits.ndim == 2 else logits.flatten()).float().cpu().numpy()


class SimpleLLM:
    def __init__(self, api_key: str, base_url: str | None, model: str):
        self.client, self.model = OpenAI(api_key=api_key, base_url=base_url, timeout=30, max_retries=2), model

    def generate(self, question: str, context: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model, temperature=0.1, max_tokens=800,
            messages=[
                {"role": "system", "content": "只能依据资料回答；不足时明确拒答，不得编造。关键结论标注资料编号，如[1]。"},
                {"role": "user", "content": f"资料：\n{context}\n\n问题：{question}"},
            ])
        return (response.choices[0].message.content or "").strip()


class HybridIndex:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self.dense = None
        self.lexical = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), sublinear_tf=True)
        self.lexical_matrix = None

    def build(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if not chunks or len(chunks) != len(vectors): raise ValueError("invalid chunks/vectors")
        vectors = np.ascontiguousarray(vectors.astype("float32")); faiss.normalize_L2(vectors)
        self.dense = faiss.IndexFlatIP(vectors.shape[1]); self.dense.add(vectors)
        self.chunks = chunks
        self.lexical_matrix = self.lexical.fit_transform([c.text for c in chunks])

    def search(self, query: str, vector: np.ndarray, count: int, cfg: RAGConfig) -> list[Hit]:
        if self.dense is None: raise RuntimeError("index is not built")
        count = min(count, len(self.chunks)); vector = np.ascontiguousarray(vector.astype("float32")); faiss.normalize_L2(vector)
        ds, di = self.dense.search(vector, count)
        ls = (self.lexical_matrix @ self.lexical.transform([query]).T).toarray().ravel()
        li = np.argsort(-ls)[:count]
        fused, draw, lraw = {}, {}, {}
        for rank, idx in enumerate(di[0]):
            if idx >= 0:
                idx = int(idx); fused[idx] = fused.get(idx, 0) + cfg.dense_weight/(61+rank); draw[idx] = float(ds[0, rank])
        for rank, idx in enumerate(li):
            idx = int(idx); fused[idx] = fused.get(idx, 0) + cfg.lexical_weight/(61+rank); lraw[idx] = float(ls[idx])
        return [Hit(self.chunks[i], s, draw.get(i, 0), lraw.get(i, 0))
                for i, s in sorted(fused.items(), key=lambda x: x[1], reverse=True)]

    def save(self, directory: str | Path, cfg: RAGConfig) -> None:
        target = Path(directory); target.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.dense, str(target/"dense.faiss"))
        with (target/"lexical.pkl").open("wb") as f: pickle.dump((self.lexical, self.lexical_matrix), f)
        (target/"manifest.json").write_text(json.dumps({"version": 1, "config": asdict(cfg),
            "chunks": [asdict(c) for c in self.chunks]}, ensure_ascii=False), encoding="utf-8")

    def load(self, directory: str | Path) -> None:
        source = Path(directory); data = json.loads((source/"manifest.json").read_text(encoding="utf-8"))
        self.chunks = [Chunk(**c) for c in data["chunks"]]; self.dense = faiss.read_index(str(source/"dense.faiss"))
        with (source/"lexical.pkl").open("rb") as f: self.lexical, self.lexical_matrix = pickle.load(f)


def split_text(text: str, size: int, overlap: int) -> list[str]:
    units = [x.strip() for x in re.split(r"(?<=[。！？!?；;])|\n\s*\n", text) if x.strip()]
    result, current = [], ""
    for unit in units:
        if len(current) + len(unit) <= size: current += unit
        else:
            if current: result.append(current)
            current = (current[-overlap:] if overlap else "") + unit
            while len(current) > size:
                result.append(current[:size]); current = current[size-overlap:]
    if current: result.append(current)
    return result


class RAGSystem:
    def __init__(self, api_key: str | None = None, base_url: str | None = None, llm_model_name: str = "qwen-turbo",
                 config: RAGConfig | None = None, embedder: Embedder | None = None, llm: Any | None = None):
        self.config = config or RAGConfig()
        self.embedding_model = embedder or EmbeddingModel(self.config.embedding_model, self.config.batch_size)
        self.vector_db = HybridIndex()
        self.reranker = CrossEncoderReranker(self.config.reranker_model) if self.config.reranker_model else None
        self.llm = llm or (SimpleLLM(api_key, base_url, llm_model_name) if api_key else None)

    def load_and_index(self, source: str, source_type: str = "file", source_name: str | None = None) -> None:
        if source_type == "file":
            path = Path(source); text = path.read_text(encoding="utf-8-sig"); name = source_name or path.name
        elif source_type == "text": text, name = source, source_name or "inline"
        else: raise ValueError("source_type must be file or text")
        texts = split_text(text, self.config.chunk_size, self.config.chunk_overlap)
        chunks = [Chunk(hashlib.sha256(f"{name}:{i}:{t}".encode()).hexdigest()[:16], t, name, i) for i, t in enumerate(texts)]
        self.vector_db.build(chunks, self.embedding_model.embed(texts)); log.info("indexed source=%s chunks=%d", name, len(chunks))

    def retrieve(self, question: str, mode: Mode = "balanced", top_k: int | None = None) -> list[Hit]:
        started = time.perf_counter()
        counts = {"fast": self.config.fast_candidates, "balanced": self.config.balanced_candidates,
                  "precise": self.config.precise_candidates}
        hits = self.vector_db.search(question, self.embedding_model.embed([question]), counts[mode], self.config)
        if mode == "precise" and self.reranker and hits:
            for hit, score in zip(hits, self.reranker.score(question, [h.chunk.text for h in hits])): hit.score = float(score)
            hits.sort(key=lambda h: h.score, reverse=True)
        elif mode != "fast": hits.sort(key=lambda h: .55*h.dense_score + .45*h.lexical_score, reverse=True)
        result = hits[:top_k or self.config.final_top_k]
        log.info("retrieve mode=%s candidates=%d returned=%d latency_ms=%.1f", mode, len(hits), len(result),
                 (time.perf_counter()-started)*1000)
        return result

    def query(self, question: str, top_k: int | None = None, mode: Mode = "balanced") -> str:
        hits = self.retrieve(question, mode, top_k)
        if not hits or max(max(h.dense_score, h.lexical_score) for h in hits) < self.config.min_relevance:
            return "根据现有知识库，无法找到足够可靠的信息来回答该问题。"
        if not self.llm: return "\n\n".join(f"[{i}] {h.chunk.text}" for i, h in enumerate(hits, 1))
        parts, used = [], 0
        for i, hit in enumerate(hits, 1):
            item = f"[{i}] 来源：{hit.chunk.source}，片段：{hit.chunk.position}\n{hit.chunk.text}"
            if used + len(item) > self.config.max_context_chars: break
            parts.append(item); used += len(item)
        return self.llm.generate(question, "\n\n".join(parts))

    def save_index(self, directory: str | Path) -> None: self.vector_db.save(directory, self.config)
    def load_index(self, directory: str | Path) -> None: self.vector_db.load(directory)


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("document", type=Path)
    p.add_argument("--mode", choices=("fast", "balanced", "precise"), default="balanced")
    p.add_argument("--embedding-model", default=os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"))
    p.add_argument("--reranker-model", default=os.getenv("RAG_RERANKER_MODEL")); args = p.parse_args()
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    cfg = RAGConfig(embedding_model=args.embedding_model, reranker_model=args.reranker_model)
    rag = RAGSystem(os.getenv("RAG_API_KEY"), os.getenv("RAG_BASE_URL"), os.getenv("RAG_LLM_MODEL", "qwen-turbo"), cfg)
    rag.load_and_index(str(args.document)); print("知识库已就绪；输入 quit 退出。")
    while (q := input("\n问题：").strip()).lower() != "quit": print(rag.query(q, mode=args.mode))


if __name__ == "__main__": main()
