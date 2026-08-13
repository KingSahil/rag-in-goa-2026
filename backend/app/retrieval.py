import json
from collections import defaultdict
from pathlib import Path
from typing import Dict

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

class Shard:
    def __init__(self, index, records):
        self.index = index
        self.records = records
        self.bm25 = BM25Okapi([r["text"].split() for r in records])

class Retriever:
    def __init__(self, index_dir: str, embedder, top_k_dense=8, top_k_bm25=8):
        self.index_dir = Path(index_dir)
        self.embedder = embedder
        self.top_k_dense = top_k_dense
        self.top_k_bm25 = top_k_bm25
        self.shards: Dict[str, Shard] = {}
        self._load()

    def _load(self):
        for faiss_path in self.index_dir.glob("*.faiss"):
            lang = faiss_path.stem
            meta = self.index_dir / f"{lang}.jsonl"
            if not meta.exists():
                continue
            records = []
            with open(meta, "r", encoding="utf-8") as f:
                for line in f:
                    records.append(json.loads(line))
            self.shards[lang] = Shard(faiss.read_index(str(faiss_path)), records)

    def retrieve(self, query: str, language: str | None = None, final_k=5):
        if language and "-" in language:
            language = language.split("-")[0]
        shard_keys = [language] if language in self.shards else list(self.shards)
        q = self.embedder.encode([f"query: {query}"], normalize_embeddings=True)
        q = np.asarray(q, dtype="float32")

        merged = []
        for lang in shard_keys:
            shard = self.shards[lang]
            dense_d, dense_i = shard.index.search(q, self.top_k_dense)

            fused = defaultdict(float)
            meta = {}
            for rank, (idx, score) in enumerate(zip(dense_i[0], dense_d[0]), 1):
                idx = int(idx)
                if idx < 0:
                    continue
                fused[idx] += 1.0 / (60 + rank)
                meta.setdefault(idx, {})["dense_score"] = float(score)

            bm25_scores = shard.bm25.get_scores(query.split())
            bm_idx = np.argsort(-bm25_scores)[:self.top_k_bm25]
            for rank, idx in enumerate(bm_idx, 1):
                idx = int(idx)
                fused[idx] += 1.0 / (60 + rank)
                meta.setdefault(idx, {})["bm25_score"] = float(bm25_scores[idx])

            for idx, rrf_score in sorted(fused.items(), key=lambda x: -x[1])[: max(final_k * 2, 8)]:
                r = shard.records[idx]
                merged.append({
                    **r,
                    "score": float(rrf_score),
                    "dense_score": float(meta.get(idx, {}).get("dense_score", 0.0)),
                    "bm25_score": float(meta.get(idx, {}).get("bm25_score", 0.0)),
                })

        merged.sort(key=lambda x: (-x["score"], -x.get("dense_score", 0.0)))
        return merged[:final_k]
