import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from app.chunking import sentence_chunks, sliding_chunks, semantic_chunks

def rrf(rank_lists, k=60):
    scores = defaultdict(float)
    for rows in rank_lists:
        for rank, rid in enumerate(rows, start=1):
            scores[rid] += 1.0 / (k + rank)
    return scores

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--languages", default="hi,pa,en")
    ap.add_argument("--rows-per-language", type=int, default=100000)
    ap.add_argument("--out", default="../indexes")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    languages = [x.strip() for x in args.languages.split(",") if x.strip()]

    model = SentenceTransformer(os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small"))

    for lang in languages:
        source_config = lang if lang != "en" else "hi"
        ds = load_dataset("ai4bharat/MSMARCO-XI", source_config, split="train", streaming=True)
        store = []
        seen = set()

        for row_i, row in enumerate(ds):
            if row_i >= args.rows_per_language:
                break

            passages = row.get("passages", {})
            translated = passages.get("English_passages", []) if lang == "en" else passages.get("Translated_passages", [])
            selected = passages.get("is_selected", []) if isinstance(passages, dict) else []

            for p_i, passage in enumerate(translated):
                if not passage:
                    continue
                key = (lang, " ".join(passage.split()).lower())
                if key in seen:
                    continue
                seen.add(key)

                parent_id = f"{lang}:{row.get('query_id', row_i)}:{p_i}"
                store.append({
                    "id": parent_id,
                    "language": lang,
                    "query_id": str(row.get("query_id", row_i)),
                    "selected": int(selected[p_i]) if p_i < len(selected) else 0,
                    "text": " ".join(passage.split()),
                })

        # Build three representations.
        records = []
        for parent in store:
            base = parent["text"]
            for c in sentence_chunks(base):
                records.append({**parent, "strategy": "sentence", "text": c})
            for c in sliding_chunks(base):
                records.append({**parent, "strategy": "sliding", "text": c})
            for c in semantic_chunks(base, model):
                records.append({**parent, "strategy": "semantic", "text": c})

        texts = [f"passage: {r['text']}" for r in records]
        emb = model.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=True)
        emb = np.asarray(emb, dtype="float32")

        # HNSW is simple and fast for a challenge-scale shard. For very large
        # final indexes, switch to IVF-PQ in the same metadata format.
        index = faiss.IndexHNSWFlat(emb.shape[1], 32)
        index.hnsw.efConstruction = 80
        index.hnsw.efSearch = 32
        index.add(emb)

        faiss.write_index(index, str(out / f"{lang}.faiss"))
        with open(out / f"{lang}.jsonl", "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"{lang}: parents={len(store)}, chunks={len(records)}")

if __name__ == "__main__":
    main()
