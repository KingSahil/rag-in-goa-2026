"""
Three chunking strategies.

1) sentence: sentence-aware groups with a soft token target.
2) sliding: overlapping windows for robustness to boundary queries.
3) semantic: sentence groups merged until adjacent sentence embeddings become
   dissimilar enough to indicate a topic boundary.

All methods are offline-only.
"""
import re
from typing import List

SENT_RE = re.compile(r"(?<=[.!?।॥])\s+")

def sentences(text: str) -> List[str]:
    return [s.strip() for s in SENT_RE.split(text.strip()) if s.strip()]

def word_count(s: str) -> int:
    return len(s.split())

def sentence_chunks(text: str, target=120, max_words=180):
    ss = sentences(text)
    out, cur = [], []
    n = 0
    for s in ss:
        w = word_count(s)
        if cur and n + w > max_words:
            out.append(" ".join(cur))
            cur, n = [], 0
        cur.append(s)
        n += w
        if n >= target:
            out.append(" ".join(cur))
            cur, n = [], 0
    if cur:
        out.append(" ".join(cur))
    return out or [text.strip()]

def sliding_chunks(text: str, window=150, overlap=45):
    ws = text.split()
    if len(ws) <= window:
        return [text.strip()]
    out = []
    step = max(1, window - overlap)
    for start in range(0, len(ws), step):
        chunk = " ".join(ws[start:start + window]).strip()
        if chunk:
            out.append(chunk)
        if start + window >= len(ws):
            break
    return out

def semantic_chunks(text: str, embedder, threshold=0.62, max_words=180):
    ss = sentences(text)
    if len(ss) <= 1:
        return [text.strip()]
    vecs = embedder.encode(ss, normalize_embeddings=True)
    groups, cur = [], [ss[0]]
    for i in range(1, len(ss)):
        sim = float(vecs[i-1] @ vecs[i])
        candidate = " ".join(cur + [ss[i]])
        if sim < threshold or word_count(candidate) > max_words:
            groups.append(" ".join(cur))
            cur = [ss[i]]
        else:
            cur.append(ss[i])
    if cur:
        groups.append(" ".join(cur))
    return groups or [text.strip()]
