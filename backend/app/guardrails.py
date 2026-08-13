import re

UNSAFE = re.compile(
    r"\b(?:kill|murder|rape|suicide|self[- ]harm|bomb|explosive|porn|child sexual)\b",
    re.I,
)

def token_set(text: str):
    return {t.lower() for t in re.findall(r"\w+", text, flags=re.UNICODE) if len(t) >= 4}

def grounding_score(answer: str, contexts: list[str]) -> float:
    a = token_set(answer)
    if not a:
        return 0.0
    c = token_set(" ".join(contexts))
    return len(a & c) / max(1, len(a))

def check_input(query: str):
    if not query.strip():
        return False, "empty_query"
    if UNSAFE.search(query):
        return False, "unsafe_or_inappropriate_query"
    return True, None

def should_refuse(results: list[dict], min_dense=0.28):
    if not results:
        return True, "no_retrieval_results"
    top = results[0]
    # RRF alone is not a confidence score, so combine it with dense evidence.
    # For lexical-only hits, require a second source to reduce accidental answers.
    if top.get("dense_score", 0.0) >= min_dense:
        return False, None
    if top.get("bm25_score", 0.0) > 0 and len(results) >= 2:
        return False, None
    return True, "low_retrieval_confidence"
