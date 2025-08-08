
def count_tokens(text: str, model: str = "text-embedding-3-small") -> int:
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)  # rough fallback


def truncate_to_tokens(text: str, max_tokens: int, model: str = "text-embedding-3-small") -> str:
    """
    Truncate text to <= max_tokens. Uses tiktoken if available; otherwise char-based fallback.
    """
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model(model)
        toks = enc.encode(text)
        if len(toks) <= max_tokens:
            return text
        return enc.decode(toks[:max_tokens])
    except Exception:
        # ~4 chars per token fallback
        approx = max_tokens * 4
        return text[:approx]
