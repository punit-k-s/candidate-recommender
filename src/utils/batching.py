# src/utils/batching.py
from typing import List, Tuple, Callable, Optional
import numpy as np
from .tokens import count_tokens, truncate_to_tokens

def _flush_batch(
    backend,
    batch_texts: List[str],
    verbose: bool,
    progress_cb: Optional[Callable[[int], None]] = None,
    sent_so_far: int = 0,
):
    if not batch_texts:
        return None
    if verbose:
        print(f"[batch] Sending batch with {len(batch_texts)} texts")
    vecs = backend.embed(batch_texts)  # expected (len(batch_texts), D)
    if progress_cb:
        progress_cb(sent_so_far + len(batch_texts))
    return np.asarray(vecs, dtype=np.float32)

def embed_in_batches(
    backend,
    texts: List[str],
    *,
    model: str = "text-embedding-3-small",
    max_text_tokens: int = 8191,      # per-text cap
    max_batch_items: int = 128,        # per-request item cap
    max_batch_tokens: int = 60000,     # per-request total tokens cap
    truncate_long: bool = True,
    verbose: bool = True,
    progress_cb: Optional[Callable[[int], None]] = None,  # gets cumulative count
) -> np.ndarray:
    """
    - Enforces per-text token cap (truncate or error).
    - Groups texts into batches under BOTH:
        * max_batch_items
        * max_batch_tokens (sum of token counts per batch)
    - Returns (N, D) embeddings.
    """
    safe_texts: List[str] = []
    token_counts: List[int] = []
    overs: List[Tuple[int, int]] = []

    # 1) Per-text guard
    for idx, t in enumerate(texts):
        n_tok = count_tokens(t, model)
        if n_tok > max_text_tokens:
            overs.append((idx, n_tok))
            if truncate_long:
                t = truncate_to_tokens(t, max_text_tokens, model)
                n_tok = count_tokens(t, model)
            else:
                raise ValueError(f"text #{idx} exceeds {max_text_tokens} tokens (~{n_tok})")
        safe_texts.append(t)
        token_counts.append(n_tok)

    if verbose and overs:
        msg = ", ".join([f"#{i} ~{n}" for (i, n) in overs])
        print(f"[batch] Truncated texts over per-text limit: {msg}")

    # 2) Token-aware packing
    batches: List[np.ndarray] = []
    cur_texts: List[str] = []
    cur_tok_sum = 0
    sent_so_far = 0

    for t, n_tok in zip(safe_texts, token_counts):
        would_exceed_items = len(cur_texts) >= max_batch_items
        would_exceed_tokens = (cur_tok_sum + n_tok) > max_batch_tokens

        if cur_texts and (would_exceed_items or would_exceed_tokens):
            arr = _flush_batch(backend, cur_texts, verbose, progress_cb, sent_so_far)
            if arr is not None:
                batches.append(arr)
                sent_so_far += len(cur_texts)
            cur_texts = []
            cur_tok_sum = 0

        cur_texts.append(t)
        cur_tok_sum += n_tok

    # flush last
    if cur_texts:
        arr = _flush_batch(backend, cur_texts, verbose, progress_cb, sent_so_far)
        if arr is not None:
            batches.append(arr)

    return np.vstack(batches) if batches else np.empty((0, 0), dtype=np.float32)
