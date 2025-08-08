'''
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def rank_by_cosine(query_vec: np.ndarray, doc_vecs: np.ndarray, names, top_k: int):
    sims = cosine_similarity(query_vec, doc_vecs)[0]
    df = pd.DataFrame({"Candidate": names, "Similarity": sims})
    df = df.sort_values("Similarity", ascending=False).reset_index(drop=True)
    # Normalize 0–1 for display
    mn, mx = df["Similarity"].min(), df["Similarity"].max()
    df["Similarity"] = (df["Similarity"] - mn) / (mx - mn + 1e-8)
    return df.head(top_k)
'''



# src/ranking.py
import numpy as np
import pandas as pd

def l2norm(x, eps=1e-10):
    norms = np.linalg.norm(x, axis=1, keepdims=True) + eps
    return x / norms

def rank_by_cosine(q_vec, doc_vecs, names, top_k=5, normalize=False):
    # L2-normalize
    qn = l2norm(q_vec)
    dn = l2norm(doc_vecs)

    # cosine with the single JD vector
    scores = (dn @ qn.T).ravel()  # shape (R,)

    if normalize:
        maxv = np.max(scores) if scores.size else 1.0
        if maxv > 0:
            scores = scores / maxv

    order = np.argsort(-scores)[:top_k]
    df = pd.DataFrame({
        "Candidate": [names[i] for i in order],
        "Similarity": [float(scores[i]) for i in order],
    })
    return df
