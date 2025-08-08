import numpy as np
from typing import List
from openai import OpenAI
from src.config import OPENAI_API_KEY, EMBED_MODEL
from .base import EmbeddingClient

class OpenAIEmbeddingClient(EmbeddingClient):
    def __init__(self, api_key: str = OPENAI_API_KEY, model: str = EMBED_MODEL):
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set. Put it in a .env or export it.")
        self.api_key = api_key            # <-- keep a copy for logging
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed(self, texts: List[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        arr = np.array([d.embedding for d in resp.data], dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-10
        return arr / norms
