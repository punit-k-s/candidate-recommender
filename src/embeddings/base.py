from typing import List
import numpy as np

class EmbeddingClient:
    def embed(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError
