import os

def extract_name(filename: str, fallback_idx: int) -> str:
    base = os.path.basename(filename)
    stem = os.path.splitext(base)[0]
    return stem or f"Pasted_Resume_{fallback_idx}"
