import os
from dotenv import load_dotenv

load_dotenv()  # loads .env if present

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
TOP_K_DEFAULT = 5
