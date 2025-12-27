import os
import hashlib
from dotenv import load_dotenv
from timescale_vector import client as ts_client

load_dotenv()

# Determine whether to use a real OpenAI client or a local mock fallback
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USE_MOCK_OPENAI = not OPENAI_API_KEY or OPENAI_API_KEY == "DUMMY"

if not USE_MOCK_OPENAI:
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    model = "text-embedding-3-small"
else:
    client = None
    model = None

# Initialize vector client if configured; allow it to be None in dev
TIMESCALE_SERVICE_URL = os.getenv("TIMESCALE_SERVICE_URL")
if TIMESCALE_SERVICE_URL:
    vec_client = ts_client.Sync(TIMESCALE_SERVICE_URL, "embeddings", 1536)
else:
    vec_client = None


def _mock_embedding(text, dim=1536):
    # deterministic pseudo-embedding based on SHA-256
    h = hashlib.sha256(text.encode("utf-8")).digest()
    vals = [b / 255.0 for b in h]
    # repeat/trim to requested dim
    rep = (dim + len(vals) - 1) // len(vals)
    emb = (vals * rep)[:dim]
    return [float(x) for x in emb]


def get_embeddings(text):
    if USE_MOCK_OPENAI:
        return _mock_embedding(text)

    embedding = (
        client.embeddings.create(
            input=[text],
            model=model,
        )
        .data[0]
        .embedding
    )
    return embedding


def get_similar_embeddings(text, limit=5):
    # If there is no vector DB available, return empty context list
    if not vec_client:
        return []

    embedding = get_embeddings(text)
    search_args = {"limit": limit}
    results = vec_client.search(embedding, **search_args)
    return [result[2] for result in results]
