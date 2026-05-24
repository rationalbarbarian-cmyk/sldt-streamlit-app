# embedding.py
from sentence_transformers import SentenceTransformer

# Load embedding model once
_model = SentenceTransformer("all-mpnet-base-v2")

def embed(text: str):
    """Return a single embedding for the full text."""
    return _model.encode(text)

def embed_each(tokens: list[str]):
    """Return embeddings for each token."""
    return _model.encode(tokens)

def tokenize(text: str):
    """Simple whitespace tokenizer (can be replaced with spaCy)."""
    return text.split()
