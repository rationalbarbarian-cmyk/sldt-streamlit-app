# axes.py
import numpy as np
from embedding import embed_each

AXES = ["Valence", "Power", "Agency", "EvaluativeForce"]

def normalize(v):
    return v / np.linalg.norm(v)

def build_axis_vectors(SEED_PHRASES):
    """Return dictionary of normalized semantic direction vectors."""
    axis_vectors = {}

    for axis in AXES:
        pos_embeds = embed_each(SEED_PHRASES[axis]["positive"])
        neg_embeds = embed_each(SEED_PHRASES[axis]["negative"])

        v_pos = np.mean(pos_embeds, axis=0)
        v_neg = np.mean(neg_embeds, axis=0)

        direction = normalize(v_pos - v_neg)
        axis_vectors[axis] = direction

    return axis_vectors
