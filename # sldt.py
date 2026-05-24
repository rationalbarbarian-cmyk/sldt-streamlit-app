# sldt.py
from embedding import embed, embed_each, tokenize
from axes import build_axis_vectors
from operators import (
    compute_static_alignment,
    compute_dynamic_motion,
    compute_coherence,
    compute_curvature,
    compute_phase_rotation
)
from diagnostics import interpret

def run_sldt(text, SEED_PHRASES):
    """Full SLDT pipeline for a single text."""

    # 1. Embed text
    x_static = embed(text)
    tokens = tokenize(text)
    x_tokens = embed_each(tokens)

    # 2. Build axes
    axis_vectors = build_axis_vectors(SEED_PHRASES)

    # 3. Static + dynamic
    alpha = compute_static_alignment(x_static, axis_vectors)
    beta = compute_dynamic_motion(x_tokens, axis_vectors)

    # 4. Geometry
    coherence = compute_coherence(x_tokens)
    curvature = compute_curvature(x_tokens)

    # 5. Phase rotation (project tokens onto axes)
    alpha_over_time = [
        [float(vec.dot(x)) for vec in axis_vectors.values()]
        for x in x_tokens
    ]
    rotation = compute_phase_rotation(alpha_over_time)

    # 6. Interpretation
    interpretation = interpret(alpha, beta, coherence, curvature, rotation)

    return {
        "alpha": alpha,
        "beta": beta,
        "coherence": coherence,
        "curvature": curvature,
        "rotation": rotation,
        "interpretation": interpretation,
        "tokens": tokens
    }
