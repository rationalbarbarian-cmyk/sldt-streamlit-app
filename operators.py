# operators.py
import numpy as np

def compute_static_alignment(x_static, axis_vectors):
    """Compute α_k for each axis."""
    alpha = {}
    for axis, vec in axis_vectors.items():
        alpha[axis] = float(np.dot(vec, x_static))
    return alpha

def difference_operator(x_tokens):
    """Compute first differences y[i] = x[i+1] - x[i]."""
    return np.diff(x_tokens, axis=0)

def compute_dynamic_motion(x_tokens, axis_vectors):
    """Compute β_k for each axis."""
    y = difference_operator(x_tokens)
    y_mean = np.mean(y, axis=0)

    beta = {}
    for axis, vec in axis_vectors.items():
        beta[axis] = float(np.dot(vec, y_mean))
    return beta

def compute_coherence(x_tokens):
    """High coherence = low variance of distances from mean."""
    mean_vec = np.mean(x_tokens, axis=0)
    distances = [np.linalg.norm(x - mean_vec) for x in x_tokens]
    return float(1 - np.var(distances))

def compute_curvature(x_tokens):
    """Second difference operator."""
    curvature = []
    for i in range(1, len(x_tokens) - 1):
        c = x_tokens[i+1] - 2*x_tokens[i] + x_tokens[i-1]
        curvature.append(float(np.linalg.norm(c)))
    return curvature

def compute_phase_rotation(alpha_over_time):
    """Compute angle change between successive projections."""
    rotation = []
    for i in range(len(alpha_over_time) - 1):
        a = alpha_over_time[i]
        b = alpha_over_time[i+1]
        cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        rotation.append(float(np.arccos(np.clip(cos_theta, -1, 1))))
    return rotation
