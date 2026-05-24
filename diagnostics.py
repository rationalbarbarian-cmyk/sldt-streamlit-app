# diagnostics.py
def rotation_is_monotonic(rotation):
    return all(rotation[i] <= rotation[i+1] for i in range(len(rotation)-1))

def interpret(alpha, beta, coherence, curvature, rotation):
    interpretation = []

    # Static
    if alpha["Valence"] > 0.5:
        interpretation.append("Strong positive emotional tone.")
    if alpha["Power"] < -0.2:
        interpretation.append("Low-power or deferential stance.")
    if alpha["Agency"] > 0.2:
        interpretation.append("Moderate sense of agency.")

    # Dynamic
    if beta["Valence"] > 0:
        interpretation.append("Increasing positivity across the sentence.")
    if beta["Agency"] > 0:
        interpretation.append("Rising sense of agency.")

    # Coherence
    if coherence > 0.75:
        interpretation.append("High semantic coherence.")

    # Curvature
    if max(curvature) > 0.15:
        interpretation.append("Curvature spike marking a narrative pivot.")

    # Phase rotation
    if rotation_is_monotonic(rotation):
        interpretation.append("Monotonic phase rotation indicating stable direction.")

    return interpretation
