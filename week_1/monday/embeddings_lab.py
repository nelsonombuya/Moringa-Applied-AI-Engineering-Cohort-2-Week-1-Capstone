import numpy as np

# Simulated 3D word embeddings.
# Each dimension represents: [Clinical-Symptom, Administrative, Time/Urgency]
embeddings = {
    "fever": np.array([0.9, 0.0, 0.3]),
    "pain": np.array([0.8, 0.1, 0.4]),
    "discharge": np.array([0.5, 0.5, 0.1]),  # Ambiguous word!
    "billing": np.array([0.0, 0.9, 0.1]),
    "admit": np.array([0.2, 0.8, 0.2]),
}


def cosine_similarity(word1, word2):
    """Calculate how close two words are in embedding space."""
    vec1 = embeddings[word1]
    vec2 = embeddings[word2]
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# If 'discharge' appears alongside 'pain', it is likely a clinical symptom.
# If it appears alongside 'billing', it is administrative.
print("--- Context-Disambiguation via Cosine Similarity ---")
print(f"Context Match (Discharge + Pain): {cosine_similarity('discharge', 'pain'):.4f}")
print(
    "Context Match (Discharge + Billing): "
    f"{cosine_similarity('discharge', 'billing'):.4f}"
)
