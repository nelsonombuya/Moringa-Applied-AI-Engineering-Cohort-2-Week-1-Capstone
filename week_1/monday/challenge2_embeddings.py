import numpy as np

embeddings = {
    "fever": np.array([0.9, 0.0, 0.3]),
    "pain": np.array([0.8, 0.1, 0.4]),
    "discharge": np.array([0.5, 0.5, 0.1]),
    "billing": np.array([0.0, 0.9, 0.1]),
    "admit": np.array([0.2, 0.8, 0.2]),
    # TODO 1: Add a 'cough' vector that sits close to 'fever' and far from 'billing'.
    "cough": np.array([0.85, 0.05, 0.25]),
}


def cosine_similarity(word1, word2):
    vec1, vec2 = embeddings[word1], embeddings[word2]
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# TODO 2: Print cosine_similarity('cough', 'fever') and cosine_similarity('cough',
# 'billing').
cough_fever_similarity = cosine_similarity("cough", "fever")
print(f"Context Match (Cough + Fever): {cough_fever_similarity:.4f}")

cough_billing_similarity = cosine_similarity("cough", "billing")
print(f"Context Match (Cough + Billing): {cough_billing_similarity:.4f}")

# TODO 3: Confirm the fever score is meaningfully higher than the billing score.
print(
    "Is fever score higher than Billing score? "
    f"{cough_fever_similarity > cough_billing_similarity}"
)
