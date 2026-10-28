import numpy as np


def predict_next_token(context, temperature=1.0):
    """
    Simulate how an LLM predicts the next token.
    Lower temperature = more predictable (safer for medical advice).
    Higher temperature = more creative (good for greetings).
    """
    predictions = {
        "patient has fever": {
            "and": 0.3,
            "with": 0.25,
            "recommend": 0.2,
            "suggesting": 0.15,
            "indicating": 0.1,
        }
    }
    if context not in predictions:
        return "[unknown context]"

    probs = predictions[context]
    tokens = list(probs.keys())
    weights = np.array(list(probs.values()))

    # Apply temperature scaling
    scaled = weights ** (1.0 / temperature)
    scaled = scaled / scaled.sum()
    print(f"Tokens: {tokens}")
    print(f"Scaled weights (temperature={temperature}): {scaled}")
    print(f"Normalized probabilities: {scaled}")

    chosen = np.random.choice(tokens, p=scaled)
    return chosen


print("--- Token Prediction Simulation ---")
print("\nLow temperature (0.3) - Conservative/Safe:")
for i in range(3):
    token = predict_next_token("patient has fever", temperature=0.3)
    print(f"  Attempt {i+1}: '{token}'")

print("\nHigh temperature (2.0) - Creative/Varied:")
for i in range(3):
    token = predict_next_token("patient has fever", temperature=2.0)
    print(f"  Attempt {i+1}: '{token}'")
