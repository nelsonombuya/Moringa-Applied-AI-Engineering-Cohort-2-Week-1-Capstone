import numpy as np


def emergency_tokeniser(text):
    """Split text into tokens while preserving '!' as separate tokens."""
    text = text.lower()
    # TODO 1: Pad '!' with spaces so split() treats them as separate tokens.
    text = text.replace("!", " ! ")

    # TODO 2: Remove any other punctuation that should not be a token.
    text = (
        text.replace(".", "")
        .replace(",", "")
        .replace("?", "")
        .replace(";", "")
        .replace(":", "")
        .replace('"', "")
        .replace("'", "")
    )

    # TODO 3: Split on whitespace and return a NumPy array.
    tokens = np.array(text.split())  # replace this with the real result
    return tokens


# Test cases
samples = [
    "Help!!!",
    "I cannot breathe!",
    "My chest hurts.",
    "My chest hurts. Is it a heart attack?",
    "My chest hurts!!!",
]
for s in samples:
    tokens = emergency_tokeniser(s)
    line = f"{s} -> {tokens}"
    width = max(len(f"{sample} -> {emergency_tokeniser(sample)}") for sample in samples)
    print(f"{line:<{width}} | count: {len(tokens)}")
