from collections import Counter

import numpy as np


def predict_next_token(context, temperature=1.0):
    probs = {
        "and": 0.3,
        "with": 0.25,
        "recommend": 0.2,
        "suggesting": 0.15,
        "indicating": 0.1,
    }
    tokens = list(probs.keys())
    weights = np.array(list(probs.values()))
    scaled = weights ** (1.0 / temperature)
    scaled = scaled / scaled.sum()
    return np.random.choice(tokens, p=scaled)


temperatures = [0.1, 0.5, 1.0, 2.0]
num_trials = 10
# TODO: For each temperature, run num_trials predictions
# and count how often each token is selected.
print("--- Temperature Experiment Results ---")
for temp in temperatures:
    results = [predict_next_token("patient has fever", temp) for _ in range(num_trials)]
    counts = Counter(results)
    top_token, top_count = counts.most_common(1)[0]
    print(
        f"Temp {temp:.1f}: top token = '{top_token}' ({top_count}/{num_trials}); "
        f"unique tokens seen = {len(counts)}"
    )
print("\nRecommendation for AfyaPlus:")
print("  Symptom triage:        temp=0.1-0.3 (consistent, safe responses)")
print("  Greetings:             temp=0.7-1.0 (natural variety)")
print("  Appointment reminders: temp=0.2-0.4 (accurate but slightly varied)")
