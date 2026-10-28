import numpy as np


def urgency_tokeniser(text):
    text = text.lower()
    text = text.replace("!", " ! ")  # bug is on this line
    tokens = np.array(text.split())
    return tokens


print(urgency_tokeniser("Help!!! I cannot breathe!"))
