import numpy as np


def simple_numpy_tokenizer(text):
    clean_text = text.lower().replace(".", "").replace("?", "").replace(",", "")
    tokens = clean_text.split()
    return np.array(tokens)


original_query = (
    "Hujambo, it is Juma. I am feeling very hot, my head hurts since yesterday, "
    "and I am coughing. I cannot go to the clinic because of the rain. What should I do?"
)

# TODO 1: Tokenise the original_query and count the tokens.
original_tokens = simple_numpy_tokenizer(original_query)
print(f"Original Query Tokens: {original_tokens}")

original_token_count = len(original_tokens)
print(f"Original Query Token Count: {original_token_count}")

# TODO 2: Manually rewrite the query as a 10-word summary of the medical facts only.
summary = "I am feeling very hot, my head hurts, and I am coughing."

# TODO 3: Tokenise the summary and count the tokens.
summary_tokens = simple_numpy_tokenizer(summary)
print(f"Summary Tokens: {summary_tokens}")

summary_token_count = len(summary_tokens)
print(f"Summary Token Count: {summary_token_count}")

# TODO 4: Calculate the percentage of tokens saved.
tokens_saved = original_token_count - summary_token_count
print(f"Tokens Saved: {tokens_saved}")

percentage_saved = (tokens_saved / original_token_count) * 100
print(f"Percentage of Tokens Saved: {percentage_saved:.2f}%")

# TODO 5: If 1,000 users send this query daily, how many tokens are saved per month?
daily_users = 1_000_000
tokens_saved_per_day = tokens_saved * daily_users
print(f"Tokens Saved Per Day: {tokens_saved_per_day}")

tokens_saved_per_month = tokens_saved_per_day * 30
print(f"Tokens Saved Per Month: {tokens_saved_per_month}")
