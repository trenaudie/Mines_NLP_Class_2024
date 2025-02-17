from transformers import PreTrainedTokenizerBase

def print_tokenizer_special_ids(tokenizer):
    # Get special tokens and their IDs
    special_tokens = tokenizer.all_special_tokens
    special_token_ids = {token: tokenizer.get_vocab()[token] for token in special_tokens}

    # Print special tokens and their IDs
    for token, token_id in special_token_ids.items():
        print(f"Token: {token}, ID: {token_id}")