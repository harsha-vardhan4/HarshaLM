from tokenizer.tokenizer import create_tokenizer

tokenizer = create_tokenizer()

print("Vocab size:", tokenizer.vocab_size)

print(
    tokenizer.user_token,
    tokenizer.user_token_id
)

print(
    tokenizer.assistant_token,
    tokenizer.assistant_token_id
)

print(
    tokenizer.system_token,
    tokenizer.system_token_id
)

print(
    tokenizer.eos_token,
    tokenizer.eos_token_id
)