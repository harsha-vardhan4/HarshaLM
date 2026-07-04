import torch

from model.token_embedding import TokenEmbedding
from utils.config import ModelConfig


config = ModelConfig()

embedding = TokenEmbedding(
    vocab_size=config.vocab_size,
    embedding_dim=config.embedding_dim
)

tokens = torch.tensor([
    [10, 20, 30, 40]
])

print("Input Shape:")
print(tokens.shape)

print()

output = embedding(tokens)

print("Output Shape:")
print(output.shape)

print()

print(output)