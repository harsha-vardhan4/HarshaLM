import torch

from model.embedding import TokenEmbedding
from model.position_embedding import PositionalEmbedding
from utils.config import ModelConfig


config = ModelConfig()

token_embedding = TokenEmbedding(
    config.vocab_size,
    config.embedding_dim
)

position_embedding = PositionalEmbedding(
    config.context_length,
    config.embedding_dim
)

tokens = torch.tensor([
    [10, 20, 30, 40]
])

token_vectors = token_embedding(tokens)

print("Token Embedding Shape")
print(token_vectors.shape)

print()

final_vectors = position_embedding(token_vectors)

print("Final Embedding Shape")
print(final_vectors.shape)

print()

print(final_vectors)