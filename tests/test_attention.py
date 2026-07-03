import torch

from model.attention import SelfAttention
from utils.config import ModelConfig

config = ModelConfig()

attention = SelfAttention(config.embedding_dim)

x = torch.randn(2, 8, config.embedding_dim)

scores, V = attention(x)

print("Attention Score Shape")
print(scores.shape)

print()

print("Value Shape")
print(V.shape)