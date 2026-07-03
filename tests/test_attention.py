import torch

from model.attention import SelfAttention
from utils.config import ModelConfig

config = ModelConfig()

attention = SelfAttention(
    config.embedding_dim
)

x = torch.randn(
    2,
    8,
    config.embedding_dim
)

Q, K, V = attention(x)

print("Input Shape")
print(x.shape)

print()

print("Query Shape")
print(Q.shape)

print()

print("Key Shape")
print(K.shape)

print()

print("Value Shape")
print(V.shape)