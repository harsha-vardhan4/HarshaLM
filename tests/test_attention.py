import torch

from model.self_attention import SelfAttention
from utils.config import ModelConfig


config = ModelConfig()

attention = SelfAttention(config)

x = torch.randn(
    2,
    8,
    config.embedding_dim
)

output = attention(x)

print("Input Shape")
print(x.shape)

print()

print("Output Shape")
print(output.shape)