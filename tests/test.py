import torch

from utils.config import ModelConfig
from model.harsha_lm import HarshaLM


config = ModelConfig()

model = HarshaLM(config)

assert (
    model.embedding_layer
        .token_embedding
        .embedding
        .weight
    is
    model.lm_head
        .linear
        .weight
)

print("✓ Weight tying successful")