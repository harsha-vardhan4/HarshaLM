# import math

# import torch
# import torch.nn as nn

# from utils.config import ModelConfig


# class SelfAttention(nn.Module):
#     """
#     Single-head causal self-attention.
#     """

#     def __init__(self, config: ModelConfig):
#         super().__init__()

#         self.embedding_dim = config.embedding_dim
#         self.context_length = config.context_length

#         # QKV projections
#         self.query = nn.Linear(
#             self.embedding_dim,
#             self.embedding_dim,
#             bias=False
#         )

#         self.key = nn.Linear(
#             self.embedding_dim,
#             self.embedding_dim,
#             bias=False
#         )

#         self.value = nn.Linear(
#             self.embedding_dim,
#             self.embedding_dim,
#             bias=False
#         )

#         # Attention dropout
#         self.attention_dropout = nn.Dropout(
#             config.attention_dropout
#         )

#         # Register causal mask once
#         mask = torch.triu(
#             torch.ones(
#                 self.context_length,
#                 self.context_length,
#                 dtype=torch.bool
#             ),
#             diagonal=1
#         )

#         self.register_buffer(
#             "causal_mask",
#             mask
#         )

#     def _compute_qkv(self, x):
#         Q = self.query(x)
#         K = self.key(x)
#         V = self.value(x)

#         return Q, K, V

#     def _compute_attention_scores(self, Q, K):
#         scores = Q @ K.transpose(-2, -1)

#         scores = scores / math.sqrt(
#             self.embedding_dim
#         )

#         return scores

#     def _apply_causal_mask(self, scores):

#         seq_len = scores.size(-1)

#         mask = self.causal_mask[
#             :seq_len,
#             :seq_len
#         ]

#         scores = scores.masked_fill(
#             mask,
#             float("-inf")
#         )

#         return scores

#     def _compute_attention_weights(self, scores):

#         weights = torch.softmax(
#             scores,
#             dim=-1
#         )

#         weights = self.attention_dropout(
#             weights
#         )

#         return weights

#     def _compute_output(self, weights, V):

#         return weights @ V

#     def forward(self, x):

#         assert x.dim() == 3, (
#             "Expected input shape "
#             "(batch, sequence, embedding)"
#         )

#         Q, K, V = self._compute_qkv(x)

#         assert Q.shape == K.shape == V.shape

#         scores = self._compute_attention_scores(
#             Q,
#             K
#         )

#         scores = self._apply_causal_mask(
#             scores
#         )

#         weights = self._compute_attention_weights(
#             scores
#         )

#         output = self._compute_output(
#             weights,
#             V
#         )

#         return output