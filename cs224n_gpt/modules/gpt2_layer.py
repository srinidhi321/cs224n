from torch import nn

import torch.nn.functional as F

from modules.attention import CausalSelfAttention

class GPT2Layer(nn.Module):
  def __init__(self, config):
    super().__init__()
    # Multi-head attention.
    self.self_attention = CausalSelfAttention(config)
    # Add-norm for multi-head attention.
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    # Feed forward.
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    # Add-norm for feed forward.
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

  def add(self, input, output, dropout):
    return input + dropout(output)


  def forward(self, hidden_states, attention_mask):
    # Layer norm 1
    x = self.attention_layer_norm(hidden_states)
    # Self-attention
    x = self.self_attention(x, attention_mask)
    x = self.attention_dense(x)
    # Residual connection 1
    y = self.add(hidden_states, x, self.attention_dropout)

    # Layer norm 2
    mlp = self.out_layer_norm(y)
    # Feed-forward
    mlp = self.interm_dense(mlp)
    mlp = self.interm_af(mlp)
    mlp = self.out_dense(mlp)
    # Residual connection 2
    out = self.add(y, mlp, self.out_dropout)
    return out
    






