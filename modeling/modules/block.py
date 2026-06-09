import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple

from modeling.modules.rms_norm import RMSNorm
from modeling.modules.attention import MHA, CrossMHA
from modeling.modules.feed_forward import SwiGLU

class BiBlock(nn.Module):
    def __init__(
        self,
        model_dim: int = 512,
        head_dim: int = 64,
        expansion_factor: int = 2,
        dropout_rate: float = 0.15,
    ):
        super().__init__()

        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)
        self.mha = MHA(model_dim, head_dim, is_causal=False)
        self.ffn = SwiGLU(model_dim, model_dim * expansion_factor)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states, attention_mask):
        """
        Args:
            hidden_states: (batch_size, seq_len, model_dim)
            attention_mask: (batch_size, seq_len)
        
        Returns: 
            hidden_states: (batch_size, seq_len, model_dim)
        """

        res = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states, _, _ = self.mha(hidden_states, attention_mask)
        hidden_states = res + self.dropout(hidden_states)

        res = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = res + self.dropout(hidden_states)
        
        return hidden_states

class CrossBlock(nn.Module):
    def __init__(
        self,         
        model_dim: int = 512,
        head_dim: int = 64,
        expansion_factor: int = 2,
        dropout_rate: float = 0.15
    ):
        super().__init__()

        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)
        self.norm3 = RMSNorm(model_dim)
        self.mha = MHA(model_dim, head_dim, is_causal=True)
        self.cross_mha = CrossMHA(model_dim, head_dim)
        self.ffn = SwiGLU(model_dim, model_dim * expansion_factor)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, hidden_states, context, context_attention_mask):
        res = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states, k, v = self.mha(hidden_states, attention_mask=None)
        hidden_states = res + self.dropout(hidden_states)

        res = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.cross_mha(hidden_states, context, context_attention_mask)
        hidden_states = res + self.dropout(hidden_states)

        res = hidden_states
        hidden_states = self.norm3(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = res + self.dropout(hidden_states)

        return hidden_states, k, v
    
    def step(self, hidden_states, context, context_attention_mask, k_cache, v_cache):
        res = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states, k_cache, v_cache = self.mha.step(hidden_states, k_cache, v_cache)
        hidden_states = res + hidden_states

        res = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = hidden_states.unsqueeze(1)
        hidden_states = self.cross_mha(hidden_states, context, context_attention_mask)
        hidden_states = hidden_states.squeeze(1)
        hidden_states = res + hidden_states

        res = hidden_states
        hidden_states = self.norm3(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = res + hidden_states
        
        return hidden_states, k_cache, v_cache
