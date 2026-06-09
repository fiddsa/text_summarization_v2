import torch
import torch.nn as nn
import torch.nn.functional as F


class MHA(nn.Module):
    """Standard multi-head self-attention."""

    def __init__(self, dim: int, head_dim: int, is_causal: bool, dropout_rate: float = 0.1):
        super().__init__()
        assert dim % head_dim == 0, "dim must be divisible by head_dim"

        self.dim = dim
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.is_causal = is_causal

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: (batch_size, seq_len, dim)
            attention_mask: (batch_size, seq_len), True for valid tokens
        """
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if self.is_causal:
            causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool)
            )
            attn_scores = attn_scores.masked_fill(~causal_mask[None, None, :, :], float("-inf"))

        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(~attention_mask[:, None, None, :].bool(), float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = attn_weights @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        return self.out_proj(out), k, v


class CrossMHA(nn.Module):
    """Standard multi-head encoder-decoder attention."""

    def __init__(self, dim: int, head_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        assert dim % head_dim == 0, "dim must be divisible by head_dim"

        self.dim = dim
        self.num_heads = dim // head_dim
        self.head_dim = head_dim

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states, context, context_attention_mask=None):
        """
        Args:
            hidden_states: (batch_size, tgt_len, dim)
            context: (batch_size, src_len, dim)
            context_attention_mask: (batch_size, src_len), True for valid tokens
        """
        batch_size, tgt_len, _ = hidden_states.shape
        src_len = context.shape[1]

        q = self.q_proj(hidden_states)
        k = self.k_proj(context)
        v = self.v_proj(context)

        q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if context_attention_mask is not None:
            attn_scores = attn_scores.masked_fill(
                ~context_attention_mask[:, None, None, :].bool(),
                float("-inf"),
            )

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = attn_weights @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.dim)
        return self.out_proj(out)
