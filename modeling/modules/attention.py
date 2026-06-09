import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) applied to query/key heads."""

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head_dim")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device=None, dtype=None):
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq.to(device))
        cos = freqs.cos().to(dtype=dtype)
        sin = freqs.sin().to(dtype=dtype)
        return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    Args:
        x:   (batch_size, num_heads, seq_len, head_dim)
        cos: (seq_len, head_dim // 2)
        sin: (seq_len, head_dim // 2)
    """
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]

    out = torch.empty_like(x)
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out


class MHA(nn.Module):
    """Multi-head self-attention. Standard Transformer attention with optional RoPE."""

    def __init__(
        self,
        dim: int,
        head_dim: int,
        is_causal: bool,
        dropout_rate: float = 0.1,
        use_rope: bool = True,
    ):
        super().__init__()
        assert dim % head_dim == 0, "dim must be divisible by head_dim"

        self.dim = dim
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.is_causal = is_causal
        self.use_rope = use_rope

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.rope = RotaryEmbedding(head_dim) if use_rope else None

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

        if self.rope is not None:
            cos, sin = self.rope(seq_len, device=hidden_states.device, dtype=q.dtype)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

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
    """Standard multi-head encoder-decoder attention.

    Cross-attention is kept standard. RoPE is used in encoder/decoder self-attention
    to provide positional information before cross-attention.
    """

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
