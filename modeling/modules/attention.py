import torch
import torch.nn as nn
import torch.nn.functional as F

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def build_rope_cache(seq_len_or_positions, dim, device, mode="seq"):
    """
    if mode == "seq":
        Returns:
            cos, sin: (seq_len, dim)

    if mode == "pos":
        Returns:
            cos, sin: (batch_size, dim)
    """
    assert dim % 2 == 0

    half_dim = dim // 2
    freq = 1.0 / (10000 ** (torch.arange(0, half_dim, device=device).float() / half_dim))

    if mode == 'seq':
        seq_len = seq_len_or_positions
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        angles = torch.einsum("i,j->ij", pos, freq)
    elif mode == 'pos':
        positions = seq_len_or_positions.float()
        angles = positions.unsqueeze(-1) * freq.unsqueeze(0)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    cos = torch.cos(angles).repeat_interleave(2, dim=-1)
    sin = torch.sin(angles).repeat_interleave(2, dim=-1)
    return cos, sin


def apply_rotary(q, k, cos, sin, mode="seq"):
    if mode == "seq":
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
    elif mode == "pos":
        cos = cos[:, None, None, :]
        sin = sin[:, None, None, :]
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin

    return q_rot, k_rot

class MHA(nn.Module):
    def __init__(self, dim, head_dim, is_causal):
        super().__init__()
        assert dim % head_dim == 0

        self.dim = dim
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.is_causal = is_causal

        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.gate_proj = nn.Linear(dim, dim)

        self.out_proj = nn.Linear(dim, dim)

    def forward(self, hidden_states, attention_mask):
        """
        Args:
            hidden_states: (batch_size, seq_len, dim)
            attention_mask: (batch_size, seq_len)

        Returns:
            hidden_states: (batch_size, seq_len, dim)
        """
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        gate = torch.sigmoid(self.gate_proj(hidden_states))

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = build_rope_cache(seq_len, self.head_dim, hidden_states.device)
        q, k = apply_rotary(q, k, cos, sin)

        scale = self.head_dim ** 0.5
        attn_score = (q @ k.transpose(-2, -1)) / scale

        if self.is_causal:
            causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool)
            )[None, None, :, :]
            attn_score = attn_score.masked_fill(~causal_mask, float("-inf"))

        if attention_mask is not None:
            attn_score = attn_score.masked_fill(~attention_mask[:, None, None, :], float("-inf"))

        attn_weight = F.softmax(attn_score, dim=-1)

        out = attn_weight @ v

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)

        return self.out_proj(out * gate), k, v

    def step(self, hidden_states, k_cache, v_cache):
        """
        Args:
            hidden_states: (batch_size, model_dim)
            k_cache, v_cache: (batch_size, num_heads, seq_len, head_dim)

        Returns:
            hidden_states: (batch_size, model_dim)
            k, v: (batch_size, num_heads, seq_len + 1, head_dim)
        """

        batch_size, _ = hidden_states.shape
        seq_len = k_cache.shape[-2]
        device = hidden_states.device

        current_lengths = torch.tensor([seq_len] * batch_size, dtype=torch.long, device=device)

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        gate = torch.sigmoid(self.gate_proj(hidden_states))

        q = q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = build_rope_cache(current_lengths, self.head_dim, device, mode='pos')
        q, k = apply_rotary(q, k, cos, sin, mode='pos')

        k = torch.cat([k_cache, k], dim=2)
        v = torch.cat([v_cache, v], dim=2)

        scale = self.head_dim ** 0.5
        attn_score = (q @ k.transpose(-2, -1)) / scale

        attn_score = F.softmax(attn_score, dim=-1)
        out = attn_score @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, self.dim)

        return self.out_proj(out * gate), k, v


class CrossMHA(nn.Module):
    def __init__(self, dim, head_dim):
        super().__init__()

        assert dim % head_dim == 0

        self.dim = dim
        self.num_heads = dim // head_dim
        self.head_dim = head_dim

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.gate_proj = nn.Linear(dim, dim)

        self.out_proj = nn.Linear(dim, dim)

    def forward(self, hidden_states, context, context_attention_mask):
        """
        Args:
            hidden_states: (batch_size, seq_len, dim)
            context: (batch_size, context_len, dim)
            context_attention_mask: (batch_size, context_len)
        """

        batch_size, seq_len, _ = hidden_states.shape
        context_len = context.shape[1]

        q = self.q_proj(hidden_states)
        k = self.k_proj(context)
        v = self.v_proj(context)
        gate = torch.sigmoid(self.gate_proj(hidden_states))

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, context_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, context_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_score = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        attn_score = attn_score.masked_fill(
            ~context_attention_mask[:, None, None, :],
            float("-inf")
        )

        attn_weight = F.softmax(attn_score, dim=-1)

        out = attn_weight @ v

        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)

        return self.out_proj(out  * gate)
