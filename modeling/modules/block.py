import torch.nn as nn

from modeling.modules.attention import MHA, CrossMHA
from modeling.modules.feed_forward import FeedForward


class BiBlock(nn.Module):
    """Transformer encoder block: self-attention + FFN + residual + LayerNorm."""

    def __init__(
        self,
        model_dim: int = 512,
        head_dim: int = 64,
        expansion_factor: int = 4,
        dropout_rate: float = 0.1,
        use_rope: bool = True,
    ):
        super().__init__()
        ff_hidden_dim = model_dim * expansion_factor

        self.mha = MHA(
            model_dim,
            head_dim,
            is_causal=False,
            dropout_rate=dropout_rate,
            use_rope=use_rope,
        )
        self.ffn = FeedForward(model_dim, ff_hidden_dim, dropout_rate=dropout_rate)

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states, attention_mask):
        attn_out, _, _ = self.mha(hidden_states, attention_mask)
        hidden_states = self.norm1(hidden_states + self.dropout(attn_out))

        ffn_out = self.ffn(hidden_states)
        hidden_states = self.norm2(hidden_states + self.dropout(ffn_out))
        return hidden_states


class CrossBlock(nn.Module):
    """Transformer decoder block: causal self-attn + cross-attn + FFN."""

    def __init__(
        self,
        model_dim: int = 512,
        head_dim: int = 64,
        expansion_factor: int = 4,
        dropout_rate: float = 0.1,
        use_rope: bool = True,
    ):
        super().__init__()
        ff_hidden_dim = model_dim * expansion_factor

        self.mha = MHA(
            model_dim,
            head_dim,
            is_causal=True,
            dropout_rate=dropout_rate,
            use_rope=use_rope,
        )
        self.cross_mha = CrossMHA(model_dim, head_dim, dropout_rate=dropout_rate)
        self.ffn = FeedForward(model_dim, ff_hidden_dim, dropout_rate=dropout_rate)

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states, context, context_attention_mask):
        self_attn_out, k, v = self.mha(hidden_states, attention_mask=None)
        hidden_states = self.norm1(hidden_states + self.dropout(self_attn_out))

        cross_attn_out = self.cross_mha(hidden_states, context, context_attention_mask)
        hidden_states = self.norm2(hidden_states + self.dropout(cross_attn_out))

        ffn_out = self.ffn(hidden_states)
        hidden_states = self.norm3(hidden_states + self.dropout(ffn_out))
        return hidden_states, k, v
