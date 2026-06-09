import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from modeling.modules import BiBlock, CrossBlock


@dataclass
class Seq2SeqConfig:
    vocab_size: int = 32000
    pad_token_id: int = 0
    bos_token_id: int = 2
    eos_token_id: int = 3

    model_dim: int = 512
    head_dim: int = 64
    expansion_factor: int = 4
    num_layers: int = 4
    dropout_rate: float = 0.1
    use_rope: bool = True

    device: str | None = None


class Seq2Seq(nn.Module):
    """Transformer encoder-decoder for text summarization.

    This is the standard Transformer structure:
    - token embedding scaled by sqrt(model_dim)
    - encoder self-attention blocks
    - decoder causal self-attention + encoder-decoder cross-attention blocks
    - position-wise FFN with ReLU
    - residual connections and LayerNorm

    The only positional mechanism is RoPE, applied to Q/K in encoder and decoder
    self-attention. Absolute sinusoidal positional encoding is not used.
    """

    def __init__(self, config: Seq2SeqConfig | None = None):
        super().__init__()

        if config is None:
            config = Seq2SeqConfig()
        self.config = config

        self.embedding = nn.Embedding(
            config.vocab_size,
            config.model_dim,
            padding_idx=config.pad_token_id,
        )
        self.dropout = nn.Dropout(config.dropout_rate)

        self.encoder_layers = nn.ModuleList([
            BiBlock(
                model_dim=config.model_dim,
                head_dim=config.head_dim,
                expansion_factor=config.expansion_factor,
                dropout_rate=config.dropout_rate,
                use_rope=config.use_rope,
            )
            for _ in range(config.num_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            CrossBlock(
                model_dim=config.model_dim,
                head_dim=config.head_dim,
                expansion_factor=config.expansion_factor,
                dropout_rate=config.dropout_rate,
                use_rope=config.use_rope,
            )
            for _ in range(config.num_layers)
        ])

        self.lm_head = nn.Linear(config.model_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def _embed(self, input_ids):
        hidden_states = self.embedding(input_ids) * math.sqrt(self.config.model_dim)
        return self.dropout(hidden_states)

    def encode(self, enc_input_ids, enc_attention_mask):
        enc_hidden_states = self._embed(enc_input_ids)
        for layer in self.encoder_layers:
            enc_hidden_states = layer(enc_hidden_states, enc_attention_mask)
        return enc_hidden_states

    def decode(self, dec_input_ids, enc_hidden_states, enc_attention_mask):
        dec_hidden_states = self._embed(dec_input_ids)
        kv_cache = []
        for layer in self.decoder_layers:
            dec_hidden_states, k, v = layer(
                dec_hidden_states,
                enc_hidden_states,
                enc_attention_mask,
            )
            kv_cache.append({"k": k, "v": v})
        return dec_hidden_states, kv_cache

    def forward(self, enc_input_ids, enc_attention_mask, dec_input_ids):
        """
        Args:
            enc_input_ids: (batch_size, src_len)
            enc_attention_mask: (batch_size, src_len), True/1 for valid source tokens
            dec_input_ids: (batch_size, tgt_len)

        Returns:
            logits: (batch_size, tgt_len, vocab_size)
            enc_hidden_states: encoder output
            kv_cache: kept only for compatibility with existing trainer code
        """
        enc_attention_mask = enc_attention_mask.bool()
        enc_hidden_states = self.encode(enc_input_ids, enc_attention_mask)
        dec_hidden_states, kv_cache = self.decode(
            dec_input_ids,
            enc_hidden_states,
            enc_attention_mask,
        )
        logits = self.lm_head(dec_hidden_states)
        return logits, enc_hidden_states, kv_cache

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100):
        """Greedy autoregressive decoding."""
        self.eval()

        batch_size = input_ids.size(0)
        device = input_ids.device
        bos_id = self.config.bos_token_id
        eos_id = self.config.eos_token_id
        pad_id = self.config.pad_token_id

        attention_mask = input_ids != pad_id
        seq_ids = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            logits, _, _ = self.forward(input_ids, attention_mask, seq_ids)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            next_token = torch.where(
                finished.unsqueeze(1),
                torch.full_like(next_token, pad_id),
                next_token,
            )
            seq_ids = torch.cat([seq_ids, next_token], dim=1)
            finished |= next_token.squeeze(1).eq(eos_id)
            if finished.all():
                break

        return seq_ids
