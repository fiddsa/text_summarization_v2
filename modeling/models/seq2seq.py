import torch
import torch.nn as nn
from dataclasses import dataclass

from modeling.modules import BiBlock, CrossBlock, RMSNorm

@dataclass
class Seq2SeqConfig:
    vocab_size: int = 32000
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    model_dim: int = 512
    head_dim: int = 8
    expansion_factor: int = 2

    num_layers: int = 4
    dropout_rate: float = 0.15

    device: str | None = None


class Seq2Seq(nn.Module):
    def __init__(self, config: Seq2SeqConfig | None = None):
        super().__init__()

        if config is None:
            config = Seq2SeqConfig()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.model_dim)

        self.encoder_layers = nn.ModuleList([
            BiBlock(
                model_dim=config.model_dim,
                head_dim=config.head_dim,
                expansion_factor=config.expansion_factor,
                dropout_rate=config.dropout_rate,
            )
            for _ in range(config.num_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            CrossBlock(
                model_dim=config.model_dim,
                head_dim=config.head_dim,
                expansion_factor=config.expansion_factor,
                dropout_rate=config.dropout_rate,
            )
            for _ in range(config.num_layers)
        ])

        self.norm =  RMSNorm(config.model_dim)

        self.lm_head = nn.Linear(config.model_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, enc_input_ids, enc_attention_mask, dec_input_ids):
        """
        Args: 
            enc_input_ids: (batch_size, enc_seq_len)
            enc_attention_mask: (batch_size, enc_seq_len)
            dec_input_ids: (batch_size, dec_seq_len)
        
        Returns:
            logits: (batch_size, dec_seq_len, vocab_size)
        """

        enc_hidden_states = self.embedding(enc_input_ids)
        dec_hidden_states = self.embedding(dec_input_ids)

        for layer in self.encoder_layers:
            enc_hidden_states = layer(enc_hidden_states, enc_attention_mask)
        
        kv_cache = []

        for layer in self.decoder_layers:
            dec_hidden_states, k, v = layer(dec_hidden_states, enc_hidden_states, enc_attention_mask)
            kv_cache.append({"k": k, "v": v})

        hidden_states = self.norm(dec_hidden_states)
        logits = self.lm_head(hidden_states)

        return logits, enc_hidden_states, kv_cache

    def step(self, input_ids, context, context_attention_mask, kv_cache):
        """
        Args:
            input_ids: (batch_size,)
        
        Returns:
            logits: (batch_size, vocab_size)
        """
        hidden_states = self.embedding(input_ids)
        for layer_idx, layer in enumerate(self.decoder_layers):
            k_cache = kv_cache[layer_idx]["k"]
            v_cache = kv_cache[layer_idx]["v"]
            hidden_states, k_cache, v_cache = layer.step(hidden_states, context, context_attention_mask, k_cache, v_cache)
            kv_cache[layer_idx]["k"] = k_cache
            kv_cache[layer_idx]["v"] = v_cache

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits, kv_cache

    def beam_search(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        beam_size: int = 4,
        length_penalty: float = 0.6,
    ):
        """
        Beam search decoding (batch_size = 1 hoặc xử lý từng sample).

        Args:
            input_ids: (batch_size, src_len)
            max_new_tokens: số token tối đa sinh ra (không tính BOS)
            beam_size: số beam
            length_penalty: alpha trong công thức length penalty:
                            score /= (len_seq) ^ alpha

        Returns:
            seq_ids: (batch_size, out_seq_len) — beam tốt nhất của mỗi sample,
                     được pad bằng pad_id về cùng độ dài.
        """
        with torch.no_grad():
            batch_size = input_ids.size(0)
            device = input_ids.device
            bos_id = self.config.bos_token_id
            eos_id = self.config.eos_token_id
            pad_id = self.config.pad_token_id

            attention_mask = (input_ids != pad_id)
            results = []

            for b in range(batch_size):
                src = input_ids[b].unsqueeze(0)          # (1, src_len)
                mask = attention_mask[b].unsqueeze(0)    # (1, src_len)

                # --- Bước khởi tạo: chạy forward với BOS ---
                dec_input = torch.full((1, 1), bos_id, dtype=torch.long, device=device)
                logits_init, context, kv_cache_init = self.forward(src, mask, dec_input)
                log_probs_init = torch.log_softmax(logits_init[:, -1, :], dim=-1)  # (1, vocab)

                # Lấy top-k token đầu tiên
                topk_log_probs, topk_ids = log_probs_init.topk(beam_size, dim=-1)
                # topk_log_probs, topk_ids: (1, beam_size)

                # Mỗi beam: (accumulated_log_prob, token_sequence, kv_cache)
                beams = []
                for k in range(beam_size):
                    seq = torch.tensor([bos_id, topk_ids[0, k].item()], dtype=torch.long, device=device)
                    score = topk_log_probs[0, k].item()

                    # Nhân bản kv_cache cho beam này
                    beam_kv = [{
                        "k": kv_cache_init[l]["k"].clone(),
                        "v": kv_cache_init[l]["v"].clone(),
                    } for l in range(len(kv_cache_init))]

                    beams.append({
                        "seq": seq,
                        "score": score,
                        "kv_cache": beam_kv,
                        "context": context,   # shared, không cần clone
                        "finished": (topk_ids[0, k].item() == eos_id),
                    })

                completed = []  # beam đã kết thúc

                for _ in range(max_new_tokens - 1):
                    if all(b["finished"] for b in beams):
                        break

                    all_candidates = []

                    for beam in beams:
                        if beam["finished"]:
                            # Beam đã kết thúc: giữ nguyên, không mở rộng
                            all_candidates.append({
                                "seq": beam["seq"],
                                "score": beam["score"],
                                "kv_cache": beam["kv_cache"],
                                "context": beam["context"],
                                "finished": True,
                            })
                            continue

                        last_token = beam["seq"][-1]
                        logits_step, new_kv = self.step(
                            last_token.unsqueeze(0),  # (1,)
                            beam["context"],
                            mask,
                            beam["kv_cache"],
                        )
                        log_probs = torch.log_softmax(logits_step, dim=-1)  # (1, vocab)

                        # Mở rộng top beam_size token tiếp theo
                        topk_lp, topk_tok = log_probs.topk(beam_size, dim=-1)

                        for k in range(beam_size):
                            tok = topk_tok[0, k].item()
                            new_score = beam["score"] + topk_lp[0, k].item()
                            new_seq = torch.cat([beam["seq"], torch.tensor([tok], device=device)])

                            new_kv_copy = [{
                                "k": new_kv[l]["k"].clone(),
                                "v": new_kv[l]["v"].clone(),
                            } for l in range(len(new_kv))]

                            all_candidates.append({
                                "seq": new_seq,
                                "score": new_score,
                                "kv_cache": new_kv_copy,
                                "context": beam["context"],
                                "finished": (tok == eos_id),
                            })

                    # --- Chọn beam_size beam tốt nhất theo length-penalized score ---
                    def penalized_score(cand):
                        length = cand["seq"].size(0) - 1  # trừ BOS
                        lp = (length ** length_penalty) if length > 0 else 1.0
                        return cand["score"] / lp

                    all_candidates.sort(key=penalized_score, reverse=True)
                    beams = all_candidates[:beam_size]

                # Chọn beam tốt nhất (có thể đã kết thúc hoặc chưa)
                def final_score(cand):
                    length = cand["seq"].size(0) - 1
                    lp = (length ** length_penalty) if length > 0 else 1.0
                    return cand["score"] / lp

                best = max(beams, key=final_score)
                seq = best["seq"]

                # Cắt sau EOS nếu có
                eos_positions = (seq == eos_id).nonzero(as_tuple=True)[0]
                if len(eos_positions) > 0:
                    seq = seq[:eos_positions[0] + 1]

                results.append(seq)

            # Pad các sequence về cùng độ dài
            max_len = max(s.size(0) for s in results)
            padded = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=device)
            for i, s in enumerate(results):
                padded[i, :s.size(0)] = s

            return padded

    def generate(self, input_ids: torch.Tensor, max_new_tokens=100):
        """
        Args:
            input_ids: (batch_size, seq_len)

        Returns:
            seq_ids: (batch_size, out_seq_len)
        """
        with torch.no_grad():
            batch_size = input_ids.size(0)
            device = input_ids.device
            bos_id = self.config.bos_token_id
            eos_id = self.config.eos_token_id
            pad_id = self.config.pad_token_id

            attention_mask = (input_ids != pad_id)

            seq_ids = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            logits, context, kv_cache = self.forward(input_ids, attention_mask, seq_ids)
            logits = logits[:, -1, :]

            for _ in range(max_new_tokens):
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.argmax(probs, dim=-1, keepdim=True)

                seq_ids = torch.cat([seq_ids, next_token], dim=1)
                finished |= (next_token.squeeze(1) == eos_id)

                if finished.all():
                    break

                logits, kv_cache = self.step(next_token.squeeze(1), context, attention_mask, kv_cache)

            eos_mask = (seq_ids == eos_id)
            first_eos = eos_mask.float().cumsum(dim=1) >= 1
            seq_ids = torch.where(first_eos, eos_id, seq_ids)
            
            return seq_ids