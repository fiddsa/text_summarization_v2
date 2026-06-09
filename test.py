import random
import numpy as np
import torch

from modeling.models.seq2seq import Seq2Seq, Seq2SeqConfig

SEED = 0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def build_model(test_generate=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = Seq2SeqConfig(
        vocab_size=1000,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=-1 if test_generate else 2,
        model_dim=128,
        head_dim=16,
        expansion_factor=2,
        num_layers=2,
        dropout_rate=0.0,
        use_rope=True,
        device=device,
    )

    model = Seq2Seq(cfg).to(device)
    model.eval()
    return model, cfg


def test_forward():
    print("test_forward")

    model, cfg = build_model(test_generate=False)
    device = cfg.device

    batch_size = 2
    src_len = 12
    tgt_len = 8
    vocab_size = cfg.vocab_size

    enc_input_ids = torch.randint(3, vocab_size, (batch_size, src_len), device=device)
    enc_attention_mask = torch.ones(batch_size, src_len, device=device, dtype=torch.bool)

    dec_input_ids = torch.empty(batch_size, tgt_len, dtype=torch.long, device=device)
    dec_input_ids[:, 0] = cfg.bos_token_id
    dec_input_ids[:, 1:] = torch.randint(3, vocab_size, (batch_size, tgt_len - 1), device=device)

    logits, enc_hidden_states, kv_cache = model(enc_input_ids, enc_attention_mask, dec_input_ids)

    assert logits.shape == (batch_size, tgt_len, vocab_size)
    assert enc_hidden_states.shape == (batch_size, src_len, cfg.model_dim)
    assert len(kv_cache) == cfg.num_layers
    print("test_forward passed")


def test_generate_forward():
    print("test_generate_forward")

    model, cfg = build_model(test_generate=True)
    device = cfg.device

    batch_size = 2
    src_len = 12
    max_new = 4
    vocab_size = cfg.vocab_size

    enc_input_ids = torch.randint(3, vocab_size, (batch_size, src_len), device=device)

    gen_ids = model.generate(enc_input_ids, max_new_tokens=max_new)
    assert gen_ids.shape[0] == batch_size
    assert gen_ids.shape[1] <= max_new + 1
    assert gen_ids[:, 0].eq(cfg.bos_token_id).all()
    print("test_generate_forward passed")


if __name__ == "__main__":
    test_forward()
    test_generate_forward()
    print("all tests passed")
