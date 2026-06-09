import random
import numpy as np
import torch

# Giả sử model nằm trong modeling/seq2seq.py
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
        eos_token_id=-1 if test_generate else 2,  # tắt early stop trong test generate
        model_dim=128,
        head_dim=16,
        expansion_factor=2,
        num_layers=2,
        dropout_rate=0.0,
        device=device,
    )

    model = Seq2Seq(cfg).to(device)
    model.eval()
    return model, cfg


def test_forward_step():
    print("test_forward_step")

    model, cfg = build_model(test_generate=False)
    device = cfg.device

    batch_size = 4
    src_len = 32
    tgt_len = 16              # số token decoder sau BOS
    vocab_size = cfg.vocab_size

    # source input (không dùng token đặc biệt)
    enc_input_ids = torch.randint(3, vocab_size, (batch_size, src_len), device=device)
    enc_attention_mask = torch.ones(batch_size, src_len, device=device, dtype=torch.bool)

    # target đầy đủ: BOS + tgt_len token ngẫu nhiên (tổng độ dài L = tgt_len+1)
    dec_full = torch.empty(batch_size, tgt_len + 1, dtype=torch.long, device=device)
    dec_full[:, 0] = cfg.bos_token_id
    dec_full[:, 1:] = torch.randint(3, vocab_size, (batch_size, tgt_len), device=device)

    # forward toàn bộ
    logits_full, _, _ = model.forward(enc_input_ids, enc_attention_mask, dec_full)
    # logits_full: (batch, L, vocab) với L = tgt_len+1

    # Khởi tạo decoder chỉ với token BOS
    dec_start = torch.full((batch_size, 1), cfg.bos_token_id, dtype=torch.long, device=device)
    logits_initial, context, kv_cache = model.forward(enc_input_ids, enc_attention_mask, dec_start)
    logits_step = logits_initial[:, -1, :]  # (batch, vocab)

    # So sánh vị trí đầu tiên (dự đoán token thứ 1)
    total_mean = 0.0
    worst = 0.0
    argmax_err = 0

    ref = logits_full[:, 0, :]
    diff = (logits_step - ref).abs()
    mean_d = diff.mean().item()
    max_d = diff.max().item()
    total_mean += mean_d
    worst = max(worst, max_d)
    if not torch.equal(logits_step.argmax(-1), ref.argmax(-1)):
        argmax_err += 1
    print(f"step 0 (BOS) mean={mean_d:.6f} max={max_d:.6f}")

    # Bước từng token còn lại
    for i in range(1, tgt_len):
        cur_token = dec_full[:, i]          # token đầu vào thứ i
        logits_step, kv_cache = model.step(
            cur_token, context, enc_attention_mask, kv_cache
        )
        ref = logits_full[:, i, :]          # logits dự đoán token i+1
        diff = (logits_step - ref).abs()

        mean_d = diff.mean().item()
        max_d = diff.max().item()
        total_mean += mean_d
        worst = max(worst, max_d)

        if not torch.equal(logits_step.argmax(-1), ref.argmax(-1)):
            argmax_err += 1

        print(f"step {i} mean={mean_d:.6f} max={max_d:.6f}")

    steps = tgt_len
    print("summary")
    print("mean", total_mean / steps)
    print("worst", worst)
    print("argmax_err", argmax_err)

    assert total_mean / steps < 1e-5
    assert argmax_err == 0


def test_generate_forward():
    print("test_generate_forward")

    model, cfg = build_model(test_generate=True)  # eos = -1
    device = cfg.device

    batch_size = 4
    src_len = 32
    max_new = 32
    vocab_size = cfg.vocab_size

    enc_input_ids = torch.randint(3, vocab_size, (batch_size, src_len), device=device)
    enc_attention_mask = torch.ones(batch_size, src_len, device=device, dtype=torch.bool)

    gen_ids = model.generate(enc_input_ids, max_new_tokens=max_new)
    # gen_ids shape: (batch, max_new + 1) vì có BOS ban đầu
    assert gen_ids.shape == (batch_size, max_new + 1)

    # forward toàn bộ với chuỗi đã sinh
    logits_full, _, _ = model.forward(enc_input_ids, enc_attention_mask, gen_ids)
    # logits_full: (batch, max_new+1, vocab)

    mismatch = 0
    for i in range(max_new):
        pred = logits_full[:, i, :].argmax(-1)   # dự đoán token ở vị trí i+1
        true = gen_ids[:, i + 1]
        mismatch += (pred != true).sum().item()

        if i % 8 == 0:
            print(f"step {i} mismatch {mismatch}")

    print(f"final mismatch {mismatch}")
    assert mismatch == 0


if __name__ == "__main__":
    test_forward_step()
    test_generate_forward()
    test_beam_search()
    print("all tests passed")

def test_beam_search():
    print("test_beam_search")

    model, cfg = build_model(test_generate=False)
    device = cfg.device

    batch_size = 4
    src_len = 32
    vocab_size = cfg.vocab_size
    beam_size = 3
    max_new = 16

    enc_input_ids = torch.randint(3, vocab_size, (batch_size, src_len), device=device)

    gen_ids = model.beam_search(enc_input_ids, max_new_tokens=max_new, beam_size=beam_size)

    assert gen_ids.shape[0] == batch_size, "batch size mismatch"
    assert gen_ids.shape[1] <= max_new + 1 + 1, "output too long"  # +1 BOS, +1 EOS
    assert gen_ids[:, 0].eq(cfg.bos_token_id).all(), "first token must be BOS"

    # Beam score >= greedy score (beam search should be at least as good)
    greedy_ids = model.generate(enc_input_ids, max_new_tokens=max_new)
    print(f"Beam output shape:   {gen_ids.shape}")
    print(f"Greedy output shape: {greedy_ids.shape}")
    print("test_beam_search passed")
