import config
from modeling.models.seq2seq import Seq2Seq, Seq2SeqConfig

def auto_model():
    return Seq2Seq(Seq2SeqConfig(
        vocab_size=config.VOCAB_SIZE,
        pad_token_id=config.PAD_ID,
        bos_token_id=config.BOS_ID,
        eos_token_id=config.EOS_ID,
        model_dim=config.MODEL_DIM,
        head_dim=config.HEAD_DIM,
        expansion_factor=config.EXPANSION_FACTOR,
        num_layers=config.NUM_LAYERS,
    )).to("cuda")