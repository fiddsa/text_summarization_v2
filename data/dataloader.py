import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from data.tokenizer import Tokenizer
from data.dataset import auto_dataset
import config

def seq2seq_collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    target_ids = [item["target_ids"] for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=config.PAD_ID)
    target_ids = pad_sequence(target_ids, batch_first=True, padding_value=config.PAD_ID)
    attention_mask = (input_ids != config.PAD_ID)

    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "attention_mask": attention_mask
    }

def auto_dataloader(tokenizer: Tokenizer, mode="train"):
    dataset = auto_dataset(tokenizer, mode)
    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=(mode == "train"),
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        collate_fn=seq2seq_collate_fn
    )