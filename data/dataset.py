import torch
from torch.utils.data import Dataset
import pandas as pd

from data.tokenizer import Tokenizer
import config

class Seq2SeqDataset(Dataset):
    def __init__(self, dataset_path: str, tokenizer: Tokenizer):
        df = pd.read_csv(dataset_path)

        self.src_ids = tokenizer.encode(list(df["source"]), add_bos=False, add_eos=False)
        self.tgt_ids = tokenizer.encode(list(df["target"]), add_bos=True, add_eos=True)

    def __len__(self):
        return len(self.src_ids)

    def __getitem__(self, index):
        return {
            "input_ids": torch.tensor(self.src_ids[index], dtype=torch.long),
            "target_ids": torch.tensor(self.tgt_ids[index], dtype=torch.long)
        }

def auto_dataset(tokenizer: Tokenizer, mode="train"):
    if mode == "train":
        path = config.TRAIN_PATH
    elif mode == "dev":
        path = config.DEV_PATH
    else:
        path = config.TEST_PATH

    return Seq2SeqDataset(path, tokenizer)