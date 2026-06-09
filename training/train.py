import torch
import argparse

from modeling.models import auto_model
from data.tokenizer import Tokenizer
from data.dataloader import auto_dataloader
from training.trainer import auto_trainer, get_warmup_scheduler
import config

def train():
    tokenizer = Tokenizer()
    train_loader = auto_dataloader(tokenizer, mode="train")
    dev_loader = auto_dataloader(tokenizer, mode="dev")
    model = auto_model()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    scheduler = get_warmup_scheduler(optimizer, warmup_steps=config.WARMUP_STEPS)

    trainer = auto_trainer(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
    )

    trainer.train()

if __name__ == "__main__":
    train()
