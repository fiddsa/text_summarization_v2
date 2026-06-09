import torch
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

import config


def get_warmup_scheduler(optimizer, warmup_steps):
    """
    Linear warmup scheduler:
    - LR increases linearly from 0 to LEARNING_RATE over `warmup_steps` steps.
    - LR remains constant at LEARNING_RATE after warmup.
    """
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


class Trainer:
    def __init__(self, model, train_loader, dev_loader, optimizer, criterion, scheduler=None):
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        self.best_dev_loss = float("inf")
        self.train_losses = []
        self.dev_losses = []
        self.start_epoch = 1
        self.global_step = 0

        if config.RESUME_TRAINING:
            checkpoint = torch.load(config.LAST_CHECKPOINT_PATH, map_location=self.device)
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.train_losses = checkpoint["train_losses"]
            self.dev_losses = checkpoint["dev_losses"]
            self.best_dev_loss = min(self.dev_losses)
            self.start_epoch = len(self.train_losses) + 1
            self.global_step = checkpoint.get("global_step", 0)
            if self.scheduler is not None and "scheduler" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler"])

    def _train_one_epoch(self):
        pass

    @torch.no_grad()
    def _eval(self):
        pass

    def train(self):
        for epoch in range(self.start_epoch, self.start_epoch + config.NUM_EPOCHS):
            print("=" * 10 + f" Epoch {epoch} " + "=" * 10)

            train_loss = self._train_one_epoch()
            dev_loss = self._eval()
            self.train_losses.append(train_loss)
            self.dev_losses.append(dev_loss)
            print(f"Train loss: {train_loss:.4f}")
            print(f"Dev loss: {dev_loss:.4f}")
            if self.scheduler is not None:
                print(f"Current LR: {self.scheduler.get_last_lr()[0]:.6f}")

            if dev_loss < self.best_dev_loss:
                self.best_dev_loss = dev_loss
                torch.save(self.model.state_dict(), config.BEST_MODEL_PATH)
                print(">>> Save best model")

            checkpoint = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "train_losses": self.train_losses,
                "dev_losses": self.dev_losses,
                "global_step": self.global_step,
            }
            if self.scheduler is not None:
                checkpoint["scheduler"] = self.scheduler.state_dict()
            torch.save(checkpoint, config.LAST_CHECKPOINT_PATH)


class Seq2SeqTrainer(Trainer):
    def _train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        for batch in tqdm(self.train_loader, desc="Train"):
            self.optimizer.zero_grad()

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            target_ids = batch["target_ids"].to(self.device)

            decoder_input = target_ids[:, :-1]
            labels = target_ids[:, 1:]

            logits, _, _ = self.model(input_ids, attention_mask, decoder_input)

            loss = self.criterion(logits.view(-1, config.VOCAB_SIZE), labels.reshape(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Step the warmup scheduler after each optimizer update
            if self.scheduler is not None:
                self.scheduler.step()

            self.global_step += 1
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def _eval(self):
        self.model.eval()
        total_loss = 0.0
        for batch in tqdm(self.dev_loader, desc="Eval"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            target_ids = batch["target_ids"].to(self.device)

            decoder_input = target_ids[:, :-1]
            labels = target_ids[:, 1:]

            logits, _, _ = self.model(input_ids, attention_mask, decoder_input)

            loss = self.criterion(logits.view(-1, config.VOCAB_SIZE), labels.reshape(-1))

            total_loss += loss.item()

        return total_loss / len(self.dev_loader)


def auto_trainer(model, train_loader, dev_loader, optimizer, criterion, scheduler=None):
    return Seq2SeqTrainer(model, train_loader, dev_loader, optimizer, criterion, scheduler)
