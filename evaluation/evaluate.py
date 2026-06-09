import torch
from tqdm import tqdm
import pandas as pd

from data.tokenizer import Tokenizer
from data.dataloader import auto_dataloader
from modeling.models import auto_model
import config
from evaluation.metrics import compute_rouge, compute_bleu, compute_bertscore

def _generate_preds_seq2seq(model, tokenizer, data_loader, device):
    all_inputs = []
    all_preds = []
    all_refs = []

    for batch in tqdm(data_loader, desc="Test"):
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)

        seq_ids = model.generate(input_ids, config.MAX_NEW_TOKENS).cpu()
        input_ids = input_ids.cpu()
        target_ids = target_ids.cpu()

        for input, pred, tgt in zip(input_ids, seq_ids, target_ids):
            input = input.tolist()
            pred = pred.tolist()
            tgt = tgt.tolist()

            input_text = tokenizer.decode(input)
            pred_text = tokenizer.decode(pred)
            tgt_text = tokenizer.decode(tgt)

            all_inputs.append(input_text)
            all_preds.append(pred_text)
            all_refs.append(tgt_text)
    
    return all_inputs, all_preds, all_refs

def _generate_preds():
    tokenizer = Tokenizer()
    test_loader = auto_dataloader(tokenizer, "test")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = auto_model().to(device)
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    return _generate_preds_seq2seq(model, tokenizer, test_loader, device)

def _write_preds(all_inputs, all_preds, all_refs):
    df = pd.DataFrame({
        "source": all_inputs,
        "target": all_refs,
        "prediction": all_preds,
    })
    df.to_csv(config.PREDS_PATH, index=False)

def evaluate():
    all_inputs, all_preds, all_refs = _generate_preds()
    _write_preds(all_inputs, all_preds, all_refs)

    results = {}

    results.update(compute_rouge(all_preds, all_refs))
    results.update(compute_bleu(all_preds, all_refs))
    results.update(compute_bertscore(all_preds, all_refs))

    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")
    
if __name__ == "__main__":
    evaluate()
