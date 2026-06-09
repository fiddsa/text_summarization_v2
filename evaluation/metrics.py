from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from bert_score import score as bert_score
import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


# ──────────────────────────────────────────────
# ROUGE
# ──────────────────────────────────────────────

def compute_rouge(preds, refs):
    """
    Returns:
        dict with keys rouge1, rouge2, rougeL (macro-averaged F1)
    """
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )

    scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for pred, ref in zip(preds, refs):
        s = scorer.score(ref, pred)
        for k in scores:
            scores[k].append(s[k].fmeasure)

    return {k: sum(v) / len(v) for k, v in scores.items()}


# ──────────────────────────────────────────────
# BLEU
# ──────────────────────────────────────────────

def compute_bleu(preds, refs):
    """
    Corpus-level BLEU-1/2/3/4 với add-1 smoothing (Chen & Cherry 2014).

    Returns:
        dict with keys bleu1, bleu2, bleu3, bleu4
    """
    smooth = SmoothingFunction().method1

    # nltk corpus_bleu: refs = list of list of list of tokens
    #                   hyps = list of list of tokens
    tokenized_refs = [[nltk.word_tokenize(r.lower())] for r in refs]
    tokenized_preds = [nltk.word_tokenize(p.lower()) for p in preds]

    results = {}
    for n in range(1, 5):
        weights = tuple(1.0 / n if i < n else 0.0 for i in range(4))
        results[f"bleu{n}"] = corpus_bleu(
            tokenized_refs, tokenized_preds,
            weights=weights,
            smoothing_function=smooth,
        )

    return results


# ──────────────────────────────────────────────
# BERTScore
# ──────────────────────────────────────────────

def compute_bertscore(preds, refs, lang="en", device=None):
    """
    Macro-averaged BERTScore P / R / F1 using microsoft/deberta-xlarge-mnli
    (default model for English in bert-score >= 0.3.12).

    Args:
        preds:  list of prediction strings
        refs:   list of reference strings
        lang:   language code passed to bert_score (default "en")
        device: "cuda" / "cpu" / None (auto-detect)

    Returns:
        dict with keys bertscore_p, bertscore_r, bertscore_f1
    """
    P, R, F1 = bert_score(
        preds, refs,
        lang=lang,
        device=device,
        verbose=False,
    )
    return {
        "bertscore_p":  P.mean().item(),
        "bertscore_r":  R.mean().item(),
        "bertscore_f1": F1.mean().item(),
    }
