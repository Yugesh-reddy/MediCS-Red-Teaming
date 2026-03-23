"""
MediCS — Perplexity-Based Detection Baseline
==============================================
Can code-switched attacks be trivially detected by input perplexity?

If YES (high AUROC): a simple filter could catch CS attacks, but this doesn't
    fix the underlying vulnerability — the model still complies if the filter
    is bypassed. Weight-level defense (SFT+DPO) addresses the root cause.

If NO (low AUROC): CS attacks are indistinguishable from normal multilingual
    input, making detection-based defense impractical. This strengthens the
    case for weight-level defense.

GPU required for compute_perplexity_batch() — runs in Colab.
perplexity_detector() and detection_by_language() are CPU-only analysis.
"""

import numpy as np
from collections import defaultdict


def compute_perplexity_batch(model, tokenizer, texts, batch_size=8, max_length=512):
    """
    Compute per-input perplexity using the target model.

    Args:
        model: loaded HuggingFace model (e.g., 4-bit quantized Llama-3)
        tokenizer: corresponding tokenizer
        texts: list of input text strings
        batch_size: inference batch size
        max_length: max sequence length for tokenization

    Returns:
        list of perplexity values (float), one per input
    """
    import torch

    model.eval()
    perplexities = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encodings = tokenizer(
            batch_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        )
        input_ids = encodings["input_ids"].to(model.device)
        attention_mask = encodings["attention_mask"].to(model.device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()

        # Per-token cross-entropy loss
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size())

        # Masked mean per sample
        for j in range(loss.size(0)):
            mask = shift_mask[j].bool()
            if mask.sum() > 0:
                avg_loss = loss[j][mask].mean().item()
                ppl = np.exp(min(avg_loss, 100))  # cap to avoid overflow
            else:
                ppl = float("inf")
            perplexities.append(round(ppl, 2))

        if (i + batch_size) % (batch_size * 10) == 0:
            print(f"  Perplexity: {min(i + batch_size, len(texts))}/{len(texts)}")

    return perplexities


def perplexity_detector(english_ppls, cs_ppls, thresholds=None):
    """
    Evaluate a threshold-based perplexity detector for CS attacks.

    Labels: English inputs = negative (0), CS inputs = positive (1).
    A higher-perplexity input is predicted as CS.

    Args:
        english_ppls: list of perplexity values for English-only inputs
        cs_ppls: list of perplexity values for code-switched inputs
        thresholds: optional list of thresholds to evaluate.
                    If None, uses percentiles of combined distribution.

    Returns:
        dict with best_threshold, best_f1, auroc, accuracy, precision, recall,
        per_threshold metrics, and ROC data (fpr, tpr).
    """
    all_ppls = english_ppls + cs_ppls
    all_labels = [0] * len(english_ppls) + [1] * len(cs_ppls)

    ppls = np.array(all_ppls)
    labels = np.array(all_labels)

    # Filter out inf/nan
    valid = np.isfinite(ppls)
    ppls = ppls[valid]
    labels = labels[valid]

    if len(ppls) == 0:
        return {"best_f1": 0.0, "auroc": 0.5, "error": "no valid perplexities"}

    if thresholds is None:
        thresholds = np.percentile(ppls, np.arange(5, 100, 5)).tolist()

    per_threshold = []
    best_f1 = 0.0
    best_thresh = 0.0

    for thresh in thresholds:
        preds = (ppls >= thresh).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        tn = np.sum((preds == 0) & (labels == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(labels)

        per_threshold.append({
            "threshold": round(thresh, 2),
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        })

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    # Compute ROC curve
    sorted_thresholds = np.sort(np.unique(ppls))
    fpr_list = [0.0]
    tpr_list = [0.0]
    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)

    for t in sorted_thresholds:
        preds = (ppls >= t).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        tpr_list.append(tp / n_pos if n_pos > 0 else 0.0)
        fpr_list.append(fp / n_neg if n_neg > 0 else 0.0)

    fpr_list.append(1.0)
    tpr_list.append(1.0)

    # Sort by FPR for proper ROC
    sorted_idx = np.argsort(fpr_list)
    fpr_sorted = [fpr_list[i] for i in sorted_idx]
    tpr_sorted = [tpr_list[i] for i in sorted_idx]

    # AUROC via trapezoidal rule
    auroc = float(np.trapz(tpr_sorted, fpr_sorted))

    # Best threshold metrics
    best_preds = (ppls >= best_thresh).astype(int)
    best_tp = np.sum((best_preds == 1) & (labels == 1))
    best_fp = np.sum((best_preds == 1) & (labels == 0))
    best_fn = np.sum((best_preds == 0) & (labels == 1))
    best_tn = np.sum((best_preds == 0) & (labels == 0))

    return {
        "best_threshold": round(best_thresh, 2),
        "best_f1": round(best_f1, 4),
        "auroc": round(auroc, 4),
        "accuracy": round((best_tp + best_tn) / len(labels), 4),
        "precision": round(best_tp / (best_tp + best_fp), 4) if (best_tp + best_fp) > 0 else 0.0,
        "recall": round(best_tp / (best_tp + best_fn), 4) if (best_tp + best_fn) > 0 else 0.0,
        "n_english": len(english_ppls),
        "n_cs": len(cs_ppls),
        "mean_ppl_english": round(float(np.mean([p for p in english_ppls if np.isfinite(p)])), 2),
        "mean_ppl_cs": round(float(np.mean([p for p in cs_ppls if np.isfinite(p)])), 2),
        "per_threshold": per_threshold,
        "roc": {
            "fpr": [round(f, 4) for f in fpr_sorted],
            "tpr": [round(t, 4) for t in tpr_sorted],
        },
    }


def detection_by_language(perplexities):
    """
    Per-language detection performance.

    Args:
        perplexities: list of dicts with 'language', 'perplexity', 'is_cs' (bool)

    Returns:
        dict: {lang: {mean_ppl_en, mean_ppl_cs, best_f1, n_samples}}
    """
    by_lang = defaultdict(lambda: {"en": [], "cs": []})
    for entry in perplexities:
        lang = entry.get("language", "unknown")
        ppl = entry.get("perplexity", 0.0)
        if not np.isfinite(ppl):
            continue
        if entry.get("is_cs", True):
            by_lang[lang]["cs"].append(ppl)
        else:
            by_lang[lang]["en"].append(ppl)

    results = {}
    for lang, data in sorted(by_lang.items()):
        if data["en"] and data["cs"]:
            det = perplexity_detector(data["en"], data["cs"])
            results[lang] = {
                "mean_ppl_en": round(float(np.mean(data["en"])), 2),
                "mean_ppl_cs": round(float(np.mean(data["cs"])), 2),
                "best_f1": det["best_f1"],
                "auroc": det["auroc"],
                "n_en": len(data["en"]),
                "n_cs": len(data["cs"]),
            }

    return results
