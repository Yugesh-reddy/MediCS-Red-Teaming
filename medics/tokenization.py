"""
MediCS — Tokenization Analysis
================================
Explains WHY code-switched attacks work by analyzing token fragmentation.
Low-resource language text fragments into many more tokens than English,
disrupting the model's ability to recognize harmful content.

No GPU needed — loads tokenizer only.
"""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np


def _tokenize_and_measure(tokenizer, text):
    """
    Tokenize text and compute basic measurements.

    Returns:
        dict with token_count, tokens (list of str), unique_token_pct, avg_token_len
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    n = len(tokens)
    if n == 0:
        return {
            "token_count": 0,
            "tokens": [],
            "unique_token_pct": 0.0,
            "avg_token_len": 0.0,
        }
    return {
        "token_count": n,
        "tokens": tokens,
        "unique_token_pct": round(len(set(tokens)) / n, 4),
        "avg_token_len": round(np.mean([len(t) for t in tokens]), 2),
    }


def _keyword_fragmentation(tokenizer, english_kw, translated_kw):
    """
    Compare tokenization of an English keyword vs its translation.

    Returns:
        dict with keyword, en_tokens, translated_tokens, ratio, en_token_list, translated_token_list
    """
    en_ids = tokenizer.encode(english_kw, add_special_tokens=False)
    tr_ids = tokenizer.encode(translated_kw, add_special_tokens=False)
    en_count = len(en_ids)
    tr_count = len(tr_ids)
    return {
        "keyword": english_kw,
        "translated": translated_kw,
        "en_tokens": en_count,
        "translated_tokens": tr_count,
        "ratio": round(tr_count / en_count, 2) if en_count > 0 else 0.0,
        "en_token_list": tokenizer.convert_ids_to_tokens(en_ids),
        "translated_token_list": tokenizer.convert_ids_to_tokens(tr_ids),
    }


def compute_oov_proxy(tokenizer, text):
    """
    Estimate OOV-like token rate: fraction of tokens that are single-character
    or byte-fallback tokens (indicating the tokenizer doesn't know the word).

    For BPE tokenizers (like Llama), byte-fallback tokens appear as single bytes
    like '<0xE0>', '<0xA4>', etc.

    Returns:
        float: fraction of tokens that look like byte fallbacks (0.0 to 1.0)
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    if not tokens:
        return 0.0
    byte_fallback = sum(
        1 for t in tokens
        if (t.startswith("<0x") and t.endswith(">"))  # byte fallback
        or (len(t.replace("▁", "")) <= 1)  # single-char after removing SentencePiece prefix
    )
    return round(byte_fallback / len(tokens), 4)


def analyze_tokenization(tokenizer_name_or_path, seeds, keywords, languages,
                          translate_fn=None, max_seeds=0):
    """
    For each seed prompt, tokenize the English version and each code-switched
    version. Compute token count ratio, fragmentation metrics, and per-keyword analysis.

    Args:
        tokenizer_name_or_path: HuggingFace model ID or local path (loads tokenizer only)
        seeds: list of seed dicts with 'prompt', 'seed_id', 'category'
        keywords: dict mapping seed_id -> list of keyword strings
        languages: list of language code strings (e.g., ["hi", "bn", "sw"])
        translate_fn: optional function(text, lang) -> translated_text.
                      If None, imports translate_with_fallback from medics.utils.
        max_seeds: max number of seeds to analyze (0 = all)

    Returns:
        list of dicts with per-seed, per-language tokenization metrics
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    if translate_fn is None:
        from medics.utils import translate_with_fallback
        translate_fn = translate_with_fallback

    if max_seeds > 0:
        seeds = seeds[:max_seeds]

    results = []
    for i, seed in enumerate(seeds):
        seed_id = seed.get("seed_id", f"seed_{i}")
        prompt = seed.get("prompt", "")
        category = seed.get("category", "unknown")
        seed_keywords = keywords.get(seed_id, [])

        # English baseline
        en_metrics = _tokenize_and_measure(tokenizer, prompt)

        for lang in languages:
            # Translate the full prompt
            try:
                translated_prompt = translate_fn(prompt, lang)
            except Exception:
                translated_prompt = prompt  # fallback to English

            cs_metrics = _tokenize_and_measure(tokenizer, translated_prompt)

            # Keyword-level fragmentation
            kw_frag = []
            for kw in seed_keywords:
                try:
                    translated_kw = translate_fn(kw, lang)
                    frag = _keyword_fragmentation(tokenizer, kw, translated_kw)
                    kw_frag.append(frag)
                except Exception:
                    continue

            en_count = en_metrics["token_count"]
            cs_count = cs_metrics["token_count"]

            results.append({
                "seed_id": seed_id,
                "category": category,
                "language": lang,
                "en_token_count": en_count,
                "cs_token_count": cs_count,
                "token_count_ratio": round(cs_count / en_count, 4) if en_count > 0 else 0.0,
                "en_unique_token_pct": en_metrics["unique_token_pct"],
                "cs_unique_token_pct": cs_metrics["unique_token_pct"],
                "oov_proxy_rate": compute_oov_proxy(tokenizer, translated_prompt),
                "keyword_fragmentation": kw_frag,
                "avg_kw_ratio": round(
                    np.mean([f["ratio"] for f in kw_frag]), 2
                ) if kw_frag else 0.0,
            })

        if (i + 1) % 50 == 0:
            print(f"  Tokenization analysis: {i+1}/{len(seeds)} seeds processed")

    return results


def compute_fragmentation_summary(analysis_results):
    """
    Aggregate tokenization results by language.

    Returns:
        dict: {language: {mean_ratio, median_ratio, max_ratio, mean_oov, n_seeds}}
    """
    by_lang = defaultdict(list)
    for r in analysis_results:
        lang = r["language"]
        by_lang[lang].append(r)

    summary = {}
    for lang, entries in sorted(by_lang.items()):
        ratios = [e["token_count_ratio"] for e in entries if e["token_count_ratio"] > 0]
        oov_rates = [e["oov_proxy_rate"] for e in entries]
        kw_ratios = [e["avg_kw_ratio"] for e in entries if e["avg_kw_ratio"] > 0]

        summary[lang] = {
            "mean_ratio": round(float(np.mean(ratios)), 3) if ratios else 0.0,
            "median_ratio": round(float(np.median(ratios)), 3) if ratios else 0.0,
            "max_ratio": round(float(np.max(ratios)), 3) if ratios else 0.0,
            "std_ratio": round(float(np.std(ratios)), 3) if ratios else 0.0,
            "mean_oov_proxy": round(float(np.mean(oov_rates)), 4) if oov_rates else 0.0,
            "mean_kw_ratio": round(float(np.mean(kw_ratios)), 3) if kw_ratios else 0.0,
            "n_seeds": len(entries),
        }

    return summary
