# MediCS -- API Cost Tracking
# Update this file after every experiment or API-calling script execution.

---

## Budget

| Item | Value |
|------|-------|
| Total budget | $10.00 |
| Spent to date | $0.00 |
| Remaining | $10.00 |

---

## Pricing Reference (as of March 2026)

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|-----------------------|------------------------|
| gpt-4o | $2.50 | $10.00 |
| gpt-4o-mini | $0.15 | $0.60 |

---

## Cost Estimates Per Pipeline Step

| Step | Script | Model | Est. Cost |
|------|--------|-------|-----------|
| Keyword extraction (500 seeds) | `01_build_dataset.py` | gpt-4o-mini | ~$0.05 |
| Dataset validation (1000 items) | `01_build_dataset.py` | gpt-4o-mini | ~$0.08 |
| MTE turn generation (~40/round) | `02_run_attack_round.py` | gpt-4o | ~$0.15/round |
| Judge calls (150/round) | `02_run_attack_round.py` | gpt-4o | ~$0.45/round |
| Refusal generation (~100/round) | `03_build_defense_data.py` | gpt-4o | ~$0.30/round |
| Helpful response gen (~100/round) | `03_build_defense_data.py` | gpt-4o | ~$0.30/round |
| Failure mode classification | `04_evaluate.py` | gpt-4o | ~$0.60 |
| **Estimated per round** | | | **~$1.20** |
| **Estimated 3 rounds total** | | | **~$4.33** |
| **Estimated 5 rounds total** | | | **~$5.90** |

---

## Actual Spend Log

Record each API-calling execution below. Include the date, which script was run, how many API calls were made, and the actual cost from the OpenAI dashboard.

| Date | Script | Task | API Calls | Actual Cost | Running Total |
|------|--------|------|-----------|-------------|---------------|
| -- | -- | No experiments run yet | -- | -- | $0.00 |

---

## Notes

- Translation via `deep-translator` is free (no API key, no cost).
- Back-translation verification uses `sentence-transformers` MiniLM (free, CPU).
- Seeds and benign twins were generated manually (no API cost).
- DPO training data reuses existing responses and refusals ($0 additional API).
- If budget runs tight, reduce `attacks_per_round` from 150 to 100 in `config/experiment_config.yaml`.
- Use `gpt-4o-mini` wherever possible for mechanical tasks (keyword extraction, validation).
- The `medics/utils.py` module supports translation and response caching to avoid duplicate API calls.
