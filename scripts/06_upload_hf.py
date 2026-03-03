#!/usr/bin/env python3
"""
Upload MediCS-500 dataset and LoRA adapters to HuggingFace.

Usage:
  python scripts/06_upload_hf.py --dataset  # Upload dataset only
  python scripts/06_upload_hf.py --adapters  # Upload LoRA adapters
  python scripts/06_upload_hf.py --all       # Upload everything
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medics.utils import load_config


def upload_dataset(repo_id, data_dir="data/medics_500"):
    """Upload MediCS-500 dataset to HuggingFace."""
    from huggingface_hub import HfApi, create_repo

    api = HfApi()

    # Create dataset repo
    try:
        create_repo(repo_id, repo_type="dataset", exist_ok=True)
        print(f"Dataset repo: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"Repo creation: {e}")

    # Upload files
    data_path = Path(data_dir)
    files_to_upload = [
        (data_path / "medics_500_full.jsonl", "medics_500_full.jsonl"),
        (data_path / "semantic_scores.json", "semantic_scores.json"),
        (data_path / "dataset_card.md", "README.md"),
    ]

    # Also upload splits
    splits_dir = Path("data/splits")
    if splits_dir.exists():
        files_to_upload.extend([
            (splits_dir / "attack_pool.jsonl", "splits/attack_pool.jsonl"),
            (splits_dir / "held_out_eval.jsonl", "splits/held_out_eval.jsonl"),
        ])

    for local_path, repo_path in files_to_upload:
        if local_path.exists():
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type="dataset",
            )
            print(f"  Uploaded: {repo_path}")
        else:
            print(f"  Skipped (not found): {local_path}")


def upload_adapters(repo_id, checkpoints_dir="checkpoints"):
    """Upload LoRA adapters to HuggingFace."""
    from huggingface_hub import HfApi, create_repo

    api = HfApi()

    try:
        create_repo(repo_id, repo_type="model", exist_ok=True)
        print(f"Model repo: https://huggingface.co/models/{repo_id}")
    except Exception as e:
        print(f"Repo creation: {e}")

    ckpt_dir = Path(checkpoints_dir)
    adapter_dirs = [
        ("sft", ckpt_dir / "sft" / "round_3" / "final"),
        ("dpo", ckpt_dir / "dpo" / "final"),
    ]

    for name, adapter_path in adapter_dirs:
        if adapter_path.exists():
            api.upload_folder(
                folder_path=str(adapter_path),
                path_in_repo=f"adapters/{name}",
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"  Uploaded adapter: {name}")
        else:
            print(f"  Skipped (not found): {adapter_path}")


def create_dataset_card():
    """Generate a dataset card for HuggingFace."""
    card = """---
language:
  - en
  - hi
  - bn
  - sw
  - yo
  - tl
  - gu
license: cc-by-nc-4.0
task_categories:
  - text-classification
tags:
  - medical
  - safety
  - red-teaming
  - code-switching
  - adversarial
size_categories:
  - 1K<n<10K
---

# MediCS-500: Medical Code-Switching Red-Teaming Benchmark

## Dataset Description

MediCS-500 is a curated benchmark for evaluating medical AI safety against
code-switching adversarial attacks. It contains 500 harmful medical prompts
across 6 categories, expanded to ~3,000 semantically verified code-switched
variants in 6 languages.

### Categories (500 seeds)
| Code | Category | Count |
|------|----------|-------|
| TOX | Toxicology / Drug Synthesis | 84 |
| SH | Self-Harm / Suicide Methods | 83 |
| MIS | Medical Misinformation | 83 |
| ULP | Unlicensed Medical Practice | 84 |
| PPV | Patient Privacy Violations | 83 |
| UCA | Unethical Clinical Advice | 83 |

### Languages (6 verified)
Hindi, Bengali, Swahili, Yoruba, Tagalog, Gujarati

### Features
- Back-translation semantic verification (MiniLM cosine similarity ≥ 0.75)
- 500 benign twin prompts for over-refusal calibration
- Stratified 80/20 train/eval splits
- Per-prompt keyword extraction

## Citation
```bibtex
@misc{medics2026,
  title={MediCS: Agentic Adversarial Training for Medical AI Safety},
  author={Sappidi, Yugesh and Yeole, Yash},
  year={2026},
}
```
"""
    output_path = Path("data/medics_500/dataset_card.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(card)
    print(f"Dataset card saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Upload to HuggingFace")
    parser.add_argument("--dataset", action="store_true", help="Upload dataset")
    parser.add_argument("--adapters", action="store_true", help="Upload adapters")
    parser.add_argument("--all", action="store_true", help="Upload everything")
    parser.add_argument("--dataset-repo", default="MediCS/MediCS-500")
    parser.add_argument("--model-repo", default="MediCS/llama3-8b-medics-safety")
    parser.add_argument("--create-card", action="store_true",
                        help="Generate dataset card")
    args = parser.parse_args()

    if args.create_card or args.all:
        create_dataset_card()

    if args.dataset or args.all:
        upload_dataset(args.dataset_repo)

    if args.adapters or args.all:
        upload_adapters(args.model_repo)

    if not (args.dataset or args.adapters or args.all or args.create_card):
        print("No action specified. Use --dataset, --adapters, --all, or --create-card")


if __name__ == "__main__":
    main()
