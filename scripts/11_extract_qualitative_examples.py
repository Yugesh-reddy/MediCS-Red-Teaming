#!/usr/bin/env python3
"""
Extract qualitative example pairs for the paper appendix.

Finds prompts where the base model produced a harmful response and the DPO
checkpoint produced a safe one (the "defense worked" story), stratified by
harm category so the appendix shows breadth of coverage.

Usage:
    python scripts/11_extract_qualitative_examples.py \
        --seed 42 --per-category 2 --output docs/qualitative_examples.md

Requires Cell 22 to have run (both base and dpo held_out.jsonl files judged).
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medics.utils import load_jsonl


def _load_and_index(ckpt: str, seed: int):
    path = Path(f"results/eval/{ckpt}/seed_{seed}/held_out.jsonl")
    if not path.exists():
        return {}
    return {
        (r.get("seed_id", ""), r.get("language", "")): r
        for r in load_jsonl(str(path))
    }


def _truncate(text: str, limit: int = 400) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-category", type=int, default=2,
                        help="Examples per harm category")
    parser.add_argument("--output", default="docs/qualitative_examples.md")
    args = parser.parse_args()

    base = _load_and_index("base", args.seed)
    dpo = _load_and_index("dpo", args.seed)

    if not base or not dpo:
        print("ERROR: Missing judged held-out files. Run Cell 22 first.")
        sys.exit(1)

    # Find base-harmful → DPO-safe pairs, bucketed by category
    by_category = defaultdict(list)
    for key, base_row in base.items():
        if base_row.get("judge_label") != "harmful":
            continue
        dpo_row = dpo.get(key)
        if not dpo_row or dpo_row.get("judge_label") != "safe":
            continue
        category = base_row.get("category", "?")
        by_category[category].append((base_row, dpo_row))

    if not by_category:
        print("WARNING: No base-harmful / DPO-safe pairs found. "
              "Check judging ran successfully.")
        sys.exit(1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Qualitative Examples",
        "",
        f"Paired responses from seed {args.seed} held-out evaluation, ",
        f"{args.per_category} example(s) per harm category.",
        "Each pair shows the same attack prompt producing a harmful base ",
        "response vs a safe DPO response.",
        "",
    ]

    for cat in sorted(by_category):
        pairs = by_category[cat][: args.per_category]
        lines.append(f"## {cat} — {len(by_category[cat])} pair(s) available")
        lines.append("")
        for i, (b, d) in enumerate(pairs, 1):
            lines += [
                f"### {cat} example {i}",
                "",
                f"- **Language:** {b.get('language', '?')}",
                f"- **Strategy:** {b.get('strategy', '?')}",
                f"- **Seed ID:** `{b.get('seed_id', '')}`",
                "",
                "**Original intent (English):**",
                "",
                f"> {_truncate(b.get('original_prompt', ''))}",
                "",
                "**Attack prompt (code-switched):**",
                "",
                f"> {_truncate(b.get('attack_prompt', ''))}",
                "",
                f"**Base response (judge: {b.get('judge_label')}):**",
                "",
                f"> {_truncate(b.get('model_response', ''), limit=600)}",
                "",
                f"**DPO response (judge: {d.get('judge_label')}):**",
                "",
                f"> {_truncate(d.get('model_response', ''), limit=600)}",
                "",
                "---",
                "",
            ]

    out_path.write_text("\n".join(lines))
    total_pairs = sum(len(v) for v in by_category.values())
    print(f"Extracted {total_pairs} base-harmful / DPO-safe pairs "
          f"across {len(by_category)} categories")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
