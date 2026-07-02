"""Command-line entry point for drugnlp.

Examples
--------
  python -m drugnlp.cli info   --data data/sample_reviews.csv
  python -m drugnlp.cli train  --train drugsComTrain_raw.tsv --test drugsComTest_raw.tsv
  python -m drugnlp.cli themes --data drugsComTrain_raw.tsv --top 8
"""
from __future__ import annotations

import argparse
import json
import sys

from .load import dataset_summary, load_reviews


def _cmd_info(args: argparse.Namespace) -> None:
    df = load_reviews(args.data)
    print(json.dumps(dataset_summary(df), indent=2))


def _cmd_train(args: argparse.Namespace) -> None:
    from .model import run
    tr, te = load_reviews(args.train), load_reviews(args.test)
    res = run(tr, te, out_dir=args.out)
    b = res["majority_baseline"]
    print(f"majority baseline    : acc {b['accuracy']:.3f}  macroF1 {b['macro_f1']:.3f}  "
          f"auc {b['roc_auc']:.2f}  (predicts {b['predicts']})")
    for key in ("text_only", "structured_only", "text_plus_structured"):
        m = res[key]
        print(f"{key:21s}: acc {m['accuracy']:.3f}  macroF1 {m['macro_f1']:.3f}  "
              f"auc {m['roc_auc']:.3f}")
    print(f"\nWrote {args.out}/sentiment_metrics.json")


def _cmd_themes(args: argparse.Namespace) -> None:
    from .themes import distinctive_negative_terms, satisfaction_by_condition
    df = load_reviews(args.data)
    print("Lowest-satisfaction conditions:")
    print(satisfaction_by_condition(df).to_string())
    print("\nDistinctive negative-review terms:")
    for cond, terms in distinctive_negative_terms(df, top_conditions=args.top).items():
        print(f"- {cond}: {', '.join(terms)}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="drugnlp")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("info", help="print dataset summary")
    pi.add_argument("--data", required=True)
    pi.set_defaults(func=_cmd_info)

    pt = sub.add_parser("train", help="train and evaluate sentiment models")
    pt.add_argument("--train", required=True)
    pt.add_argument("--test", required=True)
    pt.add_argument("--out", default="results")
    pt.set_defaults(func=_cmd_train)

    pm = sub.add_parser("themes", help="complaint terms and low-satisfaction conditions")
    pm.add_argument("--data", required=True)
    pm.add_argument("--top", type=int, default=8)
    pm.set_defaults(func=_cmd_themes)

    args = p.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
