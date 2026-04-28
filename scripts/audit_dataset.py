#!/usr/bin/env python3
"""Audit the preprocessed UD sentence CSV used by the notebooks.

The notebooks consume one row per sentence, with token-level columns stored as
Python-list strings. This script checks that those lists are aligned, prints the
core dataset counts, and summarizes the linguistic feature distributions.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


DEFAULT_CSV = Path("code/data/en_ewt-ud-train_sentences.csv")
LIST_COLUMNS = ("tokens", "index", "length", "pos", "head_dist", "relation_type", "ariety")
FEATURE_COLUMNS = ("index", "length", "pos", "head_dist", "relation_type", "ariety")


def parse_list(value: Any, *, column: str, row_id: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        raise ValueError(f"Row {row_id!r}, column {column!r}: expected list string, got {type(value).__name__}")
    try:
        parsed = ast.literal_eval(value)
    except Exception as exc:  # pragma: no cover - message is the important part
        raise ValueError(f"Row {row_id!r}, column {column!r}: cannot parse list string") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"Row {row_id!r}, column {column!r}: expected parsed list, got {type(parsed).__name__}")
    return parsed


def flatten(values: Iterable[list[Any]]) -> list[Any]:
    out: list[Any] = []
    for value in values:
        out.extend(value)
    return out


def value_counts(values: Iterable[Any]) -> list[dict[str, Any]]:
    counter = Counter(values)
    return [
        {"value": value, "count": count}
        for value, count in sorted(counter.items(), key=lambda item: (-item[1], str(item[0])))
    ]


def audit(csv_path: Path) -> dict[str, Any]:
    df = pd.read_csv(csv_path)
    missing = [column for column in ("sentence_id", *LIST_COLUMNS) if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    parsed: dict[str, list[list[Any]]] = {column: [] for column in LIST_COLUMNS}
    bad_rows: list[dict[str, Any]] = []

    for row_number, row in df.iterrows():
        row_id = row.get("sentence_id", row_number)
        row_lists = {
            column: parse_list(row[column], column=column, row_id=row_id)
            for column in LIST_COLUMNS
        }
        lengths = {column: len(values) for column, values in row_lists.items()}
        token_count = lengths["tokens"]
        if any(length != token_count for length in lengths.values()):
            bad_rows.append({"sentence_id": row_id, "lengths": lengths})
        for column, values in row_lists.items():
            parsed[column].append(values)

    tokens = flatten(parsed["tokens"])
    features = {
        column: value_counts(flatten(parsed[column]))
        for column in FEATURE_COLUMNS
    }

    return {
        "csv": str(csv_path),
        "sentences": int(len(df)),
        "tokens": int(len(tokens)),
        "unique_token_types": int(len(set(tokens))),
        "list_length_mismatch_rows": bad_rows,
        "features": features,
    }


def print_audit(report: dict[str, Any], *, top_k: int) -> None:
    print(f"CSV: {report['csv']}")
    print(f"sentences: {report['sentences']}")
    print(f"tokens: {report['tokens']}")
    print(f"unique token types: {report['unique_token_types']}")

    mismatches = report["list_length_mismatch_rows"]
    if mismatches:
        print(f"WARNING: {len(mismatches)} rows have misaligned token/feature list lengths.")
        for item in mismatches[:10]:
            print(f"  {item['sentence_id']}: {item['lengths']}")
    else:
        print("list length alignment: OK")

    for feature, counts in report["features"].items():
        print(f"\n{feature}: {len(counts)} classes")
        for item in counts[:top_k]:
            print(f"  {item['value']}: {item['count']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Preprocessed sentence CSV to audit.")
    parser.add_argument("--top-k", type=int, default=15, help="Number of feature counts to print.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path for a full JSON audit report.")
    args = parser.parse_args()

    report = audit(args.csv)
    print_audit(report, top_k=args.top_k)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(f"\nwrote JSON audit: {args.json_out}")


if __name__ == "__main__":
    main()
