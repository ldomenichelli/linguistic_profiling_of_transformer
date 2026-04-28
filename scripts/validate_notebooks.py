#!/usr/bin/env python3
"""Validate notebooks without running the expensive model experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import nbformat


SLOW_ISOSCORE_PATTERNS = (
    "return float(IsoScore.IsoScore(X))",
    "return float(_IsoScore_fn(torch.from_numpy(X)))",
    "return float(_IsoScore_fn(X))",
)


def code_cells(notebook: dict[str, Any]) -> list[dict[str, Any]]:
    return [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]


def validate_notebook(path: Path, *, execute_setup: bool) -> list[str]:
    errors: list[str] = []

    try:
        nb_node = nbformat.read(path, as_version=4)
        nbformat.validate(nb_node)
    except Exception as exc:
        return [f"{path}: invalid notebook schema: {type(exc).__name__}: {exc}"]

    notebook = json.loads(path.read_text())
    cells = code_cells(notebook)
    source = "\n".join("".join(cell.get("source", [])) for cell in cells)

    for index, cell in enumerate(cells):
        cell_source = "".join(cell.get("source", []))
        try:
            compile(cell_source, f"{path}:code_cell_{index}", "exec")
        except SyntaxError as exc:
            errors.append(f"{path}: code cell {index} does not compile: {exc}")

        if cell.get("outputs"):
            errors.append(f"{path}: code cell {index} still has stored outputs")
        if cell.get("execution_count") is not None:
            errors.append(f"{path}: code cell {index} still has execution_count={cell.get('execution_count')}")

    for pattern in SLOW_ISOSCORE_PATTERNS:
        if pattern in source:
            errors.append(f"{path}: slow direct IsoScore call remains: {pattern}")

    if "def _find_project_root" not in source or "_resolve_csv_path" not in source:
        errors.append(f"{path}: missing repo path helpers")

    if execute_setup and cells:
        namespace: dict[str, Any] = {"__name__": "__notebook_setup_validation__"}
        try:
            exec('import matplotlib; matplotlib.use("Agg")', namespace)
            exec("".join(cells[0].get("source", [])), namespace)
        except Exception as exc:
            errors.append(f"{path}: setup cell failed: {type(exc).__name__}: {exc}")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "notebooks",
        nargs="*",
        type=Path,
        default=sorted(Path("code").glob("*.ipynb")),
        help="Notebook files to validate. Defaults to code/*.ipynb.",
    )
    parser.add_argument(
        "--execute-setup",
        action="store_true",
        help="Execute the first code cell of each notebook to catch import/path errors.",
    )
    args = parser.parse_args()

    all_errors: list[str] = []
    for path in args.notebooks:
        errors = validate_notebook(path, execute_setup=args.execute_setup)
        if errors:
            all_errors.extend(errors)
        else:
            print(f"{path}: OK")

    if all_errors:
        print("\nValidation errors:")
        for error in all_errors:
            print(f"- {error}")
        raise SystemExit(1)

    print(f"\nvalidated {len(args.notebooks)} notebooks")


if __name__ == "__main__":
    main()
