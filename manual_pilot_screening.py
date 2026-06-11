"""Manual pilot screening tool for SLR exclusion decisions.

Usage
-----
python manual_pilot_screening.py --input literature.csv --output pilot_sample.csv
python manual_pilot_screening.py --input literature.csv --output pilot_sample.csv --n 50 --seed 99
python manual_pilot_screening.py --input literature.csv --output pilot_sample.csv --resume
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Exclusion codes (mirrored from screening_instructions_v6)
# ---------------------------------------------------------------------------

EXCLUSION_CODES: dict[int, str] = {
    1:  "Operational/tactical scope only (scheduling, MES/ERP, shopfloor, PM, QC, S&OP) — no physical reconfiguration.",
    2:  "No LLM / foundation model / generative AI / agentic AI / RAG / LLM-hybrid in contribution.",
    3:  "Does not address any of the five strategic subdomains (factory planning, SC network design, strategic logistics, tech-selection, GPN).",
    4:  "AI/ML without LLM component (pure RL, classical ML, metaheuristics, MILP).",
    5:  "Non-manufacturing/logistics domain (healthcare, finance, retail SC without production network).",
    6:  "Grey literature (tier not meeting inclusion threshold).",
    7:  "Publication year < 2022.",
    8:  "Language not English or German.",
    9:  "Editorial / opinion / non-methodological commentary.",
    10: "Secondary literature (SLR, narrative review, scoping review, meta-analysis).",
    11: "Duplicate record.",
    12: "Full text unavailable.",
}

INCLUDE_LABEL  = "Include"
EXCLUDE_LABEL  = "Exclude"

DECISION_COL   = "r1_decision"
EXCL_CODE_COL  = "r1_excl_code"
NOTE_COL       = "r1_note"
SCREENED_COL   = "_screened"          # internal progress tracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clear() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def _divider(char: str = "─", width: int = 72) -> str:
    return char * width


def _print_record(idx: int, total: int, row: pd.Series) -> None:
    """Pretty-print a single record for the reviewer."""
    _clear()
    print(_divider("═"))
    print(f"  Record {idx} / {total}")
    print(_divider("═"))

    title   = str(row.get("title", "")).strip() or "(no title)"
    doi    = str(row.get("doi", "")).strip()
    authors = str(row.get("author", row.get("authors", row.get("Author", "")))).strip() or "(no authors)"
    institution = str(row.get("institutions", "")).strip() or "(no institution)"
    abstract = str(row.get("abstractNote", row.get("abstract", ""))).strip() or "(no abstract)"
    year    = str(row.get("publication_year", "")).strip()
    venue   = str(row.get("source_name", row.get("venue", ""))).strip()

    print(f"\nTitle   : {title}")
    print(f"\nDOI     : {doi}" if doi else "\nDOI     : (no DOI)")
    if year:
        print(f"Year    : {year}")
    print(f"Venue   : {venue}" if venue else "\nVenue   : (no Venue)")
    print(f"Authors : {authors}")
    print(f"Institution: {institution}")
    print(f"\nAbstract:\n{abstract}")
    print()


def _ask_decision() -> str:
    """Prompt for Include / Exclude / Skip."""
    while True:
        raw = input("Decision  [i]nclude / [e]xclude / [s]kip (decide later) / [q]uit : ").strip().lower()
        if raw in ("i", "e", "s", "q"):
            return raw
        print("  → Please enter i, e, s, or q.")


def _ask_excl_code() -> int | None:
    """Print exclusion codes and ask for one."""
    print()
    print("Exclusion codes:")
    for code, desc in EXCLUSION_CODES.items():
        print(f"  {code:>2}  {desc}")
    print()
    while True:
        raw = input("Exclusion code (1–12) : ").strip()
        if raw.isdigit() and int(raw) in EXCLUSION_CODES:
            return int(raw)
        print(f"  → Enter a number between 1 and 12.")


def _ask_escalation_note() -> str:
    """Ask whether an escalation note should be added; return note text or empty string."""
    while True:
        flag = input("Add escalation note? [y]es / [n]o : ").strip().lower()
        if flag == "n":
            return ""
        if flag == "y":
            note = input("Escalation note : ").strip()
            return f"ESCALATE: {note}" if note else ""
        print("  → Please enter y or n.")


# ---------------------------------------------------------------------------
# Core screening loop
# ---------------------------------------------------------------------------

def run_screening(df: pd.DataFrame, output_path: Path, resume: bool) -> None:
    """Iterate over unscreened records and collect decisions."""

    # Ensure tracking columns exist
    if DECISION_COL not in df.columns:
        df[DECISION_COL] = pd.NA
    if EXCL_CODE_COL not in df.columns:
        df[EXCL_CODE_COL] = pd.NA
    if NOTE_COL not in df.columns:
        df[NOTE_COL] = None
    df[NOTE_COL] = df[NOTE_COL].astype(object)
    if SCREENED_COL not in df.columns:
        df[SCREENED_COL] = False

    # Determine records still to screen
    if resume:
        todo_mask = ~df[SCREENED_COL].astype(bool)
    else:
        todo_mask = pd.Series([True] * len(df), index=df.index)
        df[DECISION_COL] = pd.NA
        df[EXCL_CODE_COL] = pd.NA
        df[NOTE_COL]      = None
        df[SCREENED_COL]  = False

    todo_indices = df.index[todo_mask].tolist()
    total        = len(todo_indices)

    if total == 0:
        print("All records already screened. Use --resume=False to restart.")
        return

    print(f"\nStarting manual screening — {total} record(s) to review.")
    print("Your progress is auto-saved after every decision.\n")
    input("Press Enter to begin…")

    completed = 0

    for position, idx in enumerate(todo_indices, start=1):
        row = df.loc[idx]
        _print_record(position, total, row)

        choice = _ask_decision()

        if choice == "q":
            print("\nScreening paused. Run with --resume to continue.")
            _save(df, output_path)
            sys.exit(0)

        if choice == "s":
            # Leave decision as NA; mark not screened so it reappears on resume
            df.at[idx, SCREENED_COL] = False
            _save(df, output_path)
            completed += 1
            continue

        if choice == "i":
            df.at[idx, DECISION_COL]  = INCLUDE_LABEL
            df.at[idx, EXCL_CODE_COL] = pd.NA
            df.at[idx, NOTE_COL]      = _ask_escalation_note() or None
        else:  # "e"
            code = _ask_excl_code()
            df.at[idx, DECISION_COL]  = EXCLUDE_LABEL
            df.at[idx, EXCL_CODE_COL] = code
            df.at[idx, NOTE_COL]      = _ask_escalation_note() or None

        df.at[idx, SCREENED_COL] = True
        _save(df, output_path)
        completed += 1

    _clear()
    print(_divider("═"))
    print("  Screening complete!")
    print(_divider("═"))
    _print_summary(df)
    print(f"\nResults saved to: {output_path}")


def _save(df: pd.DataFrame, path: Path) -> None:
    """Write current state to CSV (drop internal progress column)."""
    out = df.drop(columns=[SCREENED_COL], errors="ignore")
    out.to_csv(path, index=False)


def _print_summary(df: pd.DataFrame) -> None:
    screened = df[df[SCREENED_COL].astype(bool)]
    included = (screened[DECISION_COL] == INCLUDE_LABEL).sum()
    excluded = (screened[DECISION_COL] == EXCLUDE_LABEL).sum()
    skipped  = (~screened[SCREENED_COL].astype(bool)).sum() if SCREENED_COL in screened else 0

    print(f"\nSummary  ({len(screened)} screened of {len(df)} sampled)")
    print(f"  Include : {included}")
    print(f"  Exclude : {excluded}")

    if excluded > 0:
        code_counts = screened[EXCL_CODE_COL].dropna().astype(int).value_counts().sort_index()
        for code, count in code_counts.items():
            print(f"    Code {code:>2}: {count}  — {EXCLUSION_CODES.get(int(code), '')[:60]}")

    unscreened = (~df[SCREENED_COL].astype(bool)).sum()
    if unscreened:
        print(f"  Skipped / pending : {unscreened}  (re-run with --resume to finish)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Manual pilot screening — select N random records and collect R1 decisions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",  dest="input_path",  required=True,
                   help="Input CSV with bibliographic data.")
    p.add_argument("--output", dest="output_path", default="pilot_sample.csv",
                   help="Output CSV with r1_decision and r1_excl_code columns appended.")
    p.add_argument("--n",      type=int, default=100,
                   help="Number of records to sample.")
    p.add_argument("--seed",   type=int, default=42,
                   help="Random seed for reproducibility.")
    p.add_argument("--resume", action="store_true",
                   help=(
                       "Resume an interrupted session from --output file. "
                       "Skips already-screened records."
                   ))
    return p


def main() -> None:
    args = _build_parser().parse_args()

    output_path = Path(args.output_path)

    if args.resume and output_path.exists():
        print(f"Resuming from {output_path} …")
        df = pd.read_csv(output_path)
        # Re-add internal tracking column if absent (first resume after plain save)
        if SCREENED_COL not in df.columns:
            df[SCREENED_COL] = (
                df[DECISION_COL].notna() if DECISION_COL in df.columns else False
            )
    else:
        source = pd.read_csv(args.input_path)
        n = min(args.n, len(source))
        if n < args.n:
            print(f"Warning: only {len(source)} records available; sampling all of them.")
        df = source.sample(n=n, random_state=args.seed).reset_index(drop=True)
        print(f"Sampled {n} records from {args.input_path} (seed={args.seed}).")

    run_screening(df, output_path, resume=args.resume)


if __name__ == "__main__":
    main()