"""Entry point for exclusion screening — screening_instructions_v6 compliant.

Usage
-----
# Stage 3 (title/abstract) screening:
python run_exclusion_screening.py --input literature.csv --output screened.csv --stage 3

# Stage 2 pilot validation (runs the graph, then scores; requires r1_decision column):
python run_exclusion_screening.py --input pilot_sample.csv --output pilot_out.csv --pilot

# Recompute pilot metrics ONLY, no graph (CSV must already have decision + r1_decision):
python run_exclusion_screening.py --recompute --input screening_results_pilot.csv

# Update an existing results CSV with new manual labels, then recompute metrics (no graph):
python run_exclusion_screening.py --update \
    --results screening_results_pilot.csv --sample pilot_sample.csv \
    --output screening_results_pilot.csv --key-col record_id

# Load literature + fulltext from Zotero via collection key (alternative to --input CSV):
python run_exclusion_screening.py --collection-key UF8TVRYZ --output screened.csv --stage 3
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig

from src.agent.graph_screening import State as ScreeningState
from src.agent.graph_screening import graph as graph_screening

load_dotenv()


def _new_batch_id(prefix: str = "pilot") -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d')}_{uuid.uuid4().hex[:6]}"


# ---------------------------------------------------------------------------
# Main screening entry point
# ---------------------------------------------------------------------------

async def run_exclusion_screening(
    input_path: Optional[str] = None,
    output_path: str = "screening_results.csv",
    audit_dir: str = ".",
    stage: int = 3,
    batch_id: Optional[str] = None,
    literature_items=None,
    model_name: str = "gpt-oss:120b",
    temperature: float = 0.0,
    max_output_tokens: int = 16000,
    max_fulltext_words: int = 12000,
    screening_type: str = "abstract",
    collection_key: Optional[str] = None,
) -> dict:
    """Run the exclusion screening pipeline (screening_instructions_v6).

    The PICOC scope, all 12 exclusion codes, and 13 boundary rules are embedded
    in the system prompt inside graph_screening.py — no external criteria dict
    required. Codes 6–8, 11, 12 are additionally applied as a metadata
    pre-filter fast-path when the relevant flags/fields are present.

    Args:
        input_path: Path to input CSV with bibliographic data. Screening columns
            are appended to this CSV and saved to output_path.
        output_path: Destination CSV (original bibliographic columns +
            appended screening columns per §9).
        audit_dir: Directory for the markdown audit file.
        stage: Screening stage number (2 = pilot, 3 = title/abstract).
        batch_id: Optional run identifier. Auto-generated if None.
        literature_items: Pre-loaded pd.DataFrame or List[LiteratureItem].
            Ignored when input_path is provided.
        model_name: LLM model name served via Ollama.
        temperature: Sampling temperature (default 0.2 for screening).
        max_output_tokens: Max tokens per LLM response.
        max_fulltext_words: Truncate fulltext to this many words before screening.
        screening_type: Text source for LLM screening — "abstract" (default) or "fulltext".
            When "fulltext", the full paper text is used if available, falling back to abstract.
        collection_key: Zotero collection key to load literature and fulltext from Zotero
            (alternative to input_path). Ignored when input_path or literature_items is
            provided. Falls back to the ZOTERO_COLLECTION_KEY env var when None.

    Returns:
        Final graph state dict.
    """
    state_kwargs: dict = {
        "output_path": output_path,
        "audit_dir": audit_dir,
        "stage": stage,
        "screening_type": screening_type,
    }
    if input_path:
        state_kwargs["input_path"] = input_path
    if literature_items is not None:
        state_kwargs["literature_items"] = literature_items
    if batch_id:
        state_kwargs["batch_id"] = batch_id
    if collection_key:
        state_kwargs["collection_key"] = collection_key

    initial_state = ScreeningState(**state_kwargs)

    config = RunnableConfig(
        configurable={
            "model_name": model_name,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "max_fulltext_words": max_fulltext_words,
        }
    )

    result = await graph_screening.ainvoke(initial_state, config=config)
    return result


# ---------------------------------------------------------------------------
# Inter-rater agreement metrics — pure, graph-free (§7)
# ---------------------------------------------------------------------------

def _binarise(decisions: list) -> list:
    """{Include, Maybe} → positive ('Include'); {Exclude} → negative ('Exclude')."""
    return ["Include" if d != "Exclude" else "Exclude" for d in decisions]


def compute_pilot_metrics(
    df,
    r1_decision_col: str = "r1_decision",
    decision_col: str = "decision",
) -> dict:
    """Compute Stage 2 inter-rater agreement metrics from a scored DataFrame.

    Pure function: no graph, no I/O. The DataFrame must already contain both the
    R1 ground-truth column and the LLM decision column.

    Thresholds per §7: sensitivity ≥ 0.95, specificity ≥ 0.70, AC1 ≥ 0.80.
    Cohen's κ is computed and reported for reference but is NOT a threshold criterion.

    Returns:
        dict of metric values, pass flags, confusion counts, and overall `passed`.
    """
    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for pilot metrics. "
            "Install it with: pip install scikit-learn"
        ) from exc

    for col in (r1_decision_col, decision_col):
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found. Available columns: {list(df.columns)}"
            )

    r1 = df[r1_decision_col].fillna("Maybe").tolist()
    llm = df[decision_col].fillna("Maybe").tolist()

    r1_bin = _binarise(r1)
    llm_bin = _binarise(llm)

    tp = sum(r == "Include" and l == "Include" for r, l in zip(r1_bin, llm_bin))
    fn = sum(r == "Include" and l == "Exclude" for r, l in zip(r1_bin, llm_bin))
    tn = sum(r == "Exclude" and l == "Exclude" for r, l in zip(r1_bin, llm_bin))
    fp = sum(r == "Exclude" and l == "Include" for r, l in zip(r1_bin, llm_bin))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    kappa = cohen_kappa_score(r1_bin, llm_bin)
    ac1 = _gwet_ac1(r1_bin, llm_bin)

    sens_ok = sensitivity >= 0.95
    spec_ok = specificity >= 0.70
    ac1_ok = ac1 >= 0.80
    kappa_ok = kappa >= 0.70

    return {
        "n": len(r1),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ac1": ac1,
        "kappa": kappa,
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        "sens_ok": sens_ok, "spec_ok": spec_ok,
        "ac1_ok": ac1_ok, "kappa_ok": kappa_ok,
        "passed": sens_ok and spec_ok and ac1_ok,
    }


def print_pilot_metrics(m: dict, batch_id: str) -> None:
    """Print the metrics block to stdout."""
    print("\n" + "=" * 60)
    print(f"Pilot Validation Metrics — batch {batch_id}")
    print("=" * 60)
    print(f"  Records    : {m['n']}")
    print(f"  Sensitivity: {m['sensitivity']:.3f}  {'✓ ≥0.95' if m['sens_ok'] else '✗ <0.95 FAIL'}")
    print(f"  Specificity: {m['specificity']:.3f}  {'✓ ≥0.70' if m['spec_ok'] else '✗ <0.70 FAIL'}")
    print(f"  Gwet's AC1 : {m['ac1']:.3f}  {'✓ ≥0.80' if m['ac1_ok'] else '✗ <0.80 FAIL'}")
    print(f"  Cohen's κ  : {m['kappa']:.3f}  {'✓ ≥0.70' if m['kappa_ok'] else '✗ <0.70 FAIL'}")
    print(f"  Confusion  : TP={m['tp']} FN={m['fn']} TN={m['tn']} FP={m['fp']}")
    print(f"  RESULT     : {'PASS — proceed to Stage 3' if m['passed'] else 'FAIL — refine prompt, re-run on fresh sample'}")
    print("=" * 60 + "\n")


def write_pilot_audit(m: dict, audit_dir: str, batch_id: str) -> None:
    """Append the metrics block to the batch's markdown audit file."""
    audit_path = f"{audit_dir}/screening_audit_{batch_id}.md"
    block = [
        "",
        "## Pilot validation metrics (§7)",
        "",
        "| Metric | Value | Threshold | Pass |",
        "|--------|-------|-----------|------|",
        f"| Sensitivity | {m['sensitivity']:.3f} | ≥ 0.95 | {'✓' if m['sens_ok'] else '✗'} |",
        f"| Specificity | {m['specificity']:.3f} | ≥ 0.70 | {'✓' if m['spec_ok'] else '✗'} |",
        f"| Gwet's AC1  | {m['ac1']:.3f} | ≥ 0.80 | {'✓' if m['ac1_ok'] else '✗'} |",
        f"| Cohen's κ   | {m['kappa']:.3f} | (reference) | {'✓' if m['kappa_ok'] else '✗'} |",
        "",
        "**Binarisation:** {Include, Maybe} → positive; {Exclude} → negative  ",
        f"**Confusion:** TP={m['tp']} FN={m['fn']} TN={m['tn']} FP={m['fp']}  ",
        f"**Verdict:** {'PASS' if m['passed'] else 'FAIL — refine prompt and re-run on a fresh 100-record sample'}",
        "",
    ]
    try:
        with open(audit_path, "a", encoding="utf-8") as f:
            f.write("\n".join(block) + "\n")
    except FileNotFoundError:
        pass


def score_pilot_csv(
    scored_path: str,
    audit_dir: str = ".",
    r1_decision_col: str = "r1_decision",
    decision_col: str = "decision",
    batch_id: Optional[str] = None,
) -> dict:
    """Load an already-scored CSV and (re)compute pilot metrics — NO graph call.

    Args:
        scored_path: CSV containing both `decision_col` (LLM) and `r1_decision_col` (R1).
        audit_dir: Directory for the markdown audit file.
        r1_decision_col: R1 ground-truth column name.
        decision_col: LLM decision column name.
        batch_id: Optional run identifier for reporting; auto-generated if None.

    Returns:
        The metrics dict from `compute_pilot_metrics`.
    """
    import pandas as pd

    batch_id = batch_id or _new_batch_id("recompute")
    df = pd.read_csv(scored_path)
    m = compute_pilot_metrics(df, r1_decision_col=r1_decision_col, decision_col=decision_col)
    print_pilot_metrics(m, batch_id)
    write_pilot_audit(m, audit_dir, batch_id)
    return m


# ---------------------------------------------------------------------------
# Update existing results with new manual labels, then recompute — NO graph
# ---------------------------------------------------------------------------

def update_pilot_results(
    results_path: str,
    sample_path: str,
    output_path: str,
    key_col: str = "record_id",
    r1_decision_col: str = "r1_decision",
    decision_col: str = "decision",
    audit_dir: str = ".",
    batch_id: Optional[str] = None,
) -> dict:
    """Merge new manual R1 labels into an existing results CSV, then recompute metrics.

    Joins `sample_path` onto `results_path` on `key_col`, overwriting/adding the R1
    ground-truth column from the (newer) manual sample where keys match. The LLM
    `decision_col` already in `results_path` is reused — the graph is NOT re-run.

    Args:
        results_path: Existing screening results CSV (must contain `decision_col`).
        sample_path: New manual pilot CSV (must contain `key_col` + `r1_decision_col`).
        output_path: Where the merged CSV is written (may equal results_path to overwrite).
        key_col: Join key present in both files (e.g. record_id / doi / title).
        r1_decision_col: Manual R1 label column carried over from the sample.
        decision_col: LLM decision column already present in the results CSV.
        audit_dir: Directory for the markdown audit file.
        batch_id: Optional run identifier; auto-generated if None.

    Returns:
        The metrics dict from `compute_pilot_metrics`.
    """
    import pandas as pd

    batch_id = batch_id or _new_batch_id("pilot_update")

    results_df = pd.read_csv(results_path)
    sample_df = pd.read_csv(sample_path)

    for col, path in ((key_col, results_path), (decision_col, results_path)):
        if col not in results_df.columns:
            raise ValueError(f"Column '{col}' not found in {path}. "
                             f"Available: {list(results_df.columns)}")
    for col in (key_col, r1_decision_col):
        if col not in sample_df.columns:
            raise ValueError(f"Column '{col}' not found in {sample_path}. "
                             f"Available: {list(sample_df.columns)}")

    # Left-join the manual labels; prefer the new sample label where present.
    new_col = f"{r1_decision_col}__new"
    merged = results_df.merge(
        sample_df[[key_col, r1_decision_col]].rename(columns={r1_decision_col: new_col}),
        on=key_col,
        how="left",
    )
    if r1_decision_col in merged.columns:
        merged[r1_decision_col] = merged[new_col].combine_first(merged[r1_decision_col])
    else:
        merged[r1_decision_col] = merged[new_col]
    merged = merged.drop(columns=[new_col])

    matched = int(merged[r1_decision_col].notna().sum())
    sample_keys = set(sample_df[key_col])
    result_keys = set(results_df[key_col])
    unmatched_sample = len(sample_keys - result_keys)

    merged.to_csv(output_path, index=False)
    print(f"Merged {matched} R1-labelled rows into {output_path} "
          f"({len(merged)} total rows; {unmatched_sample} sample keys not found in results).")

    scored = merged.dropna(subset=[r1_decision_col, decision_col])
    m = compute_pilot_metrics(scored, r1_decision_col=r1_decision_col, decision_col=decision_col)
    print_pilot_metrics(m, batch_id)
    write_pilot_audit(m, audit_dir, batch_id)
    return m


# ---------------------------------------------------------------------------
# Pilot validation (Stage 2) — runs the graph, then scores via the helpers above
# ---------------------------------------------------------------------------

async def run_pilot_validation(
    input_path: str,
    output_path: str = "pilot_results.csv",
    audit_dir: str = ".",
    r1_decision_col: str = "r1_decision",
    model_name: str = "gpt-oss:120b",
    temperature: float = 0.2,
) -> Optional[dict]:
    """Run Stage 2 pilot validation (graph) and compute inter-rater agreement metrics.

    This is now a thin wrapper: it runs the screening graph, then delegates scoring
    to `score_pilot_csv`. To re-score without the graph, call `score_pilot_csv` or
    `update_pilot_results` directly.

    Args:
        input_path: Labelled CSV with a column named r1_decision_col
            containing R1 ground-truth decisions (Include / Exclude / Maybe).
        output_path: CSV with appended LLM screening columns.
        audit_dir: Directory for the markdown audit file.
        r1_decision_col: Column name in input CSV with R1 labels.
        model_name: LLM model name.
        temperature: Sampling temperature.

    Returns:
        The metrics dict, or None if scoring could not run.
    """
    import pandas as pd

    df = pd.read_csv(input_path)
    if r1_decision_col not in df.columns:
        raise ValueError(
            f"Column '{r1_decision_col}' not found in {input_path}. "
            f"Available columns: {list(df.columns)}"
        )

    batch_id = _new_batch_id("pilot")

    await run_exclusion_screening(
        input_path=input_path,
        output_path=output_path,
        audit_dir=audit_dir,
        stage=2,
        batch_id=batch_id,
        model_name=model_name,
        temperature=temperature,
    )

    out_df = pd.read_csv(output_path)
    if "decision" not in out_df.columns:
        print("ERROR: 'decision' column not found in output. Metrics cannot be computed.")
        return None

    return score_pilot_csv(
        scored_path=output_path,
        audit_dir=audit_dir,
        r1_decision_col=r1_decision_col,
        decision_col="decision",
        batch_id=batch_id,
    )


def _gwet_ac1(r1: list, r2: list) -> float:
    """Gwet (2008) AC1 for two raters and K categories.

    Formula: AC1 = (p_o - p_e) / (1 - p_e)
    where p_e = (1 / (K-1)) * sum_k p_k * (1 - p_k)
    and p_k = average marginal proportion of category k across both raters.
    """
    n = len(r1)
    if n == 0:
        return float("nan")

    categories = sorted(set(r1) | set(r2))
    K = len(categories)
    if K < 2:
        return 1.0

    p_o = sum(a == b for a, b in zip(r1, r2)) / n

    p_e = 0.0
    for cat in categories:
        p_k = (r1.count(cat) + r2.count(cat)) / (2 * n)
        p_e += p_k * (1.0 - p_k)
    p_e /= (K - 1)

    if p_e >= 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run exclusion screening per screening_instructions_v6.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", dest="input_path", help="Input CSV with bibliographic data.")
    parser.add_argument("--collection-key", dest="collection_key", default=None,
                        help=(
                            "Zotero collection key to load literature and fulltext from Zotero "
                            "(alternative to --input CSV). Ignored when --input is provided. "
                            "Falls back to the ZOTERO_COLLECTION_KEY env var when unset."
                        ))
    parser.add_argument("--output", dest="output_path", default="screening_results.csv",
                        help="Output CSV (bibliographic columns + appended screening columns).")
    parser.add_argument("--audit-dir", default=".", help="Directory for markdown audit file.")
    parser.add_argument("--stage", type=int, default=3, choices=[2, 3],
                        help="Screening stage (2=pilot, 3=title/abstract).")
    parser.add_argument("--model", default="gpt-oss:120b", help="LLM model name (Ollama).")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=16000, dest="max_output_tokens")
    parser.add_argument("--max-words", type=int, default=12000, dest="max_fulltext_words")
    parser.add_argument("--batch-id", default=None, help="Override auto-generated batch ID.")
    parser.add_argument(
        "--screening-type", dest="screening_type", default="abstract", choices=["abstract", "fulltext"],
        help=(
            "Text source for LLM screening. 'abstract' uses title+abstract only (default). "
            "'fulltext' uses the full paper text when available, falling back to abstract."
        ),
    )
    parser.add_argument(
        "--pilot", action="store_true",
        help=(
            "Run Stage 2 pilot validation via the graph, then score. Input CSV must have an "
            "r1_decision column. Computes sensitivity, specificity, Gwet's AC1 (≥0.80), and κ."
        ),
    )
    parser.add_argument(
        "--recompute", action="store_true",
        help=(
            "Recompute pilot metrics only, NO graph. --input must already contain both the "
            "decision and r1_decision columns."
        ),
    )
    parser.add_argument(
        "--update", action="store_true",
        help=(
            "Merge new manual labels (--sample) into an existing results CSV (--results) on "
            "--key-col, write to --output, then recompute metrics. No graph call."
        ),
    )
    parser.add_argument("--results", dest="results_path", default=None,
                        help="Existing results CSV with LLM decisions (--update mode).")
    parser.add_argument("--sample", dest="sample_path", default=None,
                        help="New manual pilot CSV with R1 labels (--update mode).")
    parser.add_argument("--key-col", default="record_id",
                        help="Join key shared by results and sample CSVs (--update mode).")
    parser.add_argument("--r1-col", default="r1_decision",
                        help="Column name for R1 ground-truth labels (pilot / update / recompute).")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()

    if args.update:
        missing = [n for n, v in (("--results", args.results_path), ("--sample", args.sample_path)) if not v]
        if missing:
            print(f"ERROR: {', '.join(missing)} required for --update mode.", file=sys.stderr)
            sys.exit(1)
        # Default the merged output to the results file (in-place) if not given.
        out = args.output_path if args.output_path != "screening_results.csv" else args.results_path
        update_pilot_results(
            results_path=args.results_path,
            sample_path=args.sample_path,
            output_path=out,
            key_col=args.key_col,
            r1_decision_col=args.r1_col,
            audit_dir=args.audit_dir,
            batch_id=args.batch_id,
        )

    elif args.recompute:
        if not args.input_path:
            print("ERROR: --input is required for --recompute mode.", file=sys.stderr)
            sys.exit(1)
        score_pilot_csv(
            scored_path=args.input_path,
            audit_dir=args.audit_dir,
            r1_decision_col=args.r1_col,
            batch_id=args.batch_id,
        )

    elif args.pilot:
        if not args.input_path:
            print("ERROR: --input is required for --pilot mode.", file=sys.stderr)
            sys.exit(1)
        asyncio.run(run_pilot_validation(
            input_path=args.input_path,
            output_path=args.output_path,
            audit_dir=args.audit_dir,
            r1_decision_col=args.r1_col,
            model_name=args.model,
            temperature=args.temperature,
        ))

    else:
        asyncio.run(run_exclusion_screening(
            input_path=args.input_path,
            output_path=args.output_path,
            audit_dir=args.audit_dir,
            stage=args.stage,
            batch_id=args.batch_id,
            model_name=args.model,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            max_fulltext_words=args.max_fulltext_words,
            screening_type=args.screening_type,
            collection_key=args.collection_key,
        ))