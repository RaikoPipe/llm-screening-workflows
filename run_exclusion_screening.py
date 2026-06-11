"""Entry point for exclusion screening — screening_instructions_v6 compliant.

Usage
-----
# Stage 3 (title/abstract) screening:
python run_exclusion_screening.py --input literature.csv --output screened.csv --stage 3

# Stage 2 pilot validation (requires r1_decision column in input CSV):
python run_exclusion_screening.py --input pilot_sample.csv --output pilot_out.csv --pilot
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig

from src.agent.graph_screening import State as ScreeningState
from src.agent.graph_screening import graph as graph_screening

load_dotenv()


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
) -> dict:
    """Run the exclusion screening pipeline (screening_instructions_v6).

    The PICOC scope, 12 exclusion codes, and 13 boundary rules are embedded in the
    system prompt inside graph_screening.py — no external criteria dict required.

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
# Pilot validation (Stage 2) — §7 of screening_instructions_v3
# ---------------------------------------------------------------------------

async def run_pilot_validation(
    input_path: str,
    output_path: str = "pilot_results.csv",
    audit_dir: str = ".",
    r1_decision_col: str = "r1_decision",
    model_name: str = "gpt-oss:120b",
    temperature: float = 0.2,
) -> None:
    """Run Stage 2 pilot validation and compute inter-rater agreement metrics.

    Computes sensitivity, specificity, Gwet's AC1, and Cohen's κ comparing
    LLM decisions against R1 ground-truth labels from the input CSV.

    Thresholds per §7: sensitivity ≥ 0.95, specificity ≥ 0.70, AC1 ≥ 0.80.
    Cohen's κ is computed and reported for reference but is NOT a threshold criterion.

    Args:
        input_path: Labelled CSV with a column named r1_decision_col
            containing R1 ground-truth decisions (Include / Exclude / Maybe).
        output_path: CSV with appended LLM screening columns.
        audit_dir: Directory for the markdown audit file.
        r1_decision_col: Column name in input CSV with R1 labels.
        model_name: LLM model name.
        temperature: Sampling temperature.
    """
    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for pilot validation. "
            "Install it with: pip install scikit-learn"
        ) from exc

    import pandas as pd

    df = pd.read_csv(input_path)
    if r1_decision_col not in df.columns:
        raise ValueError(
            f"Column '{r1_decision_col}' not found in {input_path}. "
            f"Available columns: {list(df.columns)}"
        )

    batch_id = "pilot_" + datetime.now(timezone.utc).strftime("%Y%m%d") + "_" + __import__("uuid").uuid4().hex[:6]

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
        return

    r1 = out_df[r1_decision_col].fillna("Maybe").tolist()
    llm = out_df["decision"].fillna("Maybe").tolist()

    # Binarise for sensitivity/specificity: {Include, Maybe} → positive; {Exclude} → negative
    def _binarise(decisions: list) -> list:
        return ["Include" if d != "Exclude" else "Exclude" for d in decisions]

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

    # Thresholds per §7: sensitivity ≥ 0.95, specificity ≥ 0.70, AC1 ≥ 0.80.
    # Cohen's κ is reported only — not a threshold criterion.
    sens_ok = sensitivity >= 0.95
    spec_ok = specificity >= 0.70
    ac1_ok = ac1 >= 0.80
    passed = sens_ok and spec_ok and ac1_ok

    print("\n" + "=" * 60)
    print(f"Pilot Validation Metrics — batch {batch_id}")
    print("=" * 60)
    print(f"  Records    : {len(r1)}")
    print(f"  Sensitivity: {sensitivity:.3f}  {'✓ ≥0.95' if sens_ok else '✗ <0.95 FAIL'}")
    print(f"  Specificity: {specificity:.3f}  {'✓ ≥0.70' if spec_ok else '✗ <0.70 FAIL'}")
    print(f"  Gwet's AC1 : {ac1:.3f}  {'✓ ≥0.80' if ac1_ok else '✗ <0.80 FAIL'}")
    print(f"  Cohen's κ  : {kappa:.3f}  (reported only — not a threshold criterion)")
    print(f"  Confusion  : TP={tp} FN={fn} TN={tn} FP={fp}")
    print(f"  RESULT     : {'PASS — proceed to Stage 3' if passed else 'FAIL — refine prompt, re-run on fresh sample'}")
    print("=" * 60 + "\n")

    # Append metrics to audit markdown
    audit_path = f"{audit_dir}/screening_audit_{batch_id}.md"
    metrics_block = [
        "",
        "## Pilot validation metrics (§7)",
        "",
        "| Metric | Value | Threshold | Pass |",
        "|--------|-------|-----------|------|",
        f"| Sensitivity | {sensitivity:.3f} | ≥ 0.95 | {'✓' if sens_ok else '✗'} |",
        f"| Specificity | {specificity:.3f} | ≥ 0.70 | {'✓' if spec_ok else '✗'} |",
        f"| Gwet's AC1  | {ac1:.3f} | ≥ 0.80 | {'✓' if ac1_ok else '✗'} |",
        f"| Cohen's κ   | {kappa:.3f} | — (reported only) | — |",
        "",
        f"**Binarisation:** {{Include, Maybe}} → positive; {{Exclude}} → negative  ",
        f"**Confusion:** TP={tp} FN={fn} TN={tn} FP={fp}  ",
        f"**Verdict:** {'PASS' if passed else 'FAIL — refine prompt and re-run on a fresh 100-record sample'}",
        "",
    ]
    try:
        with open(audit_path, "a", encoding="utf-8") as f:
            f.write("\n".join(metrics_block) + "\n")
    except FileNotFoundError:
        pass


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
            "Run Stage 2 pilot validation. Input CSV must have an r1_decision column "
            "with R1 ground-truth labels. Computes sensitivity, specificity, Gwet's AC1 "
            "(threshold ≥ 0.80), and Cohen's κ (reported only)."
        ),
    )
    parser.add_argument("--r1-col", default="r1_decision",
                        help="Column name for R1 ground-truth labels (--pilot mode only).")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()

    if args.pilot:
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
        ))
