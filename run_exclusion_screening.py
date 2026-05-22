"""Entry point for the exclusion screening pipeline (screening_instructions_v3).

Usage examples:

  # Full screening run against a Zotero collection:
  python run_exclusion_screening.py --model claude-sonnet-4-6

  # Screen from an existing CSV and append results in-place:
  python run_exclusion_screening.py --input-csv data/papers.csv --model claude-sonnet-4-6

  # Pilot validation (Stage 2): compare against R1 ground-truth labels:
  python run_exclusion_screening.py --input-csv data/pilot_sample.csv --pilot \\
      --pilot-gt-col r1_decision --stage 2
"""

import argparse
import asyncio

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig

from src.agent.graph_screening import EXCLUSION_CODES, PICOC_DEFINITION
from src.agent.graph_screening import State as ScreeningState
from src.agent.graph_screening import graph as graph_screening

load_dotenv()


async def run_exclusion_screening(
    input_csv_path: str = "",
    output_path: str = "screening_results.csv",
    literature_items=None,
    batch_id: str = "",
    stage: int = 3,
    audit_dir: str = ".",
    model_name: str = "claude-sonnet-4-6",
    temperature: float = 0.2,
    max_output_tokens: int = 16000,
    pilot_mode: bool = False,
    pilot_ground_truth_col: str = "r1_decision",
):
    """Run the exclusion screening pipeline."""
    initial_state = ScreeningState(
        exclusion_criteria=EXCLUSION_CODES,
        topic=PICOC_DEFINITION,
        input_csv_path=input_csv_path,
        output_path=output_path,
        literature_items=literature_items or [],
        batch_id=batch_id,
        stage=stage,
        audit_dir=audit_dir,
        pilot_mode=pilot_mode,
        pilot_ground_truth_col=pilot_ground_truth_col,
    )

    config = RunnableConfig(
        configurable={
            "model_name": model_name,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
    )

    return await graph_screening.ainvoke(initial_state, config=config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run exclusion screening pipeline (screening_instructions_v3)"
    )
    parser.add_argument(
        "--input-csv",
        default="",
        help="Path to input CSV with bibliographic data. Screening columns are appended in-place.",
    )
    parser.add_argument(
        "--output-path",
        default="screening_results.csv",
        help="Fallback output path when --input-csv is not provided.",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="Model name (e.g. claude-sonnet-4-6, claude-opus-4-7, or an Ollama model).",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-output-tokens", type=int, default=16000)
    parser.add_argument(
        "--stage",
        type=int,
        default=3,
        choices=[2, 3],
        help="Screening stage: 2=pilot validation, 3=full title/abstract screening.",
    )
    parser.add_argument(
        "--batch-id",
        default="",
        help="Human-readable batch identifier appended to audit file names.",
    )
    parser.add_argument(
        "--audit-dir",
        default=".",
        help="Directory where screening_audit_<batch_id>.md is written.",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help=(
            "Enable Stage 2 pilot validation. Computes sensitivity, specificity, "
            "Cohen's κ, and Gwet's AC1 against R1 ground-truth labels."
        ),
    )
    parser.add_argument(
        "--pilot-gt-col",
        default="r1_decision",
        help="Column in the input CSV that holds R1 ground-truth labels for pilot mode.",
    )
    args = parser.parse_args()

    asyncio.run(
        run_exclusion_screening(
            input_csv_path=args.input_csv,
            output_path=args.output_path,
            batch_id=args.batch_id,
            stage=args.stage,
            audit_dir=args.audit_dir,
            model_name=args.model,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            pilot_mode=args.pilot,
            pilot_ground_truth_col=args.pilot_gt_col,
        )
    )
