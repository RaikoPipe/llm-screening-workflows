"""LangGraph literature screening agent for strategic manufacturing/logistics SLR.

Implements screening_instructions_v3: 12 exclusion codes, PICOC scope,
tri-level decision output (Include/Exclude/Maybe), and structured audit logging.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, TypedDict

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from pydantic import Field, create_model
from tqdm.asyncio import tqdm

from src.utils import get_paper_collection, remove_section


# ── Screening scope (§1) ─────────────────────────────────────────────────────

PICOC_DEFINITION = """
## PICOC — Screening Scope

**Population (P):** Manufacturing and logistics organisations conducting strategic planning.

**Intervention (I):** Use of Large Language Models (LLMs) as a core computational component in
planning or decision-making systems. The LLM must actively generate, reason over, or guide
strategic plans — not merely be referenced or used for incidental text processing.

**Comparator (C):** Conventional optimisation, simulation, or human-expert planning (implicit;
no explicit comparator required for inclusion).

**Outcome (O):** Improved or automated strategic planning or decision quality, demonstrated
through case study, experiment, or prototype evaluation.

**Context (C):** One or more of the following five in-scope subdomains:
  1. Factory planning (facility layout, capacity planning, production network design)
  2. Supply-chain (SC) network design (supplier selection, network configuration, inventory strategy)
  3. Strategic logistics / distribution-network design (warehouse location, transport-mode selection, last-mile design)
  4. Technology selection & investment appraisal (make-or-buy, technology roadmapping, capex decisions)
  5. Global production network (GPN) design (reshoring, offshoring, multi-site configuration)

**Physical-system-configuration inclusion criterion:** Papers that configure or redesign a physical
manufacturing or logistics system at a strategic horizon qualify under EC1, even if they do not use
the word "planning."

**Upper-tactical boundary rule:** Papers focused purely on production scheduling, operational
dispatch, or real-time control (without a strategic or structural decision component) are
out-of-scope under EC1.
"""

# ── Exclusion codes (§2) ─────────────────────────────────────────────────────

EXCLUSION_CODES: Dict[str, str] = {
    "EC1": (
        "OUT-OF-SCOPE DOMAIN: The paper's primary contribution does not address strategic "
        "manufacturing or logistics planning within the five in-scope subdomains (factory planning, "
        "SC network design, strategic logistics/distribution network design, technology selection & "
        "investment appraisal, GPN design). Apply the physical-system-configuration inclusion "
        "criterion: papers that configure or redesign a physical manufacturing/logistics system at a "
        "strategic horizon qualify. Apply the upper-tactical boundary rule: exclude papers that focus "
        "purely on production scheduling, operational dispatch, or real-time control without a "
        "structural or strategic component."
    ),
    "EC2": (
        "LLM NOT USED AS CORE COMPONENT: The paper mentions or references LLMs/generative AI but "
        "does not use an LLM as a core computational component for strategic planning or "
        "decision-making. Exclude if the LLM is used only for: literature search support, text "
        "summarisation unrelated to planning, data pre-processing, or is only mentioned in "
        "future-work sections."
    ),
    "EC3": (
        "NON-STRATEGIC PLANNING SCOPE: The study addresses only operational or tactical-level tasks "
        "(e.g., production scheduling, real-time routing, shop-floor dispatching) with no strategic "
        "or structural planning component. Strategic horizon means decisions with multi-year impact "
        "on system configuration, capacity, or network topology."
    ),
    "EC4": (
        "NO LLM-DRIVEN PLANNING SYSTEM: The paper proposes or evaluates a planning system that "
        "relies solely on classical optimisation, simulation, or rule-based methods without any LLM "
        "integration. Papers comparing LLM-based and non-LLM approaches are in-scope."
    ),
    "EC5": (
        "PURELY CONCEPTUAL — NO IMPLEMENTATION OR EVALUATION: The paper presents only theoretical "
        "frameworks, position statements, or conceptual architectures without any implementation, "
        "prototype, or empirical evaluation of an LLM-based planning system."
    ),
    "EC6": (
        "GREY LITERATURE BELOW QUALITY THRESHOLD: The document is a workshop abstract, position "
        "paper, extended abstract (<4 pages), technical report without peer review, or white paper "
        "that does not meet the minimum quality standard. Note: arXiv/SSRN preprints are NOT "
        "automatically excluded — assess on content quality."
    ),
    "EC7": (
        "OUT OF DATE RANGE: The paper was published before 2020. LLM-based planning systems "
        "require transformer-era models (2020+). Pre-2020 papers on earlier AI/ML methods are "
        "out of scope."
    ),
    "EC8": (
        "NON-ENGLISH LANGUAGE: The paper's full text is not written in English."
    ),
    "EC9": (
        "EXCLUDED DOCUMENT TYPE: The document is a thesis, dissertation, book, book chapter, "
        "editorial, letter to the editor, or conference keynote abstract without substantive "
        "technical content."
    ),
    "EC10": (
        "SECONDARY LITERATURE ONLY: The paper is a systematic review, scoping review, "
        "meta-analysis, or bibliometric study that synthesises existing work without introducing a "
        "novel LLM-based planning system. Reviews that also propose and evaluate a new framework "
        "or tool are in scope."
    ),
    "EC11": (
        "DUPLICATE PUBLICATION: The paper is a duplicate of another record already included in "
        "the screening set (same study as journal article and conference paper, or pre-print and "
        "published version). Retain the most complete version."
    ),
    "EC12": (
        "FULL TEXT NOT AVAILABLE: The full text cannot be retrieved for detailed screening and "
        "the title/abstract alone are insufficient to make an inclusion decision."
    ),
}

# EC7–EC9, EC11–EC12 are metadata checks handled deterministically (pre-filter).
# EC1–EC6, EC10 require LLM assessment of content.
LLM_ASSESSED_CODES = ["EC1", "EC2", "EC3", "EC4", "EC5", "EC6", "EC10"]
METADATA_CODES = ["EC7", "EC8", "EC9", "EC11", "EC12"]

# ── Boundary-case decision rules (§3) ────────────────────────────────────────

BOUNDARY_RULES = """
## Boundary-Case Decision Rules (§3)

Apply these rules when a paper falls near a boundary:

1. **LLM mentioned but not used** → Exclude EC2. If an LLM is cited or named but does not
   function as a computational component in the reported system, apply EC2.
2. **Digital twin paper** → Include ONLY IF the LLM is used for model generation, parameter
   estimation, or strategic decision support within the digital twin. Exclude EC2 if the LLM
   is peripheral or absent.
3. **Simulation/optimisation paper with LLM in future work only** → Exclude EC2. Mentioning
   LLM integration as future work does not qualify.
4. **Paper addresses both strategic and operational levels** → Include if the strategic level is
   the primary contribution. If strategic content is incidental, Exclude EC3.
5. **SC visibility/monitoring paper without planning** → Exclude EC3. Monitoring systems without
   a structural planning decision component are out of scope.
6. **LLM framework tested on toy examples only** → Maybe (ESCALATE). Flag for human review if
   evaluation is limited to synthetic or trivial benchmarks.
7. **Survey/review that also introduces a classification framework** → Maybe (ESCALATE) if the
   new framework is substantive; otherwise Exclude EC10.
8. **Preprint (arXiv, SSRN) without peer review** → Do NOT automatically apply EC6. Assess on
   content quality; apply EC6 only if quality is clearly insufficient.
9. **LLM used for data preprocessing only** → Exclude EC2. Pre-processing (OCR, entity
   extraction, translation) without LLM-driven planning does not qualify.
10. **Industrial case study without novel method** → Include if a deployed LLM-based system is
    described and evaluated for strategic planning, even without a novel algorithmic contribution.
11. **Paper covers multiple domains, one of which is in scope** → Include if the in-scope
    subdomain is substantive (≥25% of contribution). Exclude EC1 if in-scope content is incidental.
12. **AI-assisted design without explicit LLM** → Exclude EC2. Papers using earlier AI methods
    (ML, expert systems, evolutionary algorithms) without transformer-based LLMs do not qualify.
"""

PROMPT_VERSION = "v3.0"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _create_llm(model_name: str, temperature: float, max_output_tokens: int):
    """Instantiate the appropriate chat model based on the model name prefix."""
    if "claude" in model_name.lower():
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model_name, temperature=temperature, max_tokens=max_output_tokens)
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model_name, temperature=temperature)


def build_system_prompt() -> str:
    """Build the system prompt with PICOC scope and boundary rules embedded."""
    return f"""You are an expert systematic literature reviewer specialising in LLM-based strategic manufacturing and logistics planning systems.

{PICOC_DEFINITION}

{BOUNDARY_RULES}

## Your Task

Evaluate the provided paper against the LLM-assessed exclusion criteria (EC1–EC6, EC10) independently. For each criterion:
- Provide a single concise reasoning sentence explaining your assessment.
- Give a boolean decision: **True** if the criterion applies (paper should be excluded on this criterion), **False** if it does not apply.

Then provide an **overall screening decision**:
- **Include**: No exclusion criterion applies — the paper is in scope.
- **Exclude**: At least one criterion applies. Set `excl_code` to the FIRST applicable code (e.g. "EC1").
- **Maybe**: The paper is borderline, ambiguous, or a boundary rule calls for ESCALATE. In `note`, write "ESCALATE: <brief reason>".

**Conservative screening principle:** When genuinely uncertain between Include and Exclude, choose **Maybe** rather than Exclude. Do not exclude papers without clear evidence from the text.

Metadata codes (EC7: date, EC8: language, EC9: document type, EC11: duplicate, EC12: full text) are handled separately and must NOT be evaluated here."""


def build_screening_decision_model():
    """Build a ScreeningDecision Pydantic model dynamically for LLM-assessed codes.

    Uses create_model to avoid mutating a shared class across runs.
    """
    field_definitions: Dict[str, Any] = {}
    for code in LLM_ASSESSED_CODES:
        desc = EXCLUSION_CODES[code]
        field_definitions[f"{code}_reasoning"] = (
            str,
            Field(description=f"Single sentence: does {code} apply? Context: {desc[:120]}"),
        )
        field_definitions[f"{code}_applies"] = (
            bool,
            Field(description=f"True if exclusion criterion {code} applies (paper excluded on this criterion)."),
        )
    field_definitions["decision"] = (
        Literal["Include", "Exclude", "Maybe"],
        Field(description="Overall decision: Include (in scope), Exclude (clearly out of scope), Maybe (ESCALATE)."),
    )
    field_definitions["excl_code"] = (
        str,
        Field(description="First applicable exclusion code if decision=Exclude (e.g. 'EC1'). Empty for Include/Maybe."),
    )
    field_definitions["note"] = (
        str,
        Field(description="Brief reasoning summary. For Maybe: 'ESCALATE: <reason>'."),
    )
    return create_model("ScreeningDecision", **field_definitions)


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class LiteratureItem:
    """Represents a literature item with title, abstract, and optional full text."""

    title: str
    doi: str
    abstract: str
    fulltext: str = ""
    extra: str = ""
    overlapping_authorship: bool = False


@dataclass
class ScreeningResult:
    """Represents the tri-level screening outcome for one literature item."""

    title: str
    doi: str
    decision: str  # Include / Exclude / Maybe / MANUAL_R2
    excl_code: str = ""
    note: str = ""
    per_code_reasoning: Dict[str, str] = field(default_factory=dict)
    per_code_applies: Dict[str, bool] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class Configuration(TypedDict):
    """Configurable parameters for the agent."""

    model_name: str
    temperature: float
    max_output_tokens: int
    max_fulltext_words: int


@dataclass
class State:
    """State for the literature screening agent."""

    exclusion_criteria: Dict[str, str] = field(default_factory=lambda: EXCLUSION_CODES.copy())
    topic: str = field(default_factory=lambda: PICOC_DEFINITION)
    collection_key: str = field(default_factory=lambda: os.environ.get("ZOTERO_COLLECTION_KEY", ""))
    literature_items: List[LiteratureItem] = field(default_factory=list)
    results: List[ScreeningResult] = field(default_factory=list)
    input_csv_path: str = ""
    output_path: str = "screening_results.csv"
    batch_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    stage: int = 3
    audit_dir: str = "."
    pilot_mode: bool = False
    pilot_ground_truth_col: str = "r1_decision"
    # Populated by screen_literature for downstream logging
    run_model_name: str = ""
    run_temperature: float = 0.0
    run_prompt_hash: str = ""


# ── Graph nodes ───────────────────────────────────────────────────────────────

async def load_literature(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Load literature items from a pre-populated list, an input CSV, or Zotero."""
    if state.literature_items:
        return {"literature_items": state.literature_items}

    literature_items: List[LiteratureItem] = []

    if state.input_csv_path and os.path.exists(state.input_csv_path):
        df = pd.read_csv(state.input_csv_path)
        for _, row in df.iterrows():
            fulltext_val = row.get("fulltext", "")
            extra_val = row.get("extra", "")
            literature_items.append(LiteratureItem(
                title=str(row.get("title", "")),
                doi=str(row.get("DOI", row.get("doi", ""))),
                abstract=str(row.get("abstractNote", row.get("abstract", ""))),
                fulltext=str(fulltext_val) if pd.notna(fulltext_val) else "",
                extra=str(extra_val) if pd.notna(extra_val) else "",
                overlapping_authorship=bool(row.get("overlapping_authorship", False)),
            ))
    else:
        paper_collection = get_paper_collection(collection_key=state.collection_key, get_fulltext=True)
        for _, paper in paper_collection.iterrows():
            literature_items.append(LiteratureItem(
                title=paper.title,
                abstract=paper.abstractNote,
                doi=paper.DOI,
                fulltext=paper.fulltext,
                extra=paper.extra,
            ))

    print(f"Loaded {len(literature_items)} literature items")
    return {"literature_items": literature_items}


async def screen_literature(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Screen each literature item with LLM-assessed exclusion criteria."""
    configuration = config.get("configurable", {})
    model_name = configuration.get("model_name", "claude-sonnet-4-6")
    temperature = configuration.get("temperature", 0.2)
    max_output_tokens = configuration.get("max_output_tokens", 16000)
    max_fulltext_words = configuration.get("max_fulltext_words", 12000)

    ScreeningDecision = build_screening_decision_model()
    system_prompt = build_system_prompt()
    llm_agent = _create_llm(model_name, temperature, max_output_tokens)
    llm_agent = llm_agent.with_structured_output(ScreeningDecision)

    criteria_text = "\n".join(
        f"- **{code}**: {EXCLUSION_CODES[code]}"
        for code in LLM_ASSESSED_CODES
    )

    results: List[ScreeningResult] = []

    for item in tqdm(state.literature_items, desc="Screening literature", unit="item"):
        # Papers with overlapping authorship are routed to R2 without LLM screening (§6)
        if item.overlapping_authorship:
            results.append(ScreeningResult(
                title=item.title,
                doi=item.doi,
                decision="MANUAL_R2",
                note="Overlapping authorship — routed to R2 for manual review per §6.",
            ))
            continue

        # Papers manually flagged via the 'extra' field
        if item.extra == "skip":
            results.append(ScreeningResult(
                title=item.title,
                doi=item.doi,
                decision="Maybe",
                note="ESCALATE: Paper manually flagged as 'skip' in extra field.",
            ))
            continue

        # Determine text to screen (full text preferred, abstract as fallback)
        if isinstance(item.fulltext, str) and item.fulltext.strip():
            raw_text = remove_section(item.fulltext)
            text_to_screen = " ".join(raw_text.split()[:max_fulltext_words])
            text_label = "Fulltext"
        else:
            text_to_screen = item.abstract or ""
            text_label = "Abstract"

        human_prompt = f"""# Title: {item.title}

# {text_label}: {text_to_screen}

---
## Exclusion Criteria to Evaluate

{criteria_text}

Evaluate this paper against all exclusion criteria above. For each criterion, provide reasoning and a True/False decision. Then give an overall Include/Exclude/Maybe decision.
"""

        response = None
        last_error: Optional[Exception] = None
        for attempt in range(3):
            try:
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=human_prompt),
                ]
                response = await llm_agent.ainvoke(messages)
                break
            except Exception as e:
                last_error = e

        if response is None:
            # Conservative failure default: Maybe/ESCALATE, not Exclude
            print(f"Failed to screen '{item.title}' after 3 attempts: {last_error}")
            results.append(ScreeningResult(
                title=item.title,
                doi=item.doi,
                decision="Maybe",
                note=f"ESCALATE: LLM screening failed after 3 attempts — {last_error}",
            ))
            continue

        data = response.model_dump()
        per_code_reasoning = {code: data.get(f"{code}_reasoning", "") for code in LLM_ASSESSED_CODES}
        per_code_applies = {code: data.get(f"{code}_applies", False) for code in LLM_ASSESSED_CODES}

        results.append(ScreeningResult(
            title=item.title,
            doi=item.doi,
            decision=data["decision"],
            excl_code=data.get("excl_code", ""),
            note=data.get("note", ""),
            per_code_reasoning=per_code_reasoning,
            per_code_applies=per_code_applies,
        ))

    return {
        "results": results,
        "run_model_name": model_name,
        "run_temperature": temperature,
        "run_prompt_hash": _prompt_hash(system_prompt),
    }


async def generate_output(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Append screening columns to the original input CSV and write a markdown audit file."""
    timestamp = datetime.now(timezone.utc).isoformat()

    # Build a lookup keyed by DOI (primary) and title (fallback)
    result_by_doi: Dict[str, ScreeningResult] = {}
    result_by_title: Dict[str, ScreeningResult] = {}
    for r in state.results:
        if r.doi:
            result_by_doi[r.doi] = r
        result_by_title[r.title] = r

    def _lookup(row: pd.Series) -> Optional[ScreeningResult]:
        doi = str(row.get("DOI", row.get("doi", ""))).strip()
        title = str(row.get("title", "")).strip()
        return result_by_doi.get(doi) or result_by_title.get(title)

    # ── 1. Append decision columns to original input CSV ─────────────────────
    try:
        if state.input_csv_path and os.path.exists(state.input_csv_path):
            df_out = pd.read_csv(state.input_csv_path)
        else:
            df_out = pd.DataFrame([{"title": r.title, "doi": r.doi} for r in state.results])

        stages, reviewers, decisions, excl_codes, notes, timestamps = [], [], [], [], [], []
        for _, row in df_out.iterrows():
            res = _lookup(row)
            stages.append(state.stage)
            reviewers.append("LLM")
            decisions.append(res.decision if res else "")
            excl_codes.append(res.excl_code if res else "")
            notes.append(res.note if res else "")
            timestamps.append(timestamp)

        df_out["stage"] = stages
        df_out["reviewer"] = reviewers
        df_out["decision"] = decisions
        df_out["excl_code"] = excl_codes
        df_out["note"] = notes
        df_out["timestamp"] = timestamps

        out_path = state.input_csv_path if state.input_csv_path else state.output_path
        df_out.to_csv(out_path, index=False)
        print(f"Screening results appended to {out_path}")

    except Exception as e:
        print(f"Error writing output CSV: {e}")

    # ── 2. Generate markdown audit file ──────────────────────────────────────
    decision_counts: Dict[str, int] = {"Include": 0, "Exclude": 0, "Maybe": 0, "MANUAL_R2": 0}
    code_counts: Dict[str, int] = {code: 0 for code in EXCLUSION_CODES}
    for r in state.results:
        decision_counts[r.decision] = decision_counts.get(r.decision, 0) + 1
        if r.excl_code and r.excl_code in code_counts:
            code_counts[r.excl_code] += 1

    try:
        os.makedirs(state.audit_dir, exist_ok=True)
        audit_path = os.path.join(state.audit_dir, f"screening_audit_{state.batch_id}.md")

        lines = [
            f"# Screening Audit — Batch `{state.batch_id}`",
            "",
            "## Run Metadata",
            "",
            "| Key | Value |",
            "|-----|-------|",
            f"| Model | `{state.run_model_name}` |",
            f"| Temperature | {state.run_temperature} |",
            f"| Prompt version | {PROMPT_VERSION} |",
            f"| Prompt hash | `{state.run_prompt_hash}` |",
            f"| Stage | {state.stage} |",
            f"| Timestamp | {timestamp} |",
            f"| Records screened | {len(state.results)} |",
            f"| Batch ID | `{state.batch_id}` |",
            "",
            "## Decision Summary",
            "",
            "| Decision | Count |",
            "|----------|-------|",
        ]
        for dec, cnt in decision_counts.items():
            lines.append(f"| {dec} | {cnt} |")

        lines += [
            "",
            "## Per-Code Exclusion Counts",
            "",
            "| Code | Short Description | Count |",
            "|------|-------------------|-------|",
        ]
        for code, cnt in code_counts.items():
            short_desc = EXCLUSION_CODES[code].split(":")[0]
            lines.append(f"| {code} | {short_desc} | {cnt} |")

        lines += [
            "",
            "## Per-Record Detailed Reasoning",
            "",
            "| # | DOI | Title | Decision | Excl Code | Note |",
            "|---|-----|-------|----------|-----------|------|",
        ]
        for i, r in enumerate(state.results, 1):
            title_cell = (r.title[:60] + "…").replace("|", "\\|") if len(r.title) > 60 else r.title.replace("|", "\\|")
            note_cell = (r.note[:80] + "…").replace("|", "\\|") if len(r.note) > 80 else r.note.replace("|", "\\|")
            lines.append(f"| {i} | `{r.doi}` | {title_cell} | **{r.decision}** | {r.excl_code} | {note_cell} |")

        flagged = [r for r in state.results if r.decision in ("Maybe", "MANUAL_R2")]
        if flagged:
            lines += ["", "## Flagged Records (Maybe / ESCALATE / MANUAL_R2)", ""]
            for r in flagged:
                lines += [
                    f"### {r.title}",
                    f"- **DOI:** {r.doi}",
                    f"- **Decision:** {r.decision}",
                    f"- **Note:** {r.note}",
                    "",
                ]
                if r.per_code_reasoning:
                    lines.append("**Per-code reasoning:**")
                    lines.append("")
                    for code, reasoning in r.per_code_reasoning.items():
                        applies = r.per_code_applies.get(code, False)
                        marker = "✗ applies" if applies else "✓ clear"
                        lines.append(f"- {code} ({marker}): {reasoning}")
                    lines.append("")

        with open(audit_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Audit report written to {audit_path}")

    except Exception as e:
        print(f"Error writing audit file: {e}")

    print(f"\nScreening summary (batch {state.batch_id}):")
    for dec, cnt in decision_counts.items():
        print(f"  {dec}: {cnt}")

    return {}


async def run_pilot_validation(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Compare LLM decisions with R1 ground-truth labels and compute agreement metrics (§6).

    Thresholds: sensitivity ≥ 0.95, specificity ≥ 0.70, Cohen's κ ≥ 0.70.
    Maybe decisions are treated as Include (conservative) for binary metrics.
    """
    if not state.pilot_mode:
        return {}

    gt_col = state.pilot_ground_truth_col
    csv_path = state.input_csv_path if state.input_csv_path else state.output_path

    if not csv_path or not os.path.exists(csv_path):
        print("Pilot validation: no CSV found, skipping.")
        return {}

    df = pd.read_csv(csv_path)
    if gt_col not in df.columns:
        print(f"Pilot validation: column '{gt_col}' not found in CSV, skipping.")
        return {}

    # Conservative binary mapping: Include=1 (keep), Exclude=0 (reject); Maybe → Include
    def _to_binary(decision: str) -> int:
        return 0 if str(decision).strip().capitalize() == "Exclude" else 1

    result_by_doi = {r.doi: r for r in state.results if r.doi}
    result_by_title = {r.title: r for r in state.results}

    r1_labels: List[int] = []
    llm_labels: List[int] = []

    for _, row in df.iterrows():
        r1_raw = str(row.get(gt_col, "")).strip()
        if not r1_raw or r1_raw.lower() == "nan":
            continue
        doi = str(row.get("DOI", row.get("doi", ""))).strip()
        title = str(row.get("title", "")).strip()
        res = result_by_doi.get(doi) or result_by_title.get(title)
        if res is None:
            continue
        r1_labels.append(_to_binary(r1_raw))
        llm_labels.append(_to_binary(res.decision))

    n = len(r1_labels)
    if n == 0:
        print("Pilot validation: no matched records with ground-truth labels found.")
        return {}

    tp = sum(1 for r, l in zip(r1_labels, llm_labels) if r == 1 and l == 1)
    tn = sum(1 for r, l in zip(r1_labels, llm_labels) if r == 0 and l == 0)
    fp = sum(1 for r, l in zip(r1_labels, llm_labels) if r == 0 and l == 1)
    fn = sum(1 for r, l in zip(r1_labels, llm_labels) if r == 1 and l == 0)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    po = (tp + tn) / n

    r1_pos = (tp + fn) / n
    llm_pos = (tp + fp) / n
    pe_cohen = r1_pos * llm_pos + (1 - r1_pos) * (1 - llm_pos)
    kappa = (po - pe_cohen) / (1 - pe_cohen) if (1 - pe_cohen) > 0 else 0.0

    pw = (r1_pos + llm_pos) / 2
    pe_gwet = 2 * pw * (1 - pw)
    ac1 = (po - pe_gwet) / (1 - pe_gwet) if (1 - pe_gwet) > 0 else 0.0

    THRESH_SENS = 0.95
    THRESH_SPEC = 0.70
    THRESH_KAPPA = 0.70

    passed = sensitivity >= THRESH_SENS and specificity >= THRESH_SPEC and kappa >= THRESH_KAPPA

    print("\n" + "=" * 60)
    print(f"PILOT VALIDATION RESULTS (n={n})")
    print("=" * 60)
    print(f"  Sensitivity:  {sensitivity:.3f}  (≥{THRESH_SENS}) {'PASS' if sensitivity >= THRESH_SENS else 'FAIL'}")
    print(f"  Specificity:  {specificity:.3f}  (≥{THRESH_SPEC}) {'PASS' if specificity >= THRESH_SPEC else 'FAIL'}")
    print(f"  Cohen's κ:    {kappa:.3f}  (≥{THRESH_KAPPA}) {'PASS' if kappa >= THRESH_KAPPA else 'FAIL'}")
    print(f"  Gwet's AC1:   {ac1:.3f}")
    print(f"  TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    verdict = "PASS — Stage 3 can proceed." if passed else "FAIL — Human review required before Stage 3."
    print(f"  Overall: {verdict}")
    print("=" * 60)

    audit_path = os.path.join(state.audit_dir, f"screening_audit_{state.batch_id}.md")
    try:
        with open(audit_path, "a", encoding="utf-8") as f:
            f.write("\n## Pilot Validation Metrics\n\n")
            f.write("| Metric | Value | Threshold | Status |\n")
            f.write("|--------|-------|-----------|--------|\n")
            f.write(f"| Sensitivity | {sensitivity:.3f} | ≥{THRESH_SENS} | {'PASS' if sensitivity >= THRESH_SENS else 'FAIL'} |\n")
            f.write(f"| Specificity | {specificity:.3f} | ≥{THRESH_SPEC} | {'PASS' if specificity >= THRESH_SPEC else 'FAIL'} |\n")
            f.write(f"| Cohen's κ | {kappa:.3f} | ≥{THRESH_KAPPA} | {'PASS' if kappa >= THRESH_KAPPA else 'FAIL'} |\n")
            f.write(f"| Gwet's AC1 | {ac1:.3f} | — | — |\n")
            f.write(f"\n**Confusion matrix:** TP={tp}, TN={tn}, FP={fp}, FN={fn} (n={n})\n")
            f.write(f"\n**Overall verdict:** {verdict}\n")
    except Exception as e:
        print(f"Error appending pilot metrics to audit: {e}")

    return {}


# ── Graph definition ──────────────────────────────────────────────────────────

graph = (
    StateGraph(State, context_schema=Configuration)
    .add_node("load_literature", load_literature)
    .add_node("screen_literature", screen_literature)
    .add_node("generate_output", generate_output)
    .add_node("run_pilot_validation", run_pilot_validation)
    .add_edge("__start__", "load_literature")
    .add_edge("load_literature", "screen_literature")
    .add_edge("screen_literature", "generate_output")
    .add_edge("generate_output", "run_pilot_validation")
    .compile(name="Literature Screening Agent")
)
