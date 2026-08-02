"""LangGraph literature screening agent — screening_instructions_v6 compliant.

Tri-level decision output: Include / Exclude / Maybe (§4).

Exclusion-code split:
  LLM-evaluated       : 1–12 (all codes; content analysis + text-derived metadata)
  Metadata pre-filter : 7 (year < 2022), 8 (language), 11 (duplicate),
                        6 (grey-lit tier via item.is_grey_literature),
                        12 (full-text unavailable via item.fulltext_unavailable)

The metadata pre-filter is a fast-path that short-circuits before the LLM when
the relevant flags/fields are present. The LLM additionally evaluates all 12
codes so that records with missing metadata can still be excluded on codes
6–8, 11, 12 when the text itself reveals the condition.
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
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm

from src.utils import get_paper_collection, get_llm, remove_section


# ---------------------------------------------------------------------------
# Configuration schema
# ---------------------------------------------------------------------------

class Configuration(TypedDict):
    """Configurable parameters passed via RunnableConfig."""

    model_name: str
    temperature: float
    max_fulltext_words: int
    max_output_tokens: int


# ---------------------------------------------------------------------------
# Structured LLM output — static model, no global mutation
# ---------------------------------------------------------------------------

class CodeEvaluation(BaseModel):
    """Per-exclusion-code assessment."""

    reasoning: str = Field(description="Single-sentence reasoning for this code.")
    applies: bool = Field(description="True if this exclusion code applies to this paper.")


class ScreeningDecision(BaseModel):
    """Structured LLM output per screening_instructions_v6 §2/§3.

    All 12 exclusion codes are evaluated by the LLM. Codes 6–8, 11, 12 are
    additionally handled as metadata pre-filters before the LLM is invoked
    when the relevant flags/fields are present on the LiteratureItem.
    """

    code_1: CodeEvaluation = Field(description=(
        "Code 1 — Operational/tactical scope only: applies if the paper addresses ONLY operational "
        "or tactical decisions (scheduling, MES/ERP, shopfloor control, predictive maintenance, "
        "quality control, S&OP) WITHOUT altering physical system configuration. Upper-tactical "
        "decisions that expand capacity or reconfigure lines are NOT excluded here."
    ))
    code_2: CodeEvaluation = Field(description=(
        "Code 2 — No LLM component: applies if no LLM, foundation model, generative AI, agentic "
        "AI, RAG, or LLM-hybrid (solver, knowledge graph, digital twin) appears in the contribution. "
        "If LLM is merely mentioned but not actually used, this code applies."
    ))
    code_3: CodeEvaluation = Field(description=(
        "Code 3 — No strategic manufacturing/logistics task: applies if the paper does not address "
        "at least one of the five strategic subdomains (factory planning per VDI 5200, SC network "
        "design, strategic logistics/distribution network design, technology selection & investment "
        "appraisal, GPN design)."
    ))
    code_4: CodeEvaluation = Field(description=(
        "Code 4 — AI/ML without LLM: applies if the AI/ML contribution uses only pure RL, classical "
        "ML, or metaheuristics with no LLM component. A pure MILP/solver without LLM front-end "
        "also triggers this code."
    ))
    code_5: CodeEvaluation = Field(description=(
        "Code 5 — Non-manufacturing/logistics domain: applies if the primary domain is NOT "
        "manufacturing or logistics (e.g., healthcare, finance, or logistics outside a production "
        "network context)."
    ))
    code_6: CodeEvaluation = Field(description=(
        "Code 6 — Grey literature below Tier 1: applies if the paper is a vendor white paper, blog "
        "post, or unaffiliated preprint. Institutionally affiliated arXiv preprints are Tier 1 and "
        "are NOT excluded here. Prefer the metadata flag when present; use this only when the text "
        "itself reveals the venue type and the metadata flag is absent."
    ))
    code_7: CodeEvaluation = Field(description=(
        "Code 7 — Published before January 2022: applies if the publication date is before 2022-01-01 "
        "(online-first counts as the publication date). Prefer the metadata year field when present; "
        "use this only when the year must be inferred from the text and the metadata field is absent."
    ))
    code_8: CodeEvaluation = Field(description=(
        "Code 8 — Language not English or German: applies if the paper is written in a language other "
        "than English or German. Prefer the metadata language field when present; use this only when "
        "the language must be inferred from the text and the metadata field is absent."
    ))
    code_9: CodeEvaluation = Field(description=(
        "Code 9 — Editorial/opinion: applies if the paper is an editorial, opinion piece, or "
        "non-methodological commentary without a methodological or empirical contribution."
    ))
    code_10: CodeEvaluation = Field(description=(
        "Code 10 — Secondary literature: applies if the paper is a systematic literature review, "
        "narrative review, scoping review, or meta-analysis."
    ))
    code_11: CodeEvaluation = Field(description=(
        "Code 11 — Duplicate record: applies if the paper appears to be a duplicate of another "
        "record in the corpus. Prefer the metadata is_duplicate flag when present; use this only "
        "when the text itself reveals duplication (e.g., identical title to a peer-reviewed version) "
        "and the metadata flag is absent."
    ))
    code_12: CodeEvaluation = Field(description=(
        "Code 12 — Full text and/or abstract unavailable: applies if neither the full text nor a "
        "substantive abstract is available for screening. Where the title alone is unambiguously "
        "in-scope and full text is obtainable, do NOT apply this code (screen in on title and "
        "resolve at Stage 4). Prefer the metadata fulltext_unavailable flag when present; use this "
        "only when the absence is evident from the text and the metadata flag is absent."
    ))
    decision: Literal["Include", "Exclude", "Maybe"] = Field(description=(
        "Overall screening decision. "
        "Include = no exclusion codes apply. "
        "Exclude = at least one code applies (set excl_code to the first applicable code). "
        "Maybe = genuinely ambiguous; set note to 'ESCALATE: <one-line reason>'."
    ))
    excl_code: Optional[int] = Field(default=None, description=(
        "First applicable exclusion code number (1–12), or null for Include/Maybe decisions."
    ))
    note: str = Field(description=(
        "Brief reasoning summary for the overall decision, "
        "or 'ESCALATE: <reason>' for Maybe decisions."
    ))


# ---------------------------------------------------------------------------
# System prompt (PICOC + codes + boundary rules — §1/§2/§3/§4)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a systematic literature review (SLR) screening expert. \
The study concerns **LLM-based decision support for strategic manufacturing and logistics planning**.

## 1. PICOC

| Element | Definition | Include example | Exclude example |
|---|---|---|---|
| **P**opulation | LLM-based decision-support systems applied to manufacturing and logistics planning at the strategic or upper-tactical horizon (decisions that configure or reconfigure the physical production system) | LLM-assisted greenfield factory concept design | Shopfloor scheduling system |
| **I**ntervention | LLM, foundation model, generative AI, agentic AI, RAG, or LLM-hybrid (solver, KG, digital twin) **where the LLM generates, structures, mediates, or substantively supports the planning decision itself** | Multi-agent LLM for SC network configuration | Classical MILP solver without LLM component |
| **C**omparator | Any or none (baselines optional) | Human planner vs. LLM agent | — |
| **O**utcome | Task coverage, architectural patterns, empirical evidence, adoption barriers | Auto-generated factory layout concept; what-if network analysis | Chatbot UX satisfaction only |
| **C**ontext | Strategic manufacturing and logistics planning across five subdomains: factory planning (VDI 5200), SC network design, strategic logistics/distribution network design, technology selection & investment appraisal, GPN design | Global production network footprint reconfiguration | MES/ERP runtime operations |

---

## 2. Exclusion codes (log one per excluded record)

| Code | Reason |
|------|--------|
| 1 | Scope exclusively operational or tactical (scheduling, MES/ERP, shopfloor control, predictive maintenance, quality control, S&OP). Upper-tactical decisions are included **only** when they alter the physical system configuration (e.g., capacity expansion, line reconfiguration); decisions that schedule or sequence within a fixed configuration are excluded. |
| 2 | No qualifying LLM / foundation model / generative AI / agentic AI / RAG / LLM-hybrid component. The LLM must generate, structure, mediate, or substantively support the planning decision itself. Qualifying functions include: proposing or parametrising plans, RAG over planning knowledge that feeds the decision, reasoning over the decision, LLM↔solver query mediation, synthesising heterogenous data to derive planning-relevant conclusions, running or interpreting what-if/scenario analyses, data-driven forecasting that directly informs a planning judgment, or recommendation/advisory output that combines information to support a planning decision.  |
| 3 | Does not address at least one strategic manufacturing and logistics planning task (see §1 subdomains). **Supplier, partner, sourcing, and bid-evaluation decisions qualify only when they determine production-network topology or site/echelon structure**; procurement within a fixed network is excluded under this code. |
| 4 | AI/ML contribution without LLM component (pure RL, classical ML, metaheuristics). **VAE/GAN "generative design" is non-LLM generative ML and is excluded here.** |
| 5 | Non-manufacturing/logistics domain (healthcare, finance, logistics outside production network context). **Facility/site-selection methods qualify only in a manufacturing/logistics production context**; generic urban, retail, energy-grid, and abstract mechanism-design siting are excluded under this code. |
| 6 | Grey literature below Tier 1 (vendor white papers, blog posts, unaffiliated preprints) |
| 7 | Published before January 2022 |
| 8 | Language ≠ English or German |
| 9 | Editorial, opinion, or position piece **only**. Conceptual frameworks, taxonomies, and design-science architectures are **not** excluded here — they are valid Category-C study types and are screened **in** for full-text resolution (see §3). |
| 10 | Secondary literature (SLR, narrative review) — retain for reference snowballing only |
| 11 | Duplicate record |
| 12 | Full text and/or abstract unavailable. Where the title alone is unambiguously in-scope and full text is obtainable, screen in on title and resolve at Stage 4.
---

## 3. Boundary-case decision rules

| Case | Decision |
|------|----------|
| LLM mentioned but not used in the contribution | Exclude (code 2) |
| LLM role is peripheral: basic document QA, standalone information retrieval (web search or database lookup without combinatory insight derivation), text mining, NLP preprocessing, or literature/research assistance. | Exclude (code 2) |
| LLM+solver hybrid where the LLM mediates user intent ↔ solver for a strategic task | Include |
| Supplier / partner / sourcing / bid-evaluation decision that determines production-network topology or site/echelon structure | Include; document the configuring effect |
| Supplier / sourcing / procurement decision within a fixed network (no topology change) | Exclude (code 3) |
| Site/facility-selection method in a manufacturing/logistics production context | Include |
| Site/facility-selection in a generic urban, retail, energy-grid, or abstract mechanism-design setting | Exclude (code 5) |
| VAE/GAN "generative design" with no LLM component | Exclude (code 4) |
| Conceptual framework / taxonomy / design-science architecture (no implementation) | Include (screen-in) as Category-C; resolve at full-text. Do **not** exclude as commentary |
| Editorial / opinion / position piece | Exclude (code 9) |
| Tactical planning with strategic framing (e.g., capacity allocation that configures new facilities such as new production lines) | Include; document framing explicitly |
| Upper-tactical decision that schedules/sequences within a fixed configuration (shift-pattern adjustment, lot-sizing, production sequencing on existing lines) | Exclude (code 1) |
| Digital twin / simulation paper | Include only if LLM/GenAI is used for model generation, parametrization, or a strategic planning decision; else exclude (code 2) |
| SC network design with no LLM front end (pure MILP/solver) | Exclude (code 4) |
| Preprint + journal version both found | Keep peer-reviewed version; log preprint DOI with tag `PREPRINT-OF:{DOI}` |
| Institutionally affiliated arXiv preprint, no journal version | Include (Tier 1) |
| Unaffiliated arXiv preprint | Exclude (code 6) |
| Workshop / short paper (< 4 pages) | Include if methodological; exclude if abstract-only or poster |
| Ambiguous strategic scope | Tag `ESCALATE` with one-line note; escalate per §5 |
| Year boundary | Publication date ≥ 2022-01-01 (online-first counts) |
## Instructions

1. Evaluate each of the 12 exclusion codes (1–12) independently from the title, text, and any metadata provided.
2. Apply all boundary-case rules above before reaching a decision.
3. For codes 6 (grey-lit tier), 7 (year), 8 (language), 11 (duplicate), and 12 (fulltext/abstract unavailable), prefer an explicit metadata flag or field when present. Apply the code from the text only when the metadata is absent and the condition is clearly evident.
4. Set `decision`:
   - **Include** — no exclusion codes apply.
   - **Exclude** — at least one code applies; set `excl_code` to the first applicable code number.
   - **Maybe** — genuinely ambiguous; set `note` to "ESCALATE: <one-line reason>".
5. Provide concise single-sentence `reasoning` per code.
6. Be conservative: when in doubt choose **Maybe** over **Exclude**. In particular, escalate \
   rather than exclude when (a) the LLM's role is described but its centrality to the decision is \
   unclear, or (b) the paper is a conceptual/framework contribution.
"""

_PROMPT_VERSION = hashlib.md5(_SYSTEM_PROMPT.encode()).hexdigest()[:8]

# Pre-built cached SystemMessage — constructed once, reused for every record.
# ChatAnthropic passes cache_control blocks through to the Anthropic API as-is.
# Requires langchain-anthropic >= 0.1.15 and a Claude 3+ model.
_SYSTEM_PROMPT_CACHED = SystemMessage(content=[
    {
        "type": "text",
        "text": _SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }
])


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LiteratureItem:
    """Literature item with bibliographic and screening metadata."""

    title: str
    doi: str
    abstract: str
    fulltext: str = ""
    extra: str = ""
    record_id: str = ""
    year: Optional[int] = None
    language: Optional[str] = None
    is_duplicate: bool = False
    overlapping_authorship: bool = False
    is_grey_literature: bool = False
    fulltext_unavailable: bool = False


@dataclass
class ScreeningResult:
    """Screening decision for a single literature item."""

    record_id: str
    title: str
    doi: str
    stage: int
    reviewer: str
    decision: str
    excl_code: Optional[int]
    note: str
    timestamp: str
    screening_decision: Optional[ScreeningDecision] = None


@dataclass
class State:
    """State for the literature screening agent."""

    collection_key: str = field(default_factory=lambda: os.environ.get("ZOTERO_COLLECTION_KEY", ""))
    literature_items: Any = field(default_factory=list)
    input_path: Optional[str] = None
    output_path: str = "screening_results.csv"
    audit_dir: str = "."
    stage: int = 3
    screening_type: str = "abstract"  # "abstract" or "fulltext"
    batch_id: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%d") + "_" + uuid.uuid4().hex[:6]
    )
    results: List[ScreeningResult] = field(default_factory=list)
    run_metadata: Dict[str, Any] = field(default_factory=dict)
    original_df: Optional[pd.DataFrame] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Metadata pre-filter (codes 6–8, 11, 12)
# ---------------------------------------------------------------------------

_ALLOWED_LANGUAGES = {"en", "de", "english", "german", "eng", "deu"}


def _metadata_prefilter(item: LiteratureItem) -> Optional[int]:
    """Return first triggered metadata exclusion code, or None.

    Code split (documented in audit):
      6  — grey-lit tier          (flag: item.is_grey_literature)
      7  — year < 2022            (field: item.year)
      8  — language not EN/DE     (field: item.language)
     11  — duplicate              (flag: item.is_duplicate)
     12  — full text unavailable  (flag: item.fulltext_unavailable)
    """
    if item.is_duplicate:
        return 11
    if item.year is not None and item.year < 2022:
        return 7
    if item.language and item.language.lower() not in _ALLOWED_LANGUAGES:
        return 8
    if item.is_grey_literature:
        return 6
    if item.fulltext_unavailable:
        return 12
    return None


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

async def load_literature(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Load literature items from CSV path, pre-loaded items, or Zotero."""
    literature_items: List[LiteratureItem] = []
    original_df: Optional[pd.DataFrame] = None

    if state.input_path:
        original_df = pd.read_csv(state.input_path)
        paper_collection = original_df
    elif isinstance(state.literature_items, pd.DataFrame) and not state.literature_items.empty:
        original_df = state.literature_items
        paper_collection = original_df
    elif isinstance(state.literature_items, list) and state.literature_items:
        literature_items = state.literature_items
        paper_collection = None
    else:
        paper_collection = get_paper_collection(collection_key=state.collection_key, get_fulltext=True)

    if paper_collection is not None:
        for idx, paper in paper_collection.iterrows():
            record_id = str(paper.get("record_id", idx))
            year_raw = paper.get("year", None)
            try:
                year = int(year_raw) if year_raw is not None and pd.notna(year_raw) else None
            except (ValueError, TypeError):
                year = None

            literature_items.append(LiteratureItem(
                record_id=record_id,
                title=str(paper.get("title", "")),
                abstract=str(paper.get("abstract", paper.get("abstract", ""))),
                doi=str(paper.get("DOI", paper.get("doi", ""))),
                fulltext=str(paper.get("fulltext", "")) if pd.notna(paper.get("fulltext", "")) else "",
                extra=str(paper.get("extra", "")),
                year=year,
                language=paper.get("language", None),
                is_duplicate=bool(paper.get("is_duplicate", False)),
                overlapping_authorship=bool(paper.get("overlapping_authorship", False)),
                is_grey_literature=bool(paper.get("is_grey_literature", False)),
                fulltext_unavailable=bool(paper.get("fulltext_unavailable", False)),
            ))

    print(f"Loaded {len(literature_items)} literature items")
    return {"literature_items": literature_items, "original_df": original_df}


async def screen_literature(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Screen each literature item with tri-level output and per-code reasoning."""
    configuration = config.get("configurable", {})
    model_name = configuration.get("model_name", "gpt-oss:120b")
    temperature = configuration.get("temperature", 0.0)
    max_fulltext_words = configuration.get("max_fulltext_words", 12000)
    max_output_tokens = configuration.get("max_output_tokens", 16000)
    seed = configuration.get("seed", 0)

    llm_agent = get_llm(
        config=config,
        json_mode=True,
        temperature=temperature,
    ).with_structured_output(ScreeningDecision, method="json_mode")

    format_instructions = PydanticOutputParser(pydantic_object=ScreeningDecision).get_format_instructions()

    run_metadata: Dict[str, Any] = {
        "batch_id": state.batch_id,
        "model_name": model_name,
        "seed": seed,
        "temperature": temperature,
        "max_fulltext_words": max_fulltext_words,
        "prompt_version": _PROMPT_VERSION,
        "prompt_caching": "ephemeral",
        "stage": state.stage,
        "screening_type": state.screening_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "record_count": len(state.literature_items),
        "code_split": (
            "LLM: 1-12 (all codes) | pre-filter fast-path: 6 (grey-lit flag), 7 (year), "
            "8 (language), 11 (duplicate), 12 (fulltext unavailable)"
        ),
        "system_prompt": _SYSTEM_PROMPT
    }

    results: List[ScreeningResult] = []

    for item in tqdm(state.literature_items, desc="Screening literature", unit="item"):
        timestamp = datetime.now(timezone.utc).isoformat()
        record_id = item.record_id or item.doi or item.title[:40]

        # -- Overlapping-authorship routing (§6) ----------------------------
        if item.overlapping_authorship:
            results.append(ScreeningResult(
                record_id=record_id,
                title=item.title,
                doi=item.doi,
                stage=state.stage,
                reviewer="R1",
                decision="R2_REVIEW",
                excl_code=None,
                note="Overlapping authorship: routed to R2 per §6.",
                timestamp=timestamp,
            ))
            continue

        # -- Metadata pre-filters (codes 6–8, 11, 12) ----------------------
        meta_code = _metadata_prefilter(item)
        if meta_code is not None:
            results.append(ScreeningResult(
                record_id=record_id,
                title=item.title,
                doi=item.doi,
                stage=state.stage,
                reviewer="LLM",
                decision="Exclude",
                excl_code=meta_code,
                note=f"Metadata pre-filter triggered Code {meta_code}.",
                timestamp=timestamp,
            ))
            continue

        # -- Skip flag ------------------------------------------------------
        if item.extra == "skip":
            results.append(ScreeningResult(
                record_id=record_id,
                title=item.title,
                doi=item.doi,
                stage=state.stage,
                reviewer="LLM",
                decision="Maybe",
                excl_code=None,
                note="ESCALATE: 'skip' flag set in extra field.",
                timestamp=timestamp,
            ))
            continue

        # -- Build text to screen ------------------------------------------
        use_fulltext = (
            state.screening_type == "fulltext"
            and isinstance(item.fulltext, str)
            and item.fulltext.strip()
        )
        if use_fulltext:
            text_to_screen = " ".join(remove_section(item.fulltext).split()[:max_fulltext_words])
            text_label = "Fulltext"
        else:
            text_to_screen = item.abstract
            text_label = "Abstract"

        human_prompt = (
            f"# Title\n{item.title}\n\n"
            f"# {text_label}\n{text_to_screen}\n\n"
            "Evaluate this paper against all 12 exclusion codes (1–12). "
            "Provide per-code reasoning and decide: Include, Exclude, or Maybe.\n\n"
            f"{format_instructions}"
        )

        messages = [
            _SYSTEM_PROMPT_CACHED,
            HumanMessage(content=human_prompt),
        ]

        # -- LLM screening with retries ------------------------------------
        response: Optional[ScreeningDecision] = None
        for attempt in range(3):
            try:
                response = await llm_agent.ainvoke(messages)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"Failed to screen '{item.title}' after 3 attempts: {e}")

        if response is None:
            # Conservative default on failure: Maybe/ESCALATE (not Exclude)
            results.append(ScreeningResult(
                record_id=record_id,
                title=item.title,
                doi=item.doi,
                stage=state.stage,
                reviewer="LLM",
                decision="Maybe",
                excl_code=None,
                note="ESCALATE: LLM failed after 3 attempts.",
                timestamp=timestamp,
            ))
            continue

        results.append(ScreeningResult(
            record_id=record_id,
            title=item.title,
            doi=item.doi,
            stage=state.stage,
            reviewer="LLM",
            decision=response.decision,
            excl_code=response.excl_code,
            note=response.note,
            timestamp=timestamp,
            screening_decision=response,
        ))

    return {"results": results, "run_metadata": run_metadata}


async def generate_output(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Append screening columns to original input CSV and write markdown audit."""

    # -- Build screening columns DataFrame ----------------------------------
    screening_rows = [
        {
            "record_id": r.record_id,
            "stage": r.stage,
            "reviewer": r.reviewer,
            "decision": r.decision,
            "excl_code": r.excl_code,
            "note": r.note,
            "timestamp": r.timestamp,
        }
        for r in state.results
    ]
    screening_df = pd.DataFrame(screening_rows)

    # -- Append to original bibliographic data ------------------------------
    if state.original_df is not None:
        out_df = state.original_df.copy()
        if "record_id" in out_df.columns:
            # Drop columns that will be overwritten to avoid duplicates on re-run
            overlap = [c for c in screening_df.columns if c in out_df.columns and c != "record_id"]
            out_df = out_df.drop(columns=overlap, errors="ignore")
            out_df = out_df.merge(screening_df, on="record_id", how="left")
        else:
            out_df = pd.concat(
                [out_df.reset_index(drop=True), screening_df.reset_index(drop=True)], axis=1
            )
    else:
        bib_rows = [
            {
                "record_id": item.record_id,
                "title": item.title,
                "doi": item.doi,
                "abstract": item.abstract,
            }
            for item in state.literature_items
        ]
        out_df = pd.DataFrame(bib_rows).merge(screening_df, on="record_id", how="left")

    out_df.to_csv(state.output_path, index=False)
    print(f"Results saved to {state.output_path}")

    # -- Summary stats ------------------------------------------------------
    counts = {d: sum(1 for r in state.results if r.decision == d) for d in ("Include", "Exclude", "Maybe", "R2_REVIEW")}
    print(
        f"Summary: {counts['Include']} included, {counts['Exclude']} excluded, "
        f"{counts['Maybe']} maybe, {counts['R2_REVIEW']} R2 review — {len(state.results)} total"
    )

    # -- Markdown audit -----------------------------------------------------
    os.makedirs(state.audit_dir, exist_ok=True)
    audit_path = os.path.join(state.audit_dir, f"screening_audit_{state.batch_id}.md")
    _write_audit_markdown(state, audit_path)
    print(f"Audit saved to {audit_path}")

    return {}


def _write_audit_markdown(state: State, audit_path: str) -> None:
    """Write per-run audit markdown: metadata, code counts, flagged records, per-record reasoning."""
    lines: List[str] = []

    lines += [f"# Screening Audit — {state.batch_id}", ""]

    # Run metadata
    lines += ["## Run metadata", "", "| Field | Value |", "|-------|-------|"]
    for k, v in state.run_metadata.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    # Per-code exclusion counts
    code_counts: Dict[Any, int] = {}
    for r in state.results:
        if r.excl_code is not None:
            code_counts[r.excl_code] = code_counts.get(r.excl_code, 0) + 1

    lines += ["## Per-code exclusion counts", "", "| Code | Count |", "|------|-------|"]
    for code in sorted(code_counts):
        lines.append(f"| {code} | {code_counts[code]} |")
    lines.append("")

    # Decision summary
    decision_counts: Dict[str, int] = {}
    for r in state.results:
        decision_counts[r.decision] = decision_counts.get(r.decision, 0) + 1

    lines += ["## Decision summary", "", "| Decision | Count |", "|----------|-------|"]
    for dec, cnt in sorted(decision_counts.items()):
        lines.append(f"| {dec} | {cnt} |")
    lines.append("")

    # Flagged records (Maybe / R2_REVIEW)
    flagged = [r for r in state.results if r.decision in ("Maybe", "R2_REVIEW")]
    if flagged:
        lines += [
            "## Flagged records (Maybe / R2_REVIEW)", "",
            "| Record ID | Title | Decision | Note |",
            "|-----------|-------|----------|------|",
        ]
        for r in flagged:
            t = r.title[:60].replace("|", "\\|")
            n = r.note[:120].replace("|", "\\|")
            lines.append(f"| {r.record_id} | {t} | {r.decision} | {n} |")
        lines.append("")

    # Per-record detailed reasoning (LLM-screened only)
    llm_results = [r for r in state.results if r.screening_decision is not None]
    lines += ["## Per-record detailed reasoning", ""]
    if llm_results:
        header = (
            "| Record ID | Title | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 | C10 | C11 | C12 | "
            "Decision | Excl | Note |"
        )
        sep = (
            "|-----------|-------|----|----|----|----|----|----|----|----|----|-----|-----|-----|"
            "----------|------|------|"
        )
        lines += [header, sep]
        for r in llm_results:
            d = r.screening_decision

            def _fmt(ce: CodeEvaluation) -> str:
                mark = "✗" if ce.applies else "✓"
                return f"{mark} {ce.reasoning[:45].replace('|', chr(92) + '|')}"

            t = r.title[:40].replace("|", "\\|")
            n = r.note[:70].replace("|", "\\|")
            lines.append(
                f"| {r.record_id} | {t} | {_fmt(d.code_1)} | {_fmt(d.code_2)} | {_fmt(d.code_3)} | "
                f"{_fmt(d.code_4)} | {_fmt(d.code_5)} | {_fmt(d.code_6)} | {_fmt(d.code_7)} | "
                f"{_fmt(d.code_8)} | {_fmt(d.code_9)} | {_fmt(d.code_10)} | {_fmt(d.code_11)} | "
                f"{_fmt(d.code_12)} | {r.decision} | {r.excl_code or ''} | {n} |"
            )
    else:
        lines.append("_No LLM-screened records in this run._")
    lines.append("")

    with open(audit_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Graph definition
# ---------------------------------------------------------------------------

graph = (
    StateGraph(State, context_schema=Configuration)
    .add_node("load_literature", load_literature)
    .add_node("screen_literature", screen_literature)
    .add_node("generate_output", generate_output)
    .add_edge("__start__", "load_literature")
    .add_edge("load_literature", "screen_literature")
    .add_edge("screen_literature", "generate_output")
    .compile(name="Literature Screening Agent")
)
