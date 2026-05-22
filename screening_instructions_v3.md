# Screening Instructions v3 — Strategic Manufacturing & Logistics Planning with LLMs

> **Pre-registration reference document.**
> Pipeline implementation: `src/agent/graph_screening.py` · Entry point: `run_exclusion_screening.py`

---

## §1 — PICOC Definition & Scope

| Frame | Definition |
|-------|-----------|
| **Population (P)** | Manufacturing and logistics organisations conducting strategic planning |
| **Intervention (I)** | Use of Large Language Models (LLMs) as a core computational component in planning or decision-making systems |
| **Comparator (C)** | Conventional optimisation, simulation, or human-expert planning (implicit) |
| **Outcome (O)** | Improved or automated strategic planning/decision quality demonstrated through case study, experiment, or prototype evaluation |
| **Context (C)** | One or more of five in-scope subdomains (see below) |

### In-scope subdomains

1. **Factory planning** — facility layout, capacity planning, production network design
2. **Supply-chain (SC) network design** — supplier selection, network configuration, inventory strategy
3. **Strategic logistics / distribution-network design** — warehouse location, transport-mode selection, last-mile network design
4. **Technology selection & investment appraisal** — make-or-buy, technology roadmapping, capex decisions
5. **Global production network (GPN) design** — reshoring, offshoring, multi-site configuration

### Inclusion boundary rules

**Physical-system-configuration criterion:** Papers that configure or redesign a physical manufacturing or logistics system at a strategic horizon qualify even if they do not use the word "planning."

**Upper-tactical boundary rule:** Papers focused purely on production scheduling, operational dispatch, or real-time control — without a strategic or structural decision component — are out of scope (apply EC1).

---

## §2 — Exclusion Codes

Twelve codes are registered. Codes **EC1–EC6** and **EC10** are evaluated by the LLM. Codes **EC7–EC9**, **EC11–EC12** are metadata pre-filters applied deterministically.

### LLM-assessed codes

| Code | Name | Description |
|------|------|-------------|
| **EC1** | Out-of-scope domain | Primary contribution does not address one of the five in-scope subdomains. Apply physical-system-configuration inclusion criterion and upper-tactical boundary rule. |
| **EC2** | LLM not used as core component | Paper mentions LLMs but does not use them as a core computational component for planning/decision-making. Applies when LLM used only for text pre-processing, literature search, summarisation unrelated to planning, or mentioned only in future-work sections. |
| **EC3** | Non-strategic planning scope | Study addresses only operational/tactical tasks (scheduling, real-time dispatch, shop-floor control) with no strategic or structural planning component. Strategic horizon = multi-year impact on system configuration, capacity, or network topology. |
| **EC4** | No LLM-driven planning system | Planning system relies solely on classical optimisation, simulation, or rule-based methods without any LLM integration. Papers comparing LLM-based and non-LLM approaches are in-scope. |
| **EC5** | Purely conceptual, no implementation | Only theoretical frameworks, position statements, or conceptual architectures without implementation, prototype, or empirical evaluation. |
| **EC6** | Grey literature below quality threshold | Workshop abstract, position paper, extended abstract (<4 pages), non-peer-reviewed technical report, or white paper. arXiv/SSRN preprints are **not** automatically excluded — assess on content quality. |
| **EC10** | Secondary literature only | Systematic review, scoping review, meta-analysis, or bibliometric study without a novel LLM-based planning system. Reviews that also propose and evaluate a new framework/tool are in-scope. |

### Metadata pre-filter codes (deterministic)

| Code | Name | Description |
|------|------|-------------|
| **EC7** | Out of date range | Published before 2020. |
| **EC8** | Non-English language | Full text not written in English. |
| **EC9** | Excluded document type | Thesis, dissertation, book, book chapter, editorial, letter to the editor, or keynote abstract without substantive technical content. |
| **EC11** | Duplicate publication | Same study published multiple times (journal + conference, preprint + published). Retain the most complete version. |
| **EC12** | Full text not available | Full text cannot be retrieved and abstract alone is insufficient for a decision. |

---

## §3 — Boundary-Case Decision Rules

Apply these rules in the order listed when a paper falls near a boundary. They are embedded verbatim in the LLM system prompt.

| # | Scenario | Rule |
|---|----------|------|
| 1 | LLM mentioned but not used | Exclude **EC2**. If an LLM is cited/named but does not function as a computational component in the reported system, apply EC2. |
| 2 | Digital twin paper | Include **only if** the LLM is used for model generation, parameter estimation, or strategic decision support. Exclude EC2 if the LLM is peripheral or absent. |
| 3 | Simulation/optimisation with LLM in future work only | Exclude **EC2**. Mentioning LLM integration as future work does not qualify. |
| 4 | Paper addresses both strategic and operational levels | Include if strategic level is the primary contribution. If strategic content is incidental, Exclude **EC3**. |
| 5 | SC visibility/monitoring without planning | Exclude **EC3**. Monitoring systems without a structural planning decision component are out of scope. |
| 6 | LLM framework tested on toy examples only | **Maybe** (ESCALATE). Flag for human review if evaluation is limited to synthetic or trivial benchmarks. |
| 7 | Survey/review that also introduces a classification framework | **Maybe** (ESCALATE) if the new framework is substantive; otherwise Exclude **EC10**. |
| 8 | Preprint (arXiv, SSRN) without peer review | Do **not** automatically apply EC6. Assess on content quality; apply EC6 only if quality is clearly insufficient. |
| 9 | LLM used for data preprocessing only | Exclude **EC2**. Pre-processing (OCR, entity extraction, translation) without LLM-driven planning does not qualify. |
| 10 | Industrial case study without novel method | Include if a deployed LLM-based system is described and evaluated for strategic planning, even without a novel algorithmic contribution. |
| 11 | Paper covers multiple domains, one of which is in scope | Include if the in-scope subdomain is substantive (≥25% of contribution). Exclude **EC1** if in-scope content is incidental. |
| 12 | AI-assisted design without explicit LLM | Exclude **EC2**. Papers using earlier AI methods (ML, expert systems, evolutionary algorithms) without transformer-based LLMs do not qualify. |

---

## §4 — Tri-Level Decision Output

The LLM returns one of three decisions per paper:

| Decision | Meaning | `excl_code` | `note` |
|----------|---------|-------------|--------|
| **Include** | Paper is in scope; no exclusion criterion applies | *(empty)* | Brief reasoning |
| **Exclude** | At least one exclusion criterion applies | First triggered code (e.g. `EC2`) | Brief reasoning |
| **Maybe** | Borderline, ambiguous, or boundary rule calls for ESCALATE | *(empty)* | `ESCALATE: <reason>` |

**Conservative screening principle:** When genuinely uncertain between Include and Exclude, choose **Maybe** rather than Exclude. Do not exclude papers without clear evidence from the text.

`Maybe` records are listed in the flagged-records section of the audit file and must be resolved by a human reviewer before proceeding.

---

## §5 — Metadata Pre-Filters (EC7–EC9, EC11–EC12)

These codes are evaluated deterministically before LLM screening, using structured bibliographic metadata:

- **EC7** — check publication year field
- **EC8** — check language field
- **EC9** — check item type field
- **EC11** — deduplicate by DOI, title similarity (rapidfuzz), and author+year
- **EC12** — check whether full text was successfully retrieved

Papers flagged by a metadata pre-filter bypass LLM screening and are appended to the CSV with `decision=Exclude` and the corresponding `excl_code`.

---

## §6 — Overlapping Authorship

Papers where R1 (Richard Reider) is a co-author must **not** be screened by the LLM. They are automatically flagged with `decision=MANUAL_R2` and routed to R2 for independent manual review.

Set `overlapping_authorship=True` on the corresponding `LiteratureItem` (or in the `overlapping_authorship` column of the input CSV) to trigger this routing.

---

## §7 — Pilot Validation (Stage 2)

Before Stage 3 (full-corpus screening) can proceed, the pipeline must pass a pilot validation on a **stratified 100-record sample** labelled by R1.

Run with `--pilot --stage 2`. The pipeline computes:

| Metric | Threshold |
|--------|-----------|
| Sensitivity (recall of includes) | ≥ 0.95 |
| Specificity (recall of excludes) | ≥ 0.70 |
| Cohen's κ | ≥ 0.70 |
| Gwet's AC1 | reported, no threshold |

`Maybe` decisions are mapped to **Include** (conservative) for binary metric computation.

Pilot metrics are appended to the markdown audit file.

---

## §8 — Prompt Version Control

| Field | Value |
|-------|-------|
| Prompt version | v3.0 |
| System prompt hash | computed per run (SHA-256, first 12 hex chars) |
| Logged in | `screening_audit_<batch_id>.md` → Run Metadata table |

Changes to `PICOC_DEFINITION`, `BOUNDARY_RULES`, or the system prompt template increment the prompt version and must be recorded in this document.

---

## §9 — Logging Format

### CSV columns appended to original input file

Screening columns are appended **in-place** to the original bibliographic CSV, preserving all existing columns.

| Column | Type | Description |
|--------|------|-------------|
| `stage` | int | Screening stage (2 = pilot, 3 = title/abstract) |
| `reviewer` | str | `"LLM"` for automated runs |
| `decision` | str | `Include` / `Exclude` / `Maybe` / `MANUAL_R2` |
| `excl_code` | str | First triggered exclusion code, or empty |
| `note` | str | Brief LLM reasoning or `ESCALATE: <reason>` |
| `timestamp` | str | ISO 8601 UTC |

### Markdown audit file (`screening_audit_<batch_id>.md`)

Generated per run in `--audit-dir`. Contains:

1. **Run metadata** — model, temperature, prompt version, prompt hash, stage, timestamp, record count, batch ID
2. **Decision summary** — counts per decision type
3. **Per-code exclusion counts** — how many records triggered each code
4. **Per-record detailed reasoning table** — record number, DOI, title, decision, excl code, note
5. **Flagged records** — all Maybe / ESCALATE / MANUAL_R2 items with full per-code reasoning

---

## §10 — Change Log

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | 2024 | Initial 3-code criteria (EC1–EC3), binary exclusion, agentic AI / context engineering scope |
| v2.0 | 2025 | Expanded to 5 codes, RQ1–RQ4 on agentic AI / production & logistics scope |
| v3.0 | 2026-05-22 | Revised to 12 codes, PICOC scope (strategic manufacturing/logistics), tri-level output, boundary-case rules, in-place CSV logging, markdown audit, pilot validation mode |
