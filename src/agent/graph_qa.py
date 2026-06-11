"""LangGraph literature screening agent.

Screens literature titles and abstracts based on exclusion criteria.
"""

from __future__ import annotations

import os
import csv
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, TypedDict, Optional
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from langgraph.graph import StateGraph
from src.utils import get_paper_collection, get_llm, remove_section
from pydantic import BaseModel, Field
import pandas as pd
from tqdm.asyncio import tqdm


class Configuration(TypedDict):
    """Configurable parameters for the agent."""

    model_name: str
    temperature: float


@dataclass
class LiteratureItem:
    """Represents a literature item with title and abstract."""

    title: str
    doi: str
    abstract: str
    fulltext: str = ""  # Placeholder for full text if needed

@dataclass
class QAResult:
    """Structured output for literature quality assessment."""

    title: str
    doi: str
    qa_decision: Optional[QADecision]

class QADecision(BaseModel):
    """Structured output for literature quality assessment decision."""

    methodological_rigor_reasoning: str = Field(description="Single sentence reasoning on choice of methodological rigor scoring")
    methodological_rigor_score: int = Field(ge=0, le=2,description="Score for criterion methodological rigor (2: Highest, 1: Moderate, 0: Lowest)")

    clarity_research_design_reasoning: str = Field(description="Single sentence reasoning on choice of methodological rigor scoring")
    clarity_research_design_score: int = Field(ge=0, le=2,description="Score for criterion clarity of research design (2: Highest, 1: Moderate, 0: Lowest)")

    validation_methods_reasoning: str = Field(description="Single sentence reasoning on choice of methodological rigor scoring")
    validation_methods_score: int = Field(ge=0, le=2,description="Score for criterion validation methods (2: Highest, 1: Moderate, 0: Lowest)")

@dataclass
class State:
    """State for the literature screening agent."""

    qa_criteria: str = ""
    collection_key: str = field(default_factory=lambda: os.environ.get("ZOTERO_COLLECTION_KEY", ""))
    literature_items: List[LiteratureItem] = field(default_factory=list)
    results: List[QAResult] = field(default_factory=list)
    output_path: str = "qa_results.csv"


async def load_literature(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Load literature items from the literature folder."""
    literature_items = []
    if not state.literature_items:
        paper_collection = get_paper_collection(collection_key=state.collection_key, get_fulltext=True)
    else:
        # assume literature items in correct format
        paper_collection = state.literature_items

    for idx, paper in paper_collection.iterrows():
        literature_items.append(LiteratureItem(title=paper.title, abstract=paper.abstractNote, doi =paper.DOI, fulltext=paper.fulltext))

    print(f"Loaded {len(literature_items)} literature items")
    return {"literature_items": literature_items}

async def qa_literature(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """QA literature item based on quality assessment criteria, with up to 3 retries on error."""

    llm_agent = get_llm(config).with_structured_output(QADecision)

    system_prompt = """You are a professional literature reviewer for high ranking scientific journals in the field of computer science and engineering. Perform a quality assessment of the given paper against the quality criteria given by the user.
    ALWAYS return your answer as a JSON with reasoning and scoring attributes (brief explanation per quality criterion using the Scale [Highest: 2, Moderate: 1, Lowest: 0])."""

    results = []



    for item in tqdm(state.literature_items, desc="Screening literature", unit="item"):
        attempt = 0

        text_to_qa = remove_section(item.fulltext)
        while attempt < 3:
            try:
                human_prompt = f"""

    Title: {item.title}

    Fulltext: {text_to_qa}
    
    Quality Assessment Criteria: {state.qa_criteria}

    Which scoring should this paper receive in regards to each quality criterion?
    """

                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=human_prompt)
                ]

                response = await llm_agent.ainvoke(messages)

                result = QAResult(
                    title=item.title,
                    doi=item.doi,
                    qa_decision = response
                )
                results.append(result)
                break  # Success, exit retry loop

            except Exception as e:
                attempt += 1
                if attempt == 3:
                    print(f"Error screening {item.title} after 3 attempts: {e}")
                    # Default to exclusion on repeated error
                    result = QAResult(
                        title=item.title,
                        doi=item.doi,
                        qa_decision=None
                    )
                    results.append(result)

    return {"results": results}

async def generate_csv(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Generate CSV file with screening results."""

    try:
        with open(state.output_path, 'w', newline='', encoding='utf-8') as csvfile:
            df = pd.DataFrame([
                {
                    'title': r.title,
                    'doi': r.doi,
                    **(r.qa_decision.model_dump() if r.qa_decision else {})
                }
                for r in state.results
            ])

            df.to_csv(state.output_path, index=False)

        print(f"Results saved to {state.output_path}")

        # Calculate average score
        score_clarity = sum(r.qa_decision.clarity_research_design_score for r in state.results) / len(state.results)
        score_methodological = sum(r.qa_decision.methodological_rigor_score for r in state.results) / len(state.results)
        score_validation = sum(r.qa_decision.validation_methods_score for r in state.results) / len(state.results)
        total_average = (score_clarity + score_methodological + score_validation) / 3
        print(f"Average score for clarity of research design: {score_clarity:.2f}")
        print(f"Average score for methodological rigor: {score_methodological:.2f}")
        print(f"Average score for validation methods: {score_validation:.2f}")
        print(f"Overall average score: {total_average:.2f}")


    except Exception as e:
        print(f"Error generating CSV: {e}")

    return {}


# Define the graph

graph = (
    StateGraph(State, context_schema=Configuration)
    .add_node("load_literature", load_literature)
    .add_node("screen_literature", qa_literature)
    .add_node("generate_csv", generate_csv)
    .add_edge("__start__", "load_literature")
    .add_edge("load_literature", "screen_literature")
    .add_edge("screen_literature", "generate_csv")
    .compile(name="Literature Screening Agent")
)
