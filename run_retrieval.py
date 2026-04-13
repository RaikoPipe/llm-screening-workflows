import os

from typing import List, Optional

from src.agent.graph_structured_retrieval import State as RetrievalState
from src.agent.graph_structured_retrieval import graph as graph_retrieval
from langchain_core.runnables import RunnableConfig
import asyncio
from pydantic import BaseModel, create_model, ValidationError
from dataclasses import dataclass
from slr_data_model import AISystem, SystemArchitecture
from tqdm.asyncio import tqdm

# load environment variables from .env file
from dotenv import load_dotenv

from src.utils import get_paper_collection
from loguru import logger

load_dotenv()

from pydantic.fields import FieldInfo

meta_titles = [
    "abstract", "references", "bibliography", "acknowledgments", "acknowledgements",
    "author contributions", "funding", "conflicts of interest", "conflict of interest",
    "appendix", "appendices", "supplementary material", "supplementary materials",
    "data availability", "code availability",
    "availability of data and materials", "Declaration of competing interest",
    "Conflict of interest statement",
]

introductory_titles = [
    "introduction", "motivation", "problem statement"
]

background_titles = ["literature review", "related work", "related works", "prior work", "previous work",
                     "state of the art", "state-of-the-art", "theoretical background", "theoretical framework"]

conclusion_titles = ["conclusion", "conclusions", "future work", "future directions", "future research",
                     "outlook", "limitations and future work", "discussion", "discussion and implications",
                     "implications"]

result_titles = [
    "results", "findings", "experimental results", "evaluation", "experiments",
    "experimental evaluation", "performance evaluation"
]

def select_fields(model: type[BaseModel], include: list[str]):
    return create_model(
        f"{model.__name__}Partial",
        **{
            name: (field.annotation, field)
            for name, field in model.model_fields.items()
            if name in include
        }
    )

async def run_retrieval(retrieval_schema: BaseModel, literature_item, omit_titles) -> BaseModel:
    """Run the literature screening process."""

    initial_state = RetrievalState(
        retrieval_form=retrieval_schema,
        literature_item=literature_item,
        omit_titles=omit_titles,
        max_validation_attempts=2
    )

    config = RunnableConfig(
        configurable={
            "model_name": "gpt-oss:120b", #gpt-oss:120b" # claude-sonnet-4-6
            "temperature": 0.0,
            "num_ctx": 64000,
            "reasoning": False,
            "skip_analysis": False,
            "word_count_limit": 17000,
            "skip_on_word_count_limit": True
        }
    )

    result = await graph_retrieval.ainvoke(initial_state, config=config)
    return result

def get_doi_based_filename(doi: str, suffix: str) -> str:
    """Generate a filename based on DOI."""
    safe_doi = doi.replace('/', '_')
    return f"{safe_doi}_{suffix}.json"

@dataclass
class LiteratureItem:
    """Represents a literature item with title and abstract."""

    title: str
    doi: str
    abstract: str
    fulltext: str = ""  # Placeholder for full text if needed
    extra: str = ""

def load_literature(collection_key) -> List[LiteratureItem]:
    """Load literature items from the literature folder."""
    logger.info(f"Loading collection {collection_key}")
    literature_items = []

    paper_collection = get_paper_collection(collection_key=collection_key, get_fulltext="parsed")

    for idx, paper in paper_collection.iterrows():
        if not isinstance(paper.fulltext, str) or paper.fulltext == "":
            logger.warning("Paper fulltext not found, skipping paper: " + paper.title)
            continue
        else:
            literature_items.append(LiteratureItem(title=paper.title, abstract=paper.abstractNote, doi =paper.DOI, fulltext=paper.fulltext, extra=paper.extra))

    logger.info(f"Loaded {len(literature_items)} literature items")
    return literature_items

def dump_output(title, doi, output, reasoning):
    """flatten schema and save retrieval results as JSON"""

    # encode as list when set is encountered
    import json
    class SetEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, set):
                return list(obj)
            return json.JSONEncoder.default(self, obj)

    # save output as json
    output = {
            'title': title,
            'doi': doi,
            'retrieval': output.model_dump(),
            'reasoning': reasoning
        }

    with open("outputs/" + get_doi_based_filename(doi, "retrieval"), encoding='utf-8', mode='w') as f:
        import json
        json.dump(output, f, ensure_ascii=False, indent=4, cls=SetEncoder)


    return {'status': 'output saved as JSON'}

def get_part_retrieval(part_name, part_schema: BaseModel, literature_item, omit_titles: Optional[List[str]] = None):
    #logger.info(f"Running retrieval for {part_name}...")
    part_result = asyncio.run(run_retrieval(part_schema, literature_item, omit_titles))

    # check if result[part_name] is empty
    if not part_result["result"]:
        raise ValueError(f"Retrieval for {part_name} returned empty result for paper: {literature_item.title}")

    return part_result["result"], part_result["reasoning"]

def orchestrate_decomposed_retrieval(literature_item):
    if paper_processed(literature_item):
        logger.info(f"Paper was already processed: {literature_item.title}. Skipping...")
        return

    # decompose the schema for distributed retrieval
    agents = select_fields(
        SystemArchitecture,
        include=["agents"]
    )

    system_architecture = select_fields(
        SystemArchitecture,
        include=["orchestration", "trigger", "human_integration"]
    )

    domain = select_fields(
        AISystem,
        include=["application_domain", "application_domain_description"]
    )

    subject = select_fields(
        AISystem,
        include=["problem_description", "proposed_solution", "research_methodology", "research_methodology_description", "research_maturity"]
    )

    reported_outcomes = select_fields(
        AISystem,
        include=["reported_outcomes", "validation_methods"]
    )

    try:
        # run retrieval for each part
        result = {}
        reasoning = {}
        logger.debug("Processing agents")
        result["agents"], reasoning["agents"] = get_part_retrieval(
            "agents",
            agents,
            literature_item,
            [*meta_titles, *introductory_titles, *background_titles, *conclusion_titles, *result_titles])

        logger.debug("Processing system architecture")
        result["system_architecture"], reasoning["system_architecture"] = get_part_retrieval(
            "system_architecture",
            system_architecture,
            literature_item,
            [*meta_titles, *introductory_titles, *background_titles, *conclusion_titles, *result_titles])
        logger.debug("Processing domain")
        result["domain"], reasoning["domain"] = get_part_retrieval(
            "domain",
            domain,
            literature_item,
            [*meta_titles, *background_titles, *conclusion_titles, *result_titles])
        logger.debug("Processing subject")
        result["subject"], reasoning["subject"] = get_part_retrieval(
            "subject",
            subject,
            literature_item,
            [*meta_titles, *background_titles, *conclusion_titles, *result_titles])
        logger.debug("Processing reported outcomes")
        result["reported_outcomes"], reasoning["reported_outcomes"] = get_part_retrieval(
            "reported_outcomes",
            reported_outcomes,
            literature_item,
            [*meta_titles, *introductory_titles, *background_titles])


        system_architecture = SystemArchitecture(
            agents=result["agents"].agents,
            orchestration=result["system_architecture"].orchestration,
            trigger=result["system_architecture"].trigger,
            human_integration=result["system_architecture"].human_integration
        )

        ai_system = AISystem(
            system_architecture=system_architecture,
            application_domain=result["domain"].application_domain,
            application_domain_description=result["domain"].application_domain_description,
            problem_description=result["subject"].problem_description,
            proposed_solution=result["subject"].proposed_solution,
            validation_methods=result["reported_outcomes"].validation_methods,
            reported_outcomes=result["reported_outcomes"].reported_outcomes,
            research_maturity=result["subject"].research_maturity,
            research_methodology=result["subject"].research_methodology,
            research_methodology_description=result["subject"].research_methodology_description
        )
    except Exception as e:
        logger.error(f"Retrieval failed: Error occurred while constructing final data model: {e} \n Skipping paper due to error.")
        return

    dump_output(
        title=literature_item.title,
        doi=literature_item.doi,
        output=ai_system,
        reasoning=reasoning
    )

    logger.success(f"Retrieval completed and saved for paper: {literature_item.title}")

def orchestrate_partial_retrieval_and_append(literature_item):
    # get existing paper
    import json
    with open("outputs/" + get_doi_based_filename(literature_item.doi, "retrieval"), encoding='utf-8', mode='r') as f:
        data = json.load(f)

    retrieval = data.get("retrieval")

    if retrieval.get("integration_with_cps"):
        # skip paper
        logger.info(f"Paper was already processed: {literature_item.title}. Skipping...")
        return

    # decompose the schema for distributed retrieval
    results_and_limitations = select_fields(
        AISystem,
        include=["integration_with_cps", "baseline_methods", "evaluation_metric_categories", "identified_research_gaps", "scalability_assessment"]
    )

    try:
        # run retrieval for each part
        result = {}
        reasoning = {}
        logger.debug("Processing retrieval")
        result["results_and_limitations"], reasoning["results_and_limitations"] = get_part_retrieval(
            "agents",
            results_and_limitations,
            literature_item,
            meta_titles)

        # modify json
        retrieval["integration_with_cps"] = result["results_and_limitations"].integration_with_cps
        retrieval["baseline_methods"] = result["results_and_limitations"].baseline_methods
        retrieval["evaluation_metric_categories"] = result["results_and_limitations"].evaluation_metric_categories
        retrieval["identified_research_gaps"] = result["results_and_limitations"].identified_research_gaps
        retrieval["scalability_assessment"] = result["results_and_limitations"].scalability_assessment

        # save and close
        with open("outputs/" + get_doi_based_filename(literature_item.doi, "retrieval"), encoding='utf-8', mode='w') as f:
            json.dump(data, f, indent=2)

    except Exception as e:
        logger.error(f"Retrieval failed: Error occurred while constructing final data model: {e} \n Skipping paper due to error.")
        return

    logger.success(f"Retrieval completed and saved for paper: {literature_item.title}")


def orchestrate_retrieval(literature_item):
    if paper_processed(literature_item):
        logger.info(f"Paper was already processed: {literature_item.title}. Skipping...")
        return
    # standard retrieval, no decomp
    try:
        # run retrieval
        retrieval = asyncio.run(run_retrieval(AISystem, literature_item, omit_titles=meta_titles))
        result = retrieval["result"]
        reasoning = retrieval["reasoning"]

    except Exception as e:
        logger.error(f"Retrieval failed: Error occurred while constructing final data model: {e} \n Skipping paper due to error.")
        return
    if result == "skip":
        return

    dump_output(
        title=literature_item.title,
        doi=literature_item.doi,
        output=result,
        reasoning=reasoning
    )

    logger.success(f"Retrieval completed and saved for paper: {literature_item.title}")

def paper_processed(paper):
    # check if item was already processed
    output_filename = get_doi_based_filename(paper.doi, "retrieval")
    if os.path.exists("outputs/" + output_filename):
        return True
    return False

# Example usage
if __name__ == "__main__":
    literature = load_literature(collection_key="UF8TVRYZ")

    for item in tqdm(literature, desc="Retrieve", unit="item"):
        logger.info(f"Processing paper: {item.title}")
        orchestrate_decomposed_retrieval(item)




