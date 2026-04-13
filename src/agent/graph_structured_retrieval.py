from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, TypedDict

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, ValidationError
from typing_extensions import Optional
from loguru import logger

from src.utils.fulltext_manipulation import omit_sections_markdown
from src.utils.prompt_utils import load_prompt
from src.utils.pydantic_utils import extract_json
import json_repair

RETRIEVAL_PROMPT = load_prompt("prompts/structured_retrieval_prompt.md")
RETRIEVAL_PROMPT_THINKING = load_prompt("prompts/structured_retrieval_prompt_thinking.md")
EDIT_PROMPT = load_prompt('prompts/edit_prompt.md')
REASONING_PROMPT = load_prompt("prompts/reasoning_prompt.md")

VALIDATION_PROMPT = """You are a JSON repair assistant. Fix ONLY the specific validation errors in the JSON.

Original schema:
{schema}

Reasoning about JSON content:
{reasoning}

Current JSON with errors:
{json_content}

Validation errors:
{errors}

Return ONLY the corrected JSON with the problematic fields fixed."""

class Configuration(TypedDict):
    model_name: str
    temperature: float

@dataclass
class LiteratureItem:
    title: str
    doi: str
    abstract: str
    fulltext: str = ""
    extra: str = ""

@dataclass
class State:
    retrieval_form: Optional[BaseModel]
    literature_item: LiteratureItem
    result: Optional[BaseModel] = field(default_factory=dict)
    reasoning: Optional[str] = ""
    raw_json: Optional[str] = None
    validation_errors: Optional[str] = None
    validation_attempts: int = 0
    max_validation_attempts: int = 5
    schema_instructions: Optional[str] = None
    omit_titles : list[str] = None
    reduced_paper_text: str = None
    dummy_human_messages: list[HumanMessage] = None

async def prepare_retrieval(state: State, config: RunnableConfig) -> Dict[str, Any]:
    if state.literature_item.extra == "skip":
        logger.info(f"Skipping paper: {state.literature_item.title}")
        return {"result": "skip"}

    text_to_screen = omit_sections_markdown(
        state.literature_item.fulltext,
        omit_sections=state.omit_titles
    )

    cfg = config.get("configurable", {})

    dummy_human_messages = []
    if "claude" in cfg["model_name"]:
        dummy_human_messages = [HumanMessage(content="Please perform the task as described in the system message")]

    schema = state.retrieval_form.model_json_schema()
    schema_json = json.dumps(schema, indent=2)

    if len(text_to_screen.split()) > cfg["word_count_limit"]:
        logger.warning(f'Fulltext exceeds {cfg["word_count_limit"]} words < {len(text_to_screen.split())}')
        if cfg["skip_on_word_count_limit"]:
            logger.info(f"Skipping paper due to word limit hit: {state.literature_item.title}")
            return {"result": "skip"}

    return {
        "schema_instructions": schema_json,
        "reduced_paper_text": text_to_screen,
        "dummy_human_messages": dummy_human_messages
    }

async def generate_analysis(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Analyze paper and reason about extraction requirements."""

    llm = get_llm(config)

    cfg = config.get("configurable", {})
    if cfg["reasoning"] or cfg["skip_analysis"]:
        # thinking done by model; skip reasoning node
        return {}

    messages = [
        SystemMessage(content=REASONING_PROMPT.format(
            title=state.literature_item.title,
            fulltext=state.reduced_paper_text,
            schema=state.schema_instructions))
    ] + state.dummy_human_messages

    try:
        response = await llm.ainvoke(messages)
        return {
            "reasoning": response.content,
        }
    except Exception as e:
        logger.error(f"Reasoning error: {traceback.format_exc()}")
        return {"result": None}

async def generate_json(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Generate JSON based on reasoning."""

    llm = get_llm(config, json_mode=True)

    cfg = config.get("configurable", {})

    if cfg["reasoning"] or cfg["skip_analysis"]:
        # thinking enabled; convert full paper
        messages = [
            SystemMessage(content=RETRIEVAL_PROMPT_THINKING.format(
                schema=state.schema_instructions,
                fulltext=state.reduced_paper_text)),
        ]  + (state.dummy_human_messages or [HumanMessage(content=".")])
    else:
        messages = [
            SystemMessage(content=RETRIEVAL_PROMPT.format(
                schema=state.schema_instructions,
                reasoning=state.reasoning)),
        ] + (state.dummy_human_messages or [HumanMessage(content=".")])

    try:
        response = await llm.ainvoke(input=messages)

        # remove generation artifacts
        cleaned_content = extract_json(response.content)

        return {
            "raw_json": cleaned_content,
            "validation_attempts": 0
        }
    except Exception as e:
        logger.error(f"Generation error: {traceback.format_exc()}")
        return {"result": None}

async def validate(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Validate and parse JSON."""

    try:
        result = state.retrieval_form.model_validate_json(state.raw_json)
        return {"result": result, "validation_errors": None}

    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        return {
            "validation_errors": e.json(indent=2),
            "validation_attempts": state.validation_attempts + 1
        }

    except json.JSONDecodeError as e:
        logger.warning(f"JSON decode error: {e}")
        return {
            "validation_errors": f"JSON decode error: {str(e)}",
            "validation_attempts": state.validation_attempts + 1
        }

async def fix_validation_errors(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Repair validation errors by applying targeted JSON edits."""

    configuration = config.get("configurable", {})


    class JsonEdit(BaseModel):
        old_str: str
        new_str: str
        reason: str

    class JsonEditPlan(BaseModel):
        edits: list[JsonEdit]

    messages = [
        SystemMessage(content=EDIT_PROMPT.format(
            json_content=state.raw_json,
            errors=state.validation_errors,
            reasoning=state.reasoning
        ))
    ] + state.dummy_human_messages

    max_edit_iterations = 3
    broken_edits = []
    edited_json = state.raw_json

    llm = get_llm(config, json_mode=True, temperature=0.0)

    for edit_iteration in range(max_edit_iterations):
        try:
            response = await llm.ainvoke(messages)
            cleaned_response = extract_json(response.content)
            edit_plan = JsonEditPlan.model_validate_json(cleaned_response)

            for edit in edit_plan.edits:
                try:
                    if edited_json.count(edit.old_str) == 1:
                        edited_json = edited_json.replace(edit.old_str, edit.new_str, 1)
                    elif edited_json.count(edit.old_str) > 1:
                        raise ValueError(f"Multiple occurrences: {edit.reason} for '{edit.old_str}'")
                    else:
                        raise ValueError(f"String not found: {edit.reason} for '{edit.old_str}'")
                except ValueError as ve:
                    logger.warning(str(ve))
                    broken_edits.append(edit)

            if not broken_edits:
                break
            else:
                feedback = "\n".join([f"- {be.generate_analysis}: '{be.old_str}'" for be in broken_edits])
                messages.append(HumanMessage(
                    content=f"These edits failed:\n{feedback}\n\nRevise your edit plan with more specific strings."
                ))
                broken_edits = []

        except Exception as e:
            logger.error(f"Edit iteration {edit_iteration} error: {e}")
            break

    return {"raw_json": edited_json}

def should_repair(state: State) -> str:
    """Route based on validation status."""

    if state.result and state.result != {}:
        return END

    if state.validation_errors is None:
        return "validate"

    if state.validation_attempts >= state.max_validation_attempts:
        logger.error(f"Max validation attempts reached: {state.literature_item.title}")
        return END

    return "repair"

def should_process(state:State)-> str:

    if state.result == "skip":
        return END

    return "continue"

def get_llm(config: RunnableConfig, json_mode: bool = False, temperature: float = None) -> Any:
    cfg = config.get("configurable", {})
    if "claude" in cfg["model_name"]:
        return ChatAnthropic(
            model_name=cfg["model_name"],
            temperature=temperature if temperature is not None else cfg["temperature"],
            thinking={"type": "enabled"} if cfg["reasoning"] else None
        )
    else:
        if "gpt-oss" in cfg["model_name"] and cfg["reasoning"]:
            # gpt-oss is the only current model to support reasoning levels
            cfg["reasoning"] = "high"

        return ChatOllama(
            model=cfg["model_name"],
            reasoning=cfg["reasoning"],
            temperature=temperature if temperature is not None else cfg["temperature"],
            format="json" if json_mode else None
        )


# Build graph
graph = (
    StateGraph(State, context_schema=Configuration)
    .add_node("prepare_retrieval", prepare_retrieval)
    .add_node("reason", generate_analysis)
    .add_node("generate", generate_json)
    .add_node("validate", validate)
    .add_node("repair", fix_validation_errors)
    .add_edge("__start__", "prepare_retrieval")
    .add_conditional_edges("prepare_retrieval", should_process, {
        "continue": "reason",
        END: END
    })
    .add_edge("reason", "generate")
    .add_edge("generate", "validate")
    .add_conditional_edges("validate", should_repair, {
        "repair": "repair",
        "validate": "validate",
        END: END
    })
    .add_edge("repair", "validate")
    .compile(name="Literature Screening Agent SLR")
)