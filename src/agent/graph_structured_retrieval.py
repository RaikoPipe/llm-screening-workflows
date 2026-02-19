from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, TypedDict

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, ValidationError
from typing_extensions import Optional
from loguru import logger

from src.utils.fulltext_manipulation import omit_sections_markdown
from src.utils.prompt_utils import load_prompt

RETRIEVAL_PROMPT = load_prompt("prompts/convert_to_json_prompt.md")

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

async def reason(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Analyze paper and reason about extraction requirements."""

    if state.literature_item.extra == "skip":
        logger.info(f"Skipping paper: {state.literature_item.title}")
        return {"result": None}

    llm = get_llm(config)

    schema = state.retrieval_form.model_json_schema()
    schema_json = json.dumps(schema, indent=2)

    text_to_screen = omit_sections_markdown(
        state.literature_item.fulltext,
        omit_sections=state.omit_titles
    )

    if len(text_to_screen.split()) > 12000:
        logger.warning(f'Fulltext exceeds 12000 words < {len(text_to_screen.split())}')

    messages = [
        SystemMessage(content=REASONING_PROMPT.format(
            title=state.literature_item.title,
            fulltext=text_to_screen,
            schema=schema))
    ]

    try:
        response = await llm.ainvoke(messages)
        return {
            "reasoning": response.content,
            "schema_instructions": schema_json
        }
    except Exception as e:
        logger.error(f"Reasoning error: {traceback.format_exc()}")
        return {"result": None}

async def convert_to_json(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Generate JSON based on reasoning."""

    llm = get_llm(config, json_mode=True)

    messages = [
        SystemMessage(content=RETRIEVAL_PROMPT.format(
            reasoning=state.reasoning,
            schema=state.schema_instructions)),
    ]

    try:
        response = await llm.ainvoke(messages)
        return {
            "raw_json": response.content,
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

async def repair_edit(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Repair validation errors by applying targeted JSON edits."""

    configuration = config.get("configurable", {})


    class JsonEdit(BaseModel):
        old_str: str
        new_str: str
        reason: str

    class JsonEditPlan(BaseModel):
        edits: list[JsonEdit]

    EDIT_PROMPT = """You are a JSON repair assistant. Create targeted string replacements to fix validation errors.

Pre-collected content of the paper as markdown as required by the JSON:
{reasoning}

Current JSON with errors:
{json_content}

Validation errors:
{errors}

Return a JSON with a list of 'edits', where each edit has:
- old_str: exact string to find (must appear exactly once)
- new_str: replacement string
- reason: explanation of the fix

Make minimal, precise edits targeting only the problematic fields."""

    messages = [
        SystemMessage(content=EDIT_PROMPT.format(
            json_content=state.raw_json,
            errors=state.validation_errors,
            reasoning=state.reasoning
        ))
    ]

    max_edit_iterations = 3
    broken_edits = []
    edited_json = state.raw_json

    llm = get_llm(config, json_mode=True, temperature=0.0)

    for edit_iteration in range(max_edit_iterations):
        try:
            response = await llm.ainvoke(messages)
            edit_plan = JsonEditPlan.model_validate_json(response.content)

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
                feedback = "\n".join([f"- {be.reason}: '{be.old_str}'" for be in broken_edits])
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

def get_llm(config: RunnableConfig, json_mode: bool = False, temperature: float = None) -> ChatOllama:
    cfg = config.get("configurable", {})
    return ChatOllama(
        model=cfg["model_name"],
        temperature=temperature if temperature is not None else cfg["temperature"],
        format="json" if json_mode else None
    )

# Build graph
graph = (
    StateGraph(State, context_schema=Configuration)
    .add_node("reason", reason)
    .add_node("generate", convert_to_json)
    .add_node("validate", validate)
    .add_node("repair", repair_edit)
    .add_edge("__start__", "reason")
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