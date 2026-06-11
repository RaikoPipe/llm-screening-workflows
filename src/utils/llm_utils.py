from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama


def get_llm(config: RunnableConfig, json_mode: bool = False, temperature: float = None) -> Any:
    load_dotenv()
    cfg = config.get("configurable", {})
    model_name = cfg.get("model_name") or os.getenv("LLM_MODEL", "gpt-oss:120b")

    if "claude" in model_name:
        return ChatAnthropic(
            model_name=model_name,
            temperature=temperature if temperature is not None else cfg.get("temperature", 0.0),
            thinking={"type": "enabled"} if cfg.get("reasoning") else None,
        )

    llm_endpoint = os.getenv("LLM_ENDPOINT")
    header_auth_key = os.getenv("HEADER_AUTH_KEY")

    reasoning = cfg.get("reasoning")
    if "gpt-oss" in model_name and reasoning:
        reasoning = "high"

    kwargs: dict[str, Any] = dict(
        model=model_name,
        temperature=temperature if temperature is not None else cfg.get("temperature", 0.0),
        format="json" if json_mode else None,
    )
    if reasoning is not None:
        kwargs["reasoning"] = reasoning
    if llm_endpoint:
        kwargs["base_url"] = llm_endpoint
    if header_auth_key:
        kwargs["client_kwargs"] = {"headers": {"Authorization": header_auth_key}}

    return ChatOllama(**kwargs)
