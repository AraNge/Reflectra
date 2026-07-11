from __future__ import annotations

import os
from typing import Any

from openai import OpenAI


def resolve_api_key(
    api_key: str | None = None,
    api_key_env: str | None = None,
    base_url: str | None = None,
    config: dict[str, Any] | None = None,
) -> str | None:
    llm_config = (config or {}).get("llm", {})
    configured_key = api_key or llm_config.get("api_key")
    if configured_key:
        return str(configured_key)

    env_name = api_key_env or llm_config.get("api_key_env") or "OPENAI_API_KEY"
    env_key = os.environ.get(str(env_name))
    if env_key:
        return env_key

    if base_url or llm_config.get("base_url"):
        return "not-needed"

    return None


def create_openai_client(
    *,
    api_key: str | None = None,
    api_key_env: str | None = None,
    base_url: str | None = None,
    config: dict[str, Any] | None = None,
) -> OpenAI:
    llm_config = (config or {}).get("llm", {})
    resolved_base_url = base_url or llm_config.get("base_url") or None
    resolved_api_key = resolve_api_key(
        api_key=api_key,
        api_key_env=api_key_env,
        base_url=resolved_base_url,
        config=config,
    )

    kwargs: dict[str, Any] = {}
    if resolved_api_key:
        kwargs["api_key"] = resolved_api_key
    if resolved_base_url:
        kwargs["base_url"] = str(resolved_base_url)

    return OpenAI(**kwargs)
