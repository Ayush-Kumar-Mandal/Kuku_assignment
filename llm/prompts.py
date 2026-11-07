"""Load prompts from prompts/*.yaml into ChatPromptTemplates. No prompt text lives in code."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ConfigDict


class PromptFile(BaseModel):
    model_config = ConfigDict(extra="allow")   # agent files may carry extra fragments
    version: int = 1
    system: str | None = None
    human: str | None = None


@lru_cache(maxsize=None)
def load_prompt_file(prompts_dir: str, name: str) -> PromptFile:
    path = Path(prompts_dir) / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"prompt file not found: {path}")
    return PromptFile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def load_chat_prompt(prompts_dir: str | Path, name: str) -> ChatPromptTemplate:
    """system + human messages from prompts/<name>.yaml."""
    pf = load_prompt_file(str(prompts_dir), name)
    msgs = [(role, getattr(pf, role)) for role in ("system", "human") if getattr(pf, role)]
    if not msgs:
        raise ValueError(f"prompts/{name}.yaml has neither 'system' nor 'human'")
    return ChatPromptTemplate.from_messages(msgs)


def load_fragment(prompts_dir: str | Path, name: str, key: str) -> str:
    pf = load_prompt_file(str(prompts_dir), name)
    value = getattr(pf, key, None) if key in ("system", "human") else (pf.model_extra or {}).get(key)
    if not value:
        raise KeyError(f"prompts/{name}.yaml has no '{key}'")
    return value
