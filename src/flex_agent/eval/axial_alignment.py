"""LLM semantic alignment for axial coding category evaluation."""

from __future__ import annotations

import json
import sys
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from flex_agent.eval.axial_core import enforce_one_to_one_alignment, normalize_category
from flex_agent.eval.prompts import axial_category_alignment_prompt


class AxialSemanticMatch(BaseModel):
    agent_dimension: str
    matched_human_category: str | None = None
    thought: str = Field(default="", description="Optional brief rationale.")


class AxialTextSemanticAlignment(BaseModel):
    text_id: str
    matches: list[AxialSemanticMatch] = Field(default_factory=list)


class BatchAxialSemanticAlignment(BaseModel):
    texts: list[AxialTextSemanticAlignment] = Field(default_factory=list)


def build_axial_semantic_alignment_for_texts(
    text_batch: list[dict[str, Any]],
    llm: BaseChatModel,
) -> dict[int, dict[str, str | None]]:
    if not text_batch:
        return {}

    prompt_rows = []
    for entry in text_batch:
        prompt_rows.append({
            "text_id": str(entry["text_id"]),
            "content": entry["content"],
            "human_categories": sorted(entry["human_categories"]),
            "agent_dimensions": entry["agent_dimensions"],
        })

    try:
        prompt = ChatPromptTemplate.from_messages([("human", axial_category_alignment_prompt())])
        chain = prompt | llm.with_structured_output(BatchAxialSemanticAlignment, method="json_mode")
        result: BatchAxialSemanticAlignment = chain.invoke(
            {"texts_json": json.dumps(prompt_rows, ensure_ascii=False)}
        )
    except Exception as exc:
        print(f"  [warn] axial semantic alignment LLM call failed: {exc!r}", file=sys.stderr)
        return {}

    expected = {str(entry["text_id"]): entry for entry in text_batch}
    validated: dict[int, dict[str, str | None]] = {}
    for text in result.texts:
        if text.text_id not in expected:
            continue
        entry = expected[text.text_id]
        agent_dims = {dim["name"] for dim in entry["agent_dimensions"]}
        human_categories = {normalize_category(c) for c in entry["human_categories"]}
        matches: dict[str, str | None] = {}
        for match in text.matches:
            agent_dim = match.agent_dimension.strip()
            human_cat = (
                normalize_category(match.matched_human_category or "")
                if match.matched_human_category
                else None
            )
            if agent_dim not in agent_dims:
                continue
            if human_cat not in human_categories:
                human_cat = None
            matches[agent_dim] = human_cat
        matches = enforce_one_to_one_alignment(
            matches,
            agent_dims=agent_dims,
            human_categories=human_categories,
        )
        validated[int(text.text_id)] = matches
    return validated
