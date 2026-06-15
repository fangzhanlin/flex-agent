from __future__ import annotations

import sys
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from flex_agent.eval.prompts import dimension_name_alignment_prompt


def apply_semantic_alignment(
    agent_items: dict[int, dict[str, int]],
    alignment: dict[str, str | None],
) -> dict[int, dict[str, int]]:
    """Rewrite agent dimension names according to semantic alignment mapping."""
    if not alignment:
        return agent_items
    result: dict[int, dict[str, int]] = {}
    for text_id, items in agent_items.items():
        mapped: dict[str, int] = {}
        for dim, pol in items.items():
            if dim in alignment:
                target = alignment[dim]
                if target is None:
                    continue
                if target not in mapped:
                    mapped[target] = pol
            elif dim not in mapped:
                mapped[dim] = pol
        if mapped:
            result[text_id] = mapped
    return result


def build_dimension_name_alignment(
    agent_dimensions: list[str],
    human_dimensions: list[str],
    llm: BaseChatModel,
) -> dict[str, str | None]:
    """Use LLM to map agent dimension names to human dimension names."""
    if not agent_dimensions:
        return {}

    agent_list = "\n".join(f"- {d}" for d in agent_dimensions)
    human_list = "\n".join(f"- {d}" for d in human_dimensions)

    prompt = dimension_name_alignment_prompt(human_list=human_list, agent_list=agent_list)

    class AlignmentResult(BaseModel):
        mapping: dict[str, str | None] = Field(
            description="Mapping from agent dimension to human dimension (or null if no match)."
        )

    chat_prompt = ChatPromptTemplate.from_messages([("human", prompt)])
    chain = chat_prompt | llm.with_structured_output(AlignmentResult, method="json_mode")

    try:
        result = chain.invoke({})
        validated: dict[str, str | None] = {}
        for agent_dim, human_dim in result.mapping.items():
            if agent_dim not in agent_dimensions:
                continue
            if human_dim is None:
                validated[agent_dim] = None
            elif human_dim in human_dimensions:
                validated[agent_dim] = human_dim
            else:
                validated[agent_dim] = None
        return validated
    except Exception as exc:
        print(
            f"  [warn] semantic alignment LLM call failed: {exc!r}, falling back to exact match",
            file=sys.stderr,
        )
        return {}
