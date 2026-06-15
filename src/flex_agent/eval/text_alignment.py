"""LLM semantic alignment for human-vs-agent open coding item evaluation."""

from __future__ import annotations

import json
import re
import sys
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from flex_agent.eval.core import EvalMetrics, normalize_dimension
from flex_agent.eval.prompts import text_alignment_prompt


class SemanticMatch(BaseModel):
    agent_dimension: str
    matched_human_dimension: str | None = None
    thought: str = Field(default="", description="可选的简短判断依据。")
    action: str | None = Field(default=None, description="可选的简短匹配结果标记。")


class TextSemanticAlignment(BaseModel):
    text_id: str
    reasoning_trace: str = Field(default="", description="可选的整条文本简短判断摘要。")
    matches: list[SemanticMatch] = Field(default_factory=list)


class BatchSemanticAlignment(BaseModel):
    texts: list[TextSemanticAlignment] = Field(default_factory=list)


def _human_items_for_prompt(record: dict[str, Any], fallback_items: dict[str, int]) -> list[dict[str, Any]]:
    if isinstance(record.get("human_items"), list):
        return [
            {
                "dimension": normalize_dimension(str(item.get("dimension", ""))),
                "value": item.get("value"),
                "evidences": item.get("evidences", []),
            }
            for item in record["human_items"]
            if item.get("dimension")
        ]
    return [
        {
            "dimension": normalize_dimension(dim),
            "value": value,
            "evidences": record.get("human_spans", []),
        }
        for dim, value in fallback_items.items()
    ]


def _agent_items_for_prompt(agent_items_raw: list[dict]) -> list[dict[str, Any]]:
    dims: dict[str, dict[str, Any]] = {}
    for item in agent_items_raw:
        label_dims: list[str] = []
        normalized = str(item.get("normalized_label") or "").strip()
        if normalized:
            dim = normalize_dimension(re.split(r"[:：]", normalized, maxsplit=1)[0].strip())
            if dim:
                label_dims.append(dim)
        if not label_dims:
            labels = str(item.get("labels", ""))
            for raw in labels.replace("；", ";").split(";"):
                if ":" not in raw and "：" not in raw:
                    continue
                dim = normalize_dimension(re.split(r"[:：]", raw, maxsplit=1)[0].strip())
                if dim:
                    label_dims.append(dim)
        if not label_dims:
            dim = normalize_dimension(str(item.get("name") or ""))
            if dim:
                label_dims.append(dim)
        for dim in label_dims:
            if not dim:
                continue
            entry = dims.setdefault(dim, {"dimension": dim, "evidences": [], "reasons": []})
            evidence = item.get("evidence") or item.get("name")
            if evidence and evidence not in entry["evidences"]:
                entry["evidences"].append(evidence)
            reason = item.get("reason")
            if reason and reason not in entry["reasons"]:
                entry["reasons"].append(reason)
    return list(dims.values())


def _counts_to_metrics(nums_both: int, nums_llm_only: int, nums_human_only: int) -> EvalMetrics:
    n_agent = nums_both + nums_llm_only
    n_human = nums_both + nums_human_only
    n_union = nums_both + nums_llm_only + nums_human_only
    return EvalMetrics(
        consistency=nums_both / n_union if n_union else 0.0,
        precision=nums_both / n_agent if n_agent else 0.0,
        recall=nums_both / n_human if n_human else 0.0,
        n_human=n_human,
        n_agent=n_agent,
        n_intersection=nums_both,
        n_union=n_union,
    )


def _aggregate_semantic_metrics(
    entries: list[dict[str, Any]],
    all_alignments: dict[int, dict[str, str | None]],
) -> dict[str, Any]:
    per_text: list[dict[str, Any]] = []
    total_llm_only = 0
    total_human_only = 0
    total_both = 0
    for entry in entries:
        text_id = entry["text_id"]
        human_dims = {item["dimension"] for item in entry["human_items"]}
        agent_dims = {item["dimension"] for item in entry["agent_items"]}
        alignment = all_alignments.get(text_id, {})
        both = {agent_dim for agent_dim, human_dim in alignment.items() if human_dim}
        matched_human = {human_dim for human_dim in alignment.values() if human_dim}
        llm_only = agent_dims - both
        human_only = human_dims - matched_human

        total_llm_only += len(llm_only)
        total_human_only += len(human_only)
        total_both += len(both)
        metrics = _counts_to_metrics(len(both), len(llm_only), len(human_only))
        per_text.append({
            "text_id": text_id,
            "human_items": sorted(human_dims),
            "agent_items": sorted(agent_dims),
            "both": sorted(both),
            "llm_only": sorted(llm_only),
            "human_only": sorted(human_only),
            "nums_both": len(both),
            "nums_llm_only": len(llm_only),
            "nums_human_only": len(human_only),
            **metrics.as_dict(),
        })

    n_texts = len(per_text)
    macro = EvalMetrics(
        consistency=sum(row["consistency"] for row in per_text) / n_texts if n_texts else 0.0,
        precision=sum(row["precision"] for row in per_text) / n_texts if n_texts else 0.0,
        recall=sum(row["recall"] for row in per_text) / n_texts if n_texts else 0.0,
        n_human=total_both + total_human_only,
        n_agent=total_both + total_llm_only,
        n_intersection=total_both,
        n_union=total_both + total_llm_only + total_human_only,
    )
    micro = _counts_to_metrics(total_both, total_llm_only, total_human_only)
    return {
        "common_texts": len(entries),
        "nums_llm_only": total_llm_only,
        "nums_human_only": total_human_only,
        "nums_both": total_both,
        "macro": macro.as_dict(),
        "micro": micro.as_dict(),
        "per_text": per_text,
        "alignment": {
            text_id: all_alignments.get(text_id, {})
            for text_id in (e["text_id"] for e in entries)
        },
    }


def _human_from_react_action(action: str, human_dims: set[str]) -> str | None:
    action = action.strip()
    if not action.upper().startswith("MATCH"):
        return None
    candidate = normalize_dimension(action[5:].strip().strip("<>").strip())
    return candidate if candidate in human_dims else None


def build_semantic_alignment_for_texts(
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
            "human_items": entry["human_items"],
            "agent_items": entry["agent_items"],
        })
    try:
        prompt = ChatPromptTemplate.from_messages([("human", text_alignment_prompt())])
        chain = prompt | llm.with_structured_output(BatchSemanticAlignment, method="json_mode")
        result: BatchSemanticAlignment = chain.invoke({"texts_json": json.dumps(prompt_rows, ensure_ascii=False)})
    except Exception as exc:
        print(f"  [warn] semantic evidence alignment LLM call failed: {exc!r}", file=sys.stderr)
        return {}

    expected = {str(entry["text_id"]): entry for entry in text_batch}
    validated: dict[int, dict[str, str | None]] = {}
    for text in result.texts:
        if text.text_id not in expected:
            continue
        entry = expected[text.text_id]
        agent_dims = {item["dimension"] for item in entry["agent_items"]}
        human_dims = {item["dimension"] for item in entry["human_items"]}
        matches: dict[str, str | None] = {}
        for match in text.matches:
            agent_dim = normalize_dimension(match.agent_dimension)
            human_dim = (
                normalize_dimension(match.matched_human_dimension or "")
                if match.matched_human_dimension
                else None
            )
            if not human_dim and match.action:
                human_dim = _human_from_react_action(match.action, human_dims)
            if agent_dim not in agent_dims:
                continue
            if human_dim not in human_dims:
                human_dim = None
            matches[agent_dim] = human_dim
        validated[int(text.text_id)] = matches
    return validated
