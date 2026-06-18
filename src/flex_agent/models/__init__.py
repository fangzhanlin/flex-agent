from __future__ import annotations

from typing import Any, List

from pydantic import BaseModel, Field


def merge_quality_warnings(current: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    if not current:
        return dict(update)
    if not update:
        return dict(current)

    merged = dict(current)
    for key, value in update.items():
        if key == "notes":
            merged_notes = list(merged.get("notes", []))
            merged_notes.extend(list(value))
            merged["notes"] = merged_notes
            continue
        if isinstance(value, int):
            merged[key] = int(merged.get(key, 0)) + value
            continue
        merged[key] = value
    return merged


class TextItem(BaseModel):
    id: int
    content: str


class FinishedItemDetail(BaseModel):
    name: str
    labels: str | None = None
    evidence: str | None = None
    normalized_label: str | None = None
    reason: str | None = None


class FinishedTextItem(BaseModel):
    id: int
    content: str
    content_with_labels: str
    items: List[FinishedItemDetail] = Field(default_factory=list)


class DimensionDetail(BaseModel):
    name: str
    items: List[str] = Field(default_factory=list)
    definition: str | None = None


class RunMeta(BaseModel):
    data_path: str
    max_nums: int = 10
    codebook_nums: int = 5
    kevin_batch_size: int = 5
    open_mode: str = "pure"
    sample_mode: str = "sequential"
    random_seed: int | None = None
    concurrency_limit: int = 50
    prompts_dir: str | None = None
    workspace_dir: str | None = None


class SessionMeta(BaseModel):
    prompts_dir: str
    workspace_dir: str
    prompts_resolved: str
    workspace_resolved: str


class PartitionMeta(BaseModel):
    codebook_text_ids: List[int] = Field(default_factory=list)
    kevin_text_ids: List[int] = Field(default_factory=list)


class WorkspaceSnapshot(BaseModel):
    unfinished_texts: List[TextItem] = Field(default_factory=list)
    finished_texts: List[FinishedTextItem] = Field(default_factory=list)
    dimensions: List[DimensionDetail] = Field(default_factory=list)
    quality_warnings: dict[str, Any] = Field(default_factory=dict)
