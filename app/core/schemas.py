"""Pydantic schemas shared across API + workers.

Kept in `core` (not `api`) because the Kafka worker also needs to deserialize
the same task payloads — this avoids cross-imports between api/ and workers/.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    PARSING = "parsing"
    INDEXING = "indexing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


# ---------- Upload ----------

class UploadResponse(BaseModel):
    task_id: str
    object_name: str
    status: TaskStatus = TaskStatus.PENDING
    message: str = "Uploaded; queued for parsing."


class MultipartUploadInitRequest(BaseModel):
    filename: str
    total_size: int = Field(..., gt=0)
    total_chunks: int = Field(..., gt=0)
    file_md5: Optional[str] = None


class MultipartUploadInitResponse(BaseModel):
    upload_id: str
    task_id: str
    object_name: str
    uploaded_chunks: List[int] = []
    expires_in_seconds: int


class MultipartUploadStatusResponse(BaseModel):
    upload_id: str
    task_id: str
    object_name: str
    filename: str
    total_size: int
    total_chunks: int
    uploaded_chunks: List[int]
    uploaded_count: int
    complete: bool


class MultipartChunkResponse(BaseModel):
    upload_id: str
    chunk_index: int
    uploaded: bool = True
    uploaded_count: int
    total_chunks: int


class MultipartCompleteRequest(BaseModel):
    file_md5: Optional[str] = None


class DeleteTasksRequest(BaseModel):
    task_ids: List[str] = Field(..., min_length=1)


class DeleteTaskResult(BaseModel):
    task_id: str
    deleted: bool
    message: str = ""


class DeleteTasksResponse(BaseModel):
    results: List[DeleteTaskResult]


# ---------- Kafka task payload ----------

class ParseTask(BaseModel):
    """Message published to Kafka after a successful upload."""
    task_id: str
    object_name: str           # MinIO object key
    bucket: str
    original_filename: str
    submitted_at: datetime = Field(default_factory=datetime.utcnow)


# ---------- Parsed document (Docling output) ----------

class ParsedDocument(BaseModel):
    """In-memory representation of a Docling-parsed paper."""
    task_id: str
    markdown: str
    original_filename: str
    title: Optional[str] = None
    num_pages: Optional[int] = None
    num_tables: int = 0
    num_chars: int = 0


# ---------- Task status record (persisted) ----------

class TaskRecord(BaseModel):
    task_id: str
    object_name: str
    original_filename: str
    status: TaskStatus
    message: str = ""
    error: Optional[str] = None
    # Optional metadata populated as the task progresses through phases.
    num_pages: Optional[int] = None
    num_tables: Optional[int] = None
    num_parents: Optional[int] = None
    num_children: Optional[int] = None
    parsed_object_name: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------- Query / retrieval ----------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: Optional[int] = None
    # Optional: scope retrieval to a single paper. If absent, search across all.
    task_id: Optional[str] = None


class RetrievedChunk(BaseModel):
    parent_id: str
    parent_text: str
    child_ids: List[str] = []
    score: float
    metadata: dict = {}


class QueryResponse(BaseModel):
    query: str
    contexts: List[RetrievedChunk]


# ---------- LCEL answer ----------

class AnswerRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: Optional[int] = None
    task_id: Optional[str] = None
    use_agent: bool = False


class AnswerResponse(BaseModel):
    query: str
    answer: str
    contexts: List[RetrievedChunk]


# ---------- Paper-level index / search ----------

class PaperProfile(BaseModel):
    """论文级画像，用于 paper_search / paper_deep_search 过滤和排序。"""

    paper_id: str
    title: str = ""
    abstract: str = ""
    clean_abstract: str = ""
    abstract_summary: str = ""
    introduction_summary: str = ""
    method_summary: str = ""
    contribution_summary: str = ""
    experiment_summary: str = ""
    research_problem: str = ""
    method_name: str = ""
    authors: List[str] = Field(default_factory=list)
    year: str = ""
    source_file: str = ""
    main_task: str = ""
    modality_tags: List[str] = Field(default_factory=list)
    task_tags: List[str] = Field(default_factory=list)
    method_tags: List[str] = Field(default_factory=list)
    domain_tags: List[str] = Field(default_factory=list)
    dataset_tags: List[str] = Field(default_factory=list)
    metric_tags: List[str] = Field(default_factory=list)
    is_image_related: bool = False
    is_frequency_related: bool = False
    paper_search_text: str = ""
    matched_keywords: List[str] = Field(default_factory=list)
    image_confidence: float = 0.0
    frequency_confidence: float = 0.0
    image_evidence: List[dict] = Field(default_factory=list)
    frequency_evidence: List[dict] = Field(default_factory=list)
    summary: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class PaperSearchResult(BaseModel):
    """论文级检索返回项。"""

    paper_id: str
    title: str = ""
    main_task: str = ""
    modality_tags: List[str] = Field(default_factory=list)
    task_tags: List[str] = Field(default_factory=list)
    method_tags: List[str] = Field(default_factory=list)
    domain_tags: List[str] = Field(default_factory=list)
    dataset_tags: List[str] = Field(default_factory=list)
    metric_tags: List[str] = Field(default_factory=list)
    image_confidence: float = 0.0
    frequency_confidence: float = 0.0
    summary: str = ""
    abstract_summary: str = ""
    method_summary: str = ""
    contribution_summary: str = ""
    reason: str = ""
    source_file: str = ""
    score: float = 0.0
    evidence_chunks: List[str] = Field(default_factory=list)


# --- Agent Routing ---

class RouterDecision(BaseModel):
    """Structured output from the LLM router node."""

    route: Literal[
        "paper_search", "paper_deep_search", "chunk_qa", "task_status", "paper_profile"
    ]
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0–1")
    reason: str = Field(default="", description="Human-readable routing rationale")


class RoutingResult(BaseModel):
    """Final routing decision returned to the API layer."""

    route: Literal[
        "paper_search", "paper_deep_search", "chunk_qa", "task_status", "paper_profile"
    ]
    confidence: float = 0.0
    reason: str = ""
    source: Literal["rule", "llm", "fallback"] = "rule"
