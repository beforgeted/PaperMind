import type { AgentChatResponse } from "../types/api"

export type CitationChunk = AgentChatResponse["contexts"][number]

/**
 * 将 API / 流式返回的任意 context 条目规范为前端可安全渲染的结构。
 */
export function normalizeCitationChunk(raw: unknown, index: number): CitationChunk {
  const item = raw && typeof raw === "object" ? (raw as Record<string, unknown>) : {}
  const metadataRaw = item.metadata
  const metadata =
    metadataRaw && typeof metadataRaw === "object"
      ? { ...(metadataRaw as Record<string, unknown>) }
      : ({} as Record<string, unknown>)

  if (item.title != null && metadata.title == null) {
    metadata.title = item.title
  }
  if (item.section != null && metadata.section_title == null) {
    metadata.section_title = item.section
  }
  if (item.section_type != null && metadata.section_type == null) {
    metadata.section_type = item.section_type
  }
  if (item.paper_id != null && metadata.paper_id == null) {
    metadata.paper_id = item.paper_id
  }

  const parentText =
    (typeof item.parent_text === "string" ? item.parent_text : "") ||
    (typeof item.content === "string" ? item.content : "")

  const childIds = Array.isArray(item.child_ids)
    ? item.child_ids.map((id) => String(id))
    : Array.isArray(item.chunk_ids)
      ? item.chunk_ids.map((id) => String(id))
      : []

  const scoreRaw = item.score
  const score =
    typeof scoreRaw === "number" && Number.isFinite(scoreRaw)
      ? scoreRaw
      : typeof scoreRaw === "string"
        ? Number.parseFloat(scoreRaw) || 0
        : 0

  const parentId = String(item.parent_id ?? item.paper_id ?? metadata.paper_id ?? `source-${index}`)

  return {
    parent_id: parentId,
    parent_text: parentText,
    child_ids: childIds,
    score,
    metadata,
  }
}

/**
 * 批量规范化引用列表，过滤无效条目。
 */
export function normalizeCitationList(raw: unknown): CitationChunk[] {
  if (!Array.isArray(raw)) {
    return []
  }
  return raw.map((item, index) => normalizeCitationChunk(item, index))
}

/**
 * 从检索片段元数据中提取展示用论文/文件名。
 */
export function sourceLabel(ctx: CitationChunk, index: number): string {
  const metadata = ctx.metadata ?? {}
  const name =
    metadata.title ??
    metadata.source_file ??
    metadata.original_filename ??
    metadata.filename ??
    (metadata.source && metadata.source !== "papermind_agent" ? metadata.source : undefined) ??
    metadata.object_name ??
    ctx.parent_id
  return typeof name === "string" && name.trim() ? name : `来源 ${index + 1}`
}

/**
 * 提取章节等次要信息，用于侧栏列表副标题。
 */
export function sourceSubtitle(ctx: CitationChunk): string | null {
  const metadata = ctx.metadata ?? {}
  const section =
    (typeof metadata.section_title === "string" && metadata.section_title) ||
    (typeof metadata.section === "string" && metadata.section) ||
    null
  const sectionType =
    typeof metadata.section_type === "string" ? metadata.section_type : null
  if (section && sectionType) {
    return `${section} · ${sectionType}`
  }
  return section ?? sectionType
}

/**
 * 格式化相关度分数展示。
 */
export function formatCitationScore(score: number | undefined): string {
  const value = typeof score === "number" && Number.isFinite(score) ? score : 0
  return value.toFixed(4)
}
