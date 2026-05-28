import { requestJson } from "./http"
import SparkMD5 from "spark-md5"
import type {
  AgentChatRequest,
  AgentChatResponse,
  DeleteTaskResult,
  DeleteTasksResponse,
  HealthResponse,
  MultipartUploadInitResponse,
  MultipartUploadStatusResponse,
  SessionHistoryResponse,
  SessionRecord,
  TaskRecord,
  UploadResponse,
} from "../types/api"

// ── Session API ───────────────────────────────────────────────────

export function createSession(): Promise<SessionRecord> {
  return requestJson<SessionRecord>("/api/v1/sessions", { method: "POST" })
}

export function listSessions(): Promise<{ sessions: SessionRecord[] }> {
  return requestJson<{ sessions: SessionRecord[] }>("/api/v1/sessions", { method: "GET" })
}

export function deleteSession(sessionId: string): Promise<unknown> {
  return requestJson(`/api/v1/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" })
}

/**
 * 拉取会话完整对话历史 `GET /api/v1/sessions/{id}/history`
 */
export function fetchSessionHistory(
  sessionId: string,
  limit = 100,
): Promise<SessionHistoryResponse> {
  const params = new URLSearchParams({ limit: String(limit) })
  return requestJson<SessionHistoryResponse>(
    `/api/v1/sessions/${encodeURIComponent(sessionId)}/history?${params}`,
    { method: "GET" },
  )
}

// ── Upload / Papers ──────────────────────────────────────────────

const UPLOAD_CHUNK_SIZE = 5 * 1024 * 1024
const UPLOAD_CONCURRENCY = 3

/**
 * 健康检查 `GET /health`
 */
export function fetchHealth(): Promise<HealthResponse> {
  return requestJson<HealthResponse>("/health", { method: "GET" })
}

/**
 * 计算 Blob 的 MD5，用于服务端分片校验。
 */
async function md5Blob(blob: Blob): Promise<string> {
  const buffer = await blob.arrayBuffer()
  return SparkMD5.ArrayBuffer.hash(buffer)
}

/**
 * 初始化分片上传。
 */
function initMultipartUpload(file: File, fileMd5: string, totalChunks: number): Promise<MultipartUploadInitResponse> {
  return requestJson<MultipartUploadInitResponse>("/api/v1/papers/upload/multipart/init", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      filename: file.name,
      total_size: file.size,
      total_chunks: totalChunks,
      file_md5: fileMd5,
    }),
  })
}

/**
 * 查询 Redis Bitmap 记录的分片状态。
 */
function getMultipartUploadStatus(uploadId: string): Promise<MultipartUploadStatusResponse> {
  return requestJson<MultipartUploadStatusResponse>(`/api/v1/papers/upload/multipart/${encodeURIComponent(uploadId)}`, {
    method: "GET",
  })
}

/**
 * 上传单个分片及其 MD5。
 */
async function uploadMultipartChunk(uploadId: string, index: number, chunk: Blob): Promise<void> {
  const form = new FormData()
  form.append("chunk_index", String(index))
  form.append("chunk_md5", await md5Blob(chunk))
  form.append("chunk", chunk, `${index}.part`)
  await requestJson(`/api/v1/papers/upload/multipart/${encodeURIComponent(uploadId)}/chunks`, {
    method: "POST",
    body: form,
  })
}

/**
 * 通知后端校验 bitmap、合并 MinIO 分片并投递解析任务。
 */
function completeMultipartUpload(uploadId: string, fileMd5: string): Promise<UploadResponse> {
  return requestJson<UploadResponse>(`/api/v1/papers/upload/multipart/${encodeURIComponent(uploadId)}/complete`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ file_md5: fileMd5 }),
  })
}

/**
 * 上传 PDF 并创建解析任务。使用 Redis Bitmap + MinIO 分片上传以提高可靠性。
 */
export async function uploadPaper(
  file: File,
  onProgress?: (uploaded: number, total: number) => void,
): Promise<UploadResponse> {
  const totalChunks = Math.ceil(file.size / UPLOAD_CHUNK_SIZE)
  const fileMd5 = await md5Blob(file)
  const resumeKey = `papermind-upload:${file.name}:${file.size}:${fileMd5}`
  const cachedUploadId = localStorage.getItem(resumeKey)

  let uploadId = ""
  let uploaded = new Set<number>()
  if (cachedUploadId) {
    try {
      const status = await getMultipartUploadStatus(cachedUploadId)
      uploadId = status.upload_id
      uploaded = new Set(status.uploaded_chunks)
    } catch {
      localStorage.removeItem(resumeKey)
    }
  }

  if (!uploadId) {
    const init = await initMultipartUpload(file, fileMd5, totalChunks)
    uploadId = init.upload_id
    uploaded = new Set(init.uploaded_chunks)
    localStorage.setItem(resumeKey, uploadId)
  }

  onProgress?.(uploaded.size, totalChunks)
  const pending = Array.from({ length: totalChunks }, (_, index) => index).filter(
    (index) => !uploaded.has(index),
  )
  let cursor = 0

  /**
   * 并发上传 worker：每次取一个未完成分片。
   */
  async function uploadWorker() {
    while (cursor < pending.length) {
      const index = pending[cursor]
      cursor += 1
      const start = index * UPLOAD_CHUNK_SIZE
      const chunk = file.slice(start, Math.min(file.size, start + UPLOAD_CHUNK_SIZE))
      await uploadMultipartChunk(uploadId, index, chunk)
      uploaded.add(index)
      onProgress?.(uploaded.size, totalChunks)
    }
  }

  await Promise.all(
    Array.from({ length: Math.min(UPLOAD_CONCURRENCY, pending.length) }, () => uploadWorker()),
  )
  const response = await completeMultipartUpload(uploadId, fileMd5)
  localStorage.removeItem(resumeKey)
  return response
}

/**
 * 查询单任务 `GET /api/v1/papers/{taskId}`
 */
export function fetchTask(taskId: string): Promise<TaskRecord> {
  return requestJson<TaskRecord>(`/api/v1/papers/${encodeURIComponent(taskId)}`, {
    method: "GET",
  })
}

/**
 * 删除单个任务及其 MinIO / ES 数据。
 */
export function deletePaper(taskId: string): Promise<DeleteTaskResult> {
  return requestJson<DeleteTaskResult>(`/api/v1/papers/${encodeURIComponent(taskId)}`, {
    method: "DELETE",
  })
}

/**
 * 批量删除任务及其 MinIO / ES 数据。
 */
export function deletePapers(taskIds: string[]): Promise<DeleteTasksResponse> {
  return requestJson<DeleteTasksResponse>("/api/v1/papers/batch", {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ task_ids: taskIds }),
  })
}

/**
 * 最近任务列表 `GET /api/v1/papers?limit=`
 */
export function listTasks(limit = 50): Promise<TaskRecord[]> {
  const q = new URLSearchParams({ limit: String(limit) })
  return requestJson<TaskRecord[]>(`/api/v1/papers?${q.toString()}`, { method: "GET" })
}

/**
 * Agent 对话 `POST /api/v1/agent/chat`
 */
export function chatWithAgent(body: AgentChatRequest): Promise<AgentChatResponse> {
  return requestJson<AgentChatResponse>("/api/v1/agent/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query: body.query,
      top_k: body.top_k,
      task_id: body.task_id || undefined,
      session_id: body.session_id || undefined,
    }),
  })
}

/**
 * Agent 流式对话 `POST /api/v1/agent/chat/stream` → SSE
 * 回调会依次收到 {type, ...} 事件对象。
 */
export async function chatWithAgentStream(
  body: AgentChatRequest,
  onEvent: (event: Record<string, unknown>) => void,
): Promise<void> {
  const base = (import.meta.env.VITE_API_BASE as string | undefined) ?? ""
  const url = `${base}/api/v1/agent/chat/stream`

  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query: body.query,
      top_k: body.top_k,
      task_id: body.task_id || undefined,
      session_id: body.session_id || undefined,
    }),
  })

  if (!res.ok) {
    let message = res.statusText
    try {
      const err = (await res.json()) as { detail?: string }
      if (err.detail) message = err.detail
    } catch { /* ignore */ }
    throw new Error(message)
  }

  const reader = res.body?.getReader()
  if (!reader) throw new Error("Stream not supported")

  const decoder = new TextDecoder()
  let buffer = ""

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    // Parse SSE frames
    const lines = buffer.split("\n")
    buffer = lines.pop() ?? ""

    let eventType = ""
    for (const line of lines) {
      if (line.startsWith("event: ")) {
        eventType = line.slice(7).trim()
      } else if (line.startsWith("data: ")) {
        try {
          const data = JSON.parse(line.slice(6)) as Record<string, unknown>
          data._event = eventType
          onEvent(data)
        } catch { /* skip malformed */ }
      }
    }
  }
}

