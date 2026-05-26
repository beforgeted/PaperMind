import { requestJson } from "./http"
import { getApiBase } from "../config"
import SparkMD5 from "spark-md5"
import type {
  AnswerRequestBody,
  AnswerResponseBody,
  AnswerStreamEvent,
  DeleteTaskResult,
  DeleteTasksResponse,
  HealthResponse,
  MultipartUploadInitResponse,
  MultipartUploadStatusResponse,
  QueryRequestBody,
  QueryResponseBody,
  TaskRecord,
  UploadResponse,
} from "../types/api"

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
 * 混合检索 `POST /api/v1/papers/query`
 */
export function queryPapers(body: QueryRequestBody): Promise<QueryResponseBody> {
  return requestJson<QueryResponseBody>("/api/v1/papers/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query: body.query,
      top_k: body.top_k,
      task_id: body.task_id || undefined,
    }),
  })
}

/**
 * 检索 + 大模型回答 `POST /api/v1/papers/answer`
 */
export function answerQuestion(body: AnswerRequestBody): Promise<AnswerResponseBody> {
  return requestJson<AnswerResponseBody>("/api/v1/papers/answer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query: body.query,
      top_k: body.top_k,
      task_id: body.task_id || undefined,
    }),
  })
}

/**
 * 流式检索 + 大模型回答 `POST /api/v1/papers/answer/stream`
 */
export async function streamAnswerQuestion(
  body: AnswerRequestBody,
  onEvent: (event: AnswerStreamEvent) => void,
): Promise<void> {
  const base = getApiBase()
  const res = await fetch(`${base}/api/v1/papers/answer/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query: body.query,
      top_k: body.top_k,
      task_id: body.task_id || undefined,
    }),
  })
  if (!res.ok) {
    throw new Error(res.statusText || `HTTP ${res.status}`)
  }
  if (!res.body) {
    throw new Error("浏览器不支持流式响应。")
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""

  while (true) {
    const { done, value } = await reader.read()
    if (done) {
      break
    }
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split("\n")
    buffer = lines.pop() ?? ""
    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed) {
        continue
      }
      const event = JSON.parse(trimmed) as AnswerStreamEvent
      onEvent(event)
      if (event.type === "error") {
        throw new Error(event.message)
      }
    }
  }

  buffer += decoder.decode()
  const trailing = buffer.trim()
  if (trailing) {
    const event = JSON.parse(trailing) as AnswerStreamEvent
    onEvent(event)
    if (event.type === "error") {
      throw new Error(event.message)
    }
  }
}
