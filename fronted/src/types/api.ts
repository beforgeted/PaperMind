/**
 * 与 `app.core.schemas` 对齐的 API 类型（供前端解包 JSON 用）
 */
export type TaskStatus =
  | "pending"
  | "parsing"
  | "indexing"
  | "succeeded"
  | "failed"

export interface TaskRecord {
  task_id: string
  object_name: string
  original_filename: string
  status: TaskStatus
  message: string
  error?: string | null
  num_pages?: number | null
  num_tables?: number | null
  num_parents?: number | null
  num_children?: number | null
  parsed_object_name?: string | null
  created_at: string
  updated_at: string
}

export interface UploadResponse {
  task_id: string
  object_name: string
  status: TaskStatus
  message: string
}

export interface MultipartUploadInitResponse {
  upload_id: string
  task_id: string
  object_name: string
  uploaded_chunks: number[]
  expires_in_seconds: number
}

export interface MultipartUploadStatusResponse {
  upload_id: string
  task_id: string
  object_name: string
  filename: string
  total_size: number
  total_chunks: number
  uploaded_chunks: number[]
  uploaded_count: number
  complete: boolean
}

export interface DeleteTaskResult {
  task_id: string
  deleted: boolean
  message: string
}

export interface DeleteTasksResponse {
  results: DeleteTaskResult[]
}

export interface RetrievedChunk {
  parent_id: string
  parent_text: string
  child_ids: string[]
  score: number
  metadata: Record<string, unknown>
}

export interface QueryRequestBody {
  query: string
  top_k?: number
  task_id?: string
}

export interface QueryResponseBody {
  query: string
  contexts: RetrievedChunk[]
}

export interface AnswerRequestBody {
  query: string
  top_k?: number
  task_id?: string
}

export interface AnswerResponseBody {
  query: string
  answer: string
  contexts: RetrievedChunk[]
}

export type AnswerStreamEvent =
  | {
      type: "metadata"
      query: string
      contexts: RetrievedChunk[]
    }
  | {
      type: "delta"
      text: string
    }
  | {
      type: "done"
    }
  | {
      type: "error"
      message: string
    }

export interface HealthResponse {
  status: string
  app: string
  env: string
}
