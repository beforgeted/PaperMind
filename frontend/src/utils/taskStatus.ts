import type { TaskRecord } from "../types/api"

export type TaskStatusFilter = "all" | "ready" | "processing" | "failed"

/**
 * 根据任务状态返回用于样式类名的后缀。
 */
export function statusBadgeClass(status: TaskRecord["status"]): string {
  switch (status) {
    case "pending":
      return "badge--pending"
    case "parsing":
    case "indexing":
      return "badge--parsing"
    case "succeeded":
      return "badge--succeeded"
    case "failed":
      return "badge--failed"
    default:
      return ""
  }
}

/**
 * 将任务状态翻译为中文。
 */
export function statusLabel(status: TaskRecord["status"]): string {
  const map: Record<TaskRecord["status"], string> = {
    pending: "排队",
    parsing: "解析中",
    indexing: "索引中",
    succeeded: "可检索",
    failed: "失败",
  }
  return map[status] ?? status
}

/**
 * 判断任务是否处于处理中状态。
 */
export function isTaskProcessing(status: TaskRecord["status"]): boolean {
  return status === "pending" || status === "parsing" || status === "indexing"
}

/**
 * 按关键词与状态筛选任务列表。
 */
export function filterTasks(
  tasks: TaskRecord[],
  keyword: string,
  statusFilter: TaskStatusFilter,
): TaskRecord[] {
  const q = keyword.trim().toLowerCase()
  return tasks.filter((task) => {
    if (q && !task.original_filename.toLowerCase().includes(q)) {
      return false
    }
    if (statusFilter === "ready" && task.status !== "succeeded") {
      return false
    }
    if (statusFilter === "processing" && !isTaskProcessing(task.status)) {
      return false
    }
    if (statusFilter === "failed" && task.status !== "failed") {
      return false
    }
    return true
  })
}

/**
 * 将时间转成易读格式。
 */
export function formatTaskTime(input: string): string {
  const date = new Date(input)
  if (Number.isNaN(date.getTime())) {
    return input
  }
  return date.toLocaleString("zh-CN", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  })
}
