import { getApiBase } from "../config"

/**
 * 从 FastAPI 错误响应中解析 `detail` 供人类阅读
 */
function parseErrorDetail(detail: unknown): string {
  if (typeof detail === "string") {
    return detail
  }
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (item && typeof item === "object" && "msg" in item) {
          return String((item as { msg: unknown }).msg)
        }
        return JSON.stringify(item)
      })
      .filter(Boolean)
      .join("; ")
  }
  if (detail && typeof detail === "object") {
    return JSON.stringify(detail)
  }
  return "请求失败"
}

/**
 * 以 JSON 方式请求后端；自动拼接 `VITE_API_BASE`。
 *
 * @param path - 以 `/` 开头的路径，如 `/api/v1/papers`
 * @param init - 传给 `fetch` 的选项
 * @template T 响应体 JSON 结构
 * @returns 已解析的 JSON
 */
export async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const base = getApiBase()
  const url = `${base}${path.startsWith("/") ? path : `/${path}`}`
  const res = await fetch(url, init)
  if (!res.ok) {
    let message = res.statusText || `HTTP ${res.status}`
    try {
      const body: unknown = await res.json()
      if (body && typeof body === "object" && "detail" in body) {
        message = parseErrorDetail((body as { detail: unknown }).detail)
      }
    } catch {
      // 保留 statusText
    }
    throw new Error(message)
  }
  return (await res.json()) as T
}
