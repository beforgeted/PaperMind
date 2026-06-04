/**
 * 解析与后端通信使用的 API 根路径（不含尾斜杠）。
 * 开发环境建议留空，由 `vite.config` 将 `/api`、`/health` 代理到 FastAPI。
 */
export function getApiBase(): string {
  const raw = import.meta.env.VITE_API_BASE
  if (raw == null || String(raw).trim() === "") {
    return ""
  }
  return String(raw).replace(/\/+$/, "")
}
