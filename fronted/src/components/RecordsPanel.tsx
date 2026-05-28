import { useMemo, useState } from "react"
import type { SessionRecord } from "../types/api"

interface RecordsPanelProps {
  sessions: SessionRecord[]
  currentSessionId: string
  busy?: boolean
  onNewSession: () => void
  onOpenSession: (sessionId: string) => void
  onDeleteSession: (sessionId: string, event: React.MouseEvent) => void
}

/**
 * 将会话更新时间格式化为相对或简短绝对时间。
 */
function formatSessionTime(input: string | undefined): string {
  if (!input) {
    return ""
  }
  const date = new Date(input)
  if (Number.isNaN(date.getTime())) {
    return input
  }
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffMin = Math.floor(diffMs / 60_000)

  if (diffMin < 1) {
    return "刚刚"
  }
  if (diffMin < 60) {
    return `${diffMin} 分钟前`
  }

  const isToday =
    date.getFullYear() === now.getFullYear() &&
    date.getMonth() === now.getMonth() &&
    date.getDate() === now.getDate()
  const yesterday = new Date(now)
  yesterday.setDate(yesterday.getDate() - 1)
  const isYesterday =
    date.getFullYear() === yesterday.getFullYear() &&
    date.getMonth() === yesterday.getMonth() &&
    date.getDate() === yesterday.getDate()

  const timePart = date.toLocaleString("zh-CN", { hour: "2-digit", minute: "2-digit", hour12: false })
  if (isToday) {
    return `今天 ${timePart}`
  }
  if (isYesterday) {
    return `昨天 ${timePart}`
  }
  if (date.getFullYear() === now.getFullYear()) {
    return date.toLocaleString("zh-CN", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", hour12: false })
  }
  return date.toLocaleString("zh-CN", {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  })
}

/**
 * 聊天记录列表页（会话卡片网格）。
 */
export function RecordsPanel({
  sessions,
  currentSessionId,
  busy = false,
  onNewSession,
  onOpenSession,
  onDeleteSession,
}: RecordsPanelProps) {
  const [keyword, setKeyword] = useState("")

  const filteredSessions = useMemo(() => {
    const q = keyword.trim().toLowerCase()
    if (!q) {
      return sessions
    }
    return sessions.filter((s) => (s.title || "新会话").toLowerCase().includes(q))
  }, [sessions, keyword])

  const totalTurns = useMemo(
    () => sessions.reduce((sum, s) => sum + (s.turn_count ?? 0), 0),
    [sessions],
  )

  return (
    <section className="panel records-page" aria-labelledby="records-page-title">
      <header className="records-page__header">
        <div className="records-page__intro">
          <p className="records-page__eyebrow">会话历史</p>
          <h1 id="records-page-title">聊天记录</h1>
          <p className="records-page__subtitle">管理多轮对话，点击卡片继续上次话题</p>
        </div>
        <button type="button" className="btn btn--primary records-page__new" onClick={onNewSession}>
          <span className="records-page__new-icon" aria-hidden="true">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <path d="M12 5v14M5 12h14" />
            </svg>
          </span>
          新会话
        </button>
      </header>

      <div className="records-page__stats" aria-live="polite">
        <div className="records-stat">
          <span className="records-stat__value">{sessions.length}</span>
          <span className="records-stat__label">会话</span>
        </div>
        <div className="records-stat">
          <span className="records-stat__value">{totalTurns}</span>
          <span className="records-stat__label">累计轮次</span>
        </div>
        {currentSessionId && (
          <div className="records-stat records-stat--active">
            <span className="records-stat__label">当前会话</span>
            <span className="records-stat__hint">已选中，可在聊天页继续</span>
          </div>
        )}
      </div>

      {sessions.length > 0 && (
        <div className="records-page__search">
          <span className="records-page__search-icon" aria-hidden="true">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="11" cy="11" r="7" />
              <path d="M20 20l-3-3" />
            </svg>
          </span>
          <input
            type="search"
            value={keyword}
            onChange={(e) => setKeyword(e.target.value)}
            placeholder="搜索会话标题…"
            aria-label="搜索聊天记录"
            autoComplete="off"
          />
          {keyword.trim() && (
            <button
              type="button"
              className="records-page__search-clear"
              onClick={() => setKeyword("")}
              aria-label="清除搜索"
            >
              ×
            </button>
          )}
        </div>
      )}

      {busy && sessions.length === 0 ? (
        <ul className="records-grid records-grid--loading" aria-busy="true" aria-label="加载中">
          {[1, 2, 3, 4].map((i) => (
            <li key={i}>
              <div className="record-card record-card--skeleton" />
            </li>
          ))}
        </ul>
      ) : filteredSessions.length === 0 ? (
        <div className="records-empty">
          <div className="records-empty__visual" aria-hidden="true">
            <span className="records-empty__glow" />
            <svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.25">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
              <path d="M8 10h8M8 14h5" strokeLinecap="round" />
            </svg>
          </div>
          <h2>{keyword.trim() ? "没有匹配的会话" : "还没有聊天记录"}</h2>
          <p>
            {keyword.trim()
              ? "试试其他关键词，或清空搜索查看全部会话。"
              : "开始第一次对话后，会话会自动出现在这里。"}
          </p>
          {!keyword.trim() && (
            <button type="button" className="btn btn--soft" onClick={onNewSession}>
              开始第一次对话
            </button>
          )}
        </div>
      ) : (
        <ul className="records-grid">
          {filteredSessions.map((s) => {
            const isActive = currentSessionId === s.session_id
            const turns = s.turn_count ?? 0
            return (
              <li key={s.session_id}>
                <button
                  type="button"
                  className={`record-card ${isActive ? "is-active" : ""}`}
                  onClick={() => onOpenSession(s.session_id)}
                >
                  <span className="record-card__accent" aria-hidden="true" />
                  <span className="record-card__icon" aria-hidden="true">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75">
                      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                    </svg>
                  </span>
                  <span className="record-card__body">
                    <span className="record-card__title">{s.title || "新会话"}</span>
                    <span className="record-card__meta">
                      <span className="record-card__badge">{turns} 轮</span>
                      <span className="record-card__time">{formatSessionTime(s.updated_at || s.created_at)}</span>
                    </span>
                  </span>
                  {isActive && <span className="record-card__pill">当前</span>}
                  <span
                    className="record-card__delete"
                    role="button"
                    tabIndex={0}
                    title="删除会话"
                    aria-label={`删除会话：${s.title || "新会话"}`}
                    onClick={(e) => {
                      e.stopPropagation()
                      onDeleteSession(s.session_id, e)
                    }}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault()
                        e.stopPropagation()
                        onDeleteSession(s.session_id, e as unknown as React.MouseEvent)
                      }
                    }}
                  >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  </span>
                </button>
              </li>
            )
          })}
        </ul>
      )}
    </section>
  )
}
