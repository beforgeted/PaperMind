import type { AgentChatResponse } from "../types/api"
import { formatCitationScore, normalizeCitationList, sourceLabel, sourceSubtitle } from "../utils/citations"

export interface SourcesPanelMessage {
  id: string
  question: string
  contexts: AgentChatResponse["contexts"]
  sources: AgentChatResponse["sources"]
  used_tools: string[]
}

interface SourcesPanelProps {
  message: SourcesPanelMessage | null
  selectedIndex: number
  onSelectIndex: (index: number) => void
  onClose: () => void
}

/**
 * 右侧引用来源面板（参考 ChatGPT「来源」侧栏）。
 */
export function SourcesPanel({
  message,
  selectedIndex,
  onSelectIndex,
  onClose,
}: SourcesPanelProps) {
  if (!message) {
    return null
  }

  const contexts = normalizeCitationList(message.contexts)
  const count = contexts.length
  const usedTools = Array.isArray(message.used_tools) ? message.used_tools : []
  const safeIndex = count > 0 ? Math.min(Math.max(selectedIndex, 0), count - 1) : 0
  const active = count > 0 ? contexts[safeIndex] : undefined

  return (
    <aside id="sources-panel" className="sources-panel" aria-label="引用来源">
      <header className="sources-panel__header">
        <div className="sources-panel__title">
          <span className="sources-panel__title-icon" aria-hidden="true">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
              <polyline points="14 2 14 8 20 8" />
              <line x1="16" y1="13" x2="8" y2="13" />
              <line x1="16" y1="17" x2="8" y2="17" />
            </svg>
          </span>
          <div>
            <h2>来源</h2>
            <p className="sources-panel__subtitle">
              {count > 0 ? `${count} 条引用片段` : "暂无引用"}
            </p>
          </div>
        </div>
        <button type="button" className="sources-panel__close" onClick={onClose} aria-label="关闭来源面板">
          ✕
        </button>
      </header>

      {usedTools.length > 0 && (
        <section className="sources-panel__section">
          <h3 className="sources-panel__section-title">活动</h3>
          <div className="sources-panel__tools">
            {usedTools.map((tool) => (
              <span key={tool} className="sources-panel__tool-tag">
                {tool}
              </span>
            ))}
          </div>
        </section>
      )}

      <section className="sources-panel__section sources-panel__section--grow">
        <h3 className="sources-panel__section-title">引用片段</h3>
        {count === 0 ? (
          <p className="sources-panel__empty">本轮回答未返回可展示的检索片段。</p>
        ) : (
          <>
            <ul className="sources-panel__list" role="listbox" aria-label="引用列表">
              {contexts.map((ctx, index) => {
                const subtitle = sourceSubtitle(ctx)
                const isActive = index === safeIndex
                return (
                  <li key={`${ctx.parent_id}-${index}`}>
                    <button
                      type="button"
                      role="option"
                      aria-selected={isActive}
                      className={`sources-panel__list-item ${isActive ? "is-active" : ""}`}
                      onClick={() => onSelectIndex(index)}
                    >
                      <span className="sources-panel__list-index">{index + 1}</span>
                      <span className="sources-panel__list-body">
                        <span className="sources-panel__list-title">{sourceLabel(ctx, index)}</span>
                        {subtitle && <span className="sources-panel__list-meta">{subtitle}</span>}
                        <span className="sources-panel__list-score">相关度 {formatCitationScore(ctx.score)}</span>
                      </span>
                    </button>
                  </li>
                )
              })}
            </ul>

            {active && (
              <article className="sources-panel__detail">
                <header className="sources-panel__detail-head">
                  <strong>{sourceLabel(active, safeIndex)}</strong>
                  {sourceSubtitle(active) && <span>{sourceSubtitle(active)}</span>}
                </header>
                <div className="sources-panel__detail-text">{active.parent_text || "（无正文）"}</div>
                {(active.child_ids?.length ?? 0) > 0 && (
                  <p className="sources-panel__detail-foot">
                    子块 ID：{active.child_ids!.slice(0, 3).join(", ")}
                    {active.child_ids!.length > 3 ? " …" : ""}
                  </p>
                )}
              </article>
            )}
          </>
        )}
      </section>

      <footer className="sources-panel__footer">
        <p className="sources-panel__query-hint" title={message.question ?? ""}>
          对应问题：
          {(message.question ?? "").length > 80
            ? `${(message.question ?? "").slice(0, 80)}…`
            : message.question ?? ""}
        </p>
      </footer>
    </aside>
  )
}
