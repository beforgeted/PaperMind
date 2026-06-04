import { type RefObject } from "react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import type { AgentChatResponse, HealthResponse, SessionRecord, TaskRecord } from "../types/api"
import { ChatEmptyState } from "./ChatEmptyState"
import { SourcesPanel, type SourcesPanelMessage } from "./SourcesPanel"
import { StreamingDots } from "./StreamingDots"

export interface ChatMessageItem {
  id: string
  question: string
  answer: string
  createdAt: string
  contexts: AgentChatResponse["contexts"]
  sources: AgentChatResponse["sources"]
  used_tools: AgentChatResponse["used_tools"]
}

interface ChatPanelProps {
  chatMessages: ChatMessageItem[]
  historyLoading?: boolean
  chatInput: string
  setChatInput: (value: string) => void
  answerBusy: boolean
  onSendMessage: () => void
  chatListRef: RefObject<HTMLDivElement | null>
  sourcesPanelMessage: SourcesPanelMessage | null
  sourcesPanelIndex: number
  setSourcesPanelIndex: (index: number) => void
  setSourcesPanelMessage: (msg: SourcesPanelMessage | null) => void
  onOpenSources: (item: ChatMessageItem) => void
  formatTime: (input: string) => string
  sessions: SessionRecord[]
  currentSessionId: string
  onNewSession: () => void
  onSwitchSession: (sessionId: string) => void
  onOpenRecords: () => void
  scopeTaskId: string
  setScopeTaskId: (id: string) => void
  topK: number | ""
  setTopK: (value: number | "") => void
  searchableTasks: TaskRecord[]
  selectedScope?: TaskRecord
  onRefreshOneTask: () => void
  health: HealthResponse | null
  healthErr: string | null
}

/**
 * 聊天主界面：会话栏、消息流、输入区与来源侧栏。
 */
export function ChatPanel({
  chatMessages,
  historyLoading = false,
  chatInput,
  setChatInput,
  answerBusy,
  onSendMessage,
  chatListRef,
  sourcesPanelMessage,
  sourcesPanelIndex,
  setSourcesPanelIndex,
  setSourcesPanelMessage,
  onOpenSources,
  formatTime,
  sessions,
  currentSessionId,
  onNewSession,
  onSwitchSession,
  onOpenRecords,
  scopeTaskId,
  setScopeTaskId,
  topK,
  setTopK,
  searchableTasks,
  selectedScope,
  onRefreshOneTask,
  health,
  healthErr,
}: ChatPanelProps) {
  const activeSession = sessions.find((s) => s.session_id === currentSessionId)
  const sessionTitle = activeSession?.title || (currentSessionId ? "当前会话" : "新对话")

  return (
    <section className={`chat-page ${sourcesPanelMessage ? "chat-page--with-sources" : ""}`}>
      <div className="chat-shell">
        <header className="chat-header">
          <div className="chat-header__main">
            <h1 className="chat-header__title">{sessionTitle}</h1>
            <p className="chat-header__hint">
              {currentSessionId ? "多轮上下文已启用" : "发送首条消息将自动创建会话"}
            </p>
          </div>
          <div className="chat-header__actions">
            <button type="button" className="btn btn--ghost btn--sm" onClick={onOpenRecords}>
              全部记录
            </button>
            <button type="button" className="btn btn--soft btn--sm" onClick={onNewSession}>
              新对话
            </button>
          </div>
        </header>

        {sessions.length > 0 && (
          <div className="chat-session-strip" role="tablist" aria-label="最近会话">
            {sessions.slice(0, 8).map((s) => (
              <button
                key={s.session_id}
                type="button"
                role="tab"
                aria-selected={currentSessionId === s.session_id}
                className={`session-pill ${currentSessionId === s.session_id ? "is-active" : ""}`}
                onClick={() => onSwitchSession(s.session_id)}
                title={s.title || "新会话"}
              >
                {s.title || "新会话"}
              </button>
            ))}
          </div>
        )}

        <div className="chat-list" ref={chatListRef}>
          {historyLoading && (
            <div className="chat-history-loading" role="status" aria-live="polite">
              <StreamingDots />
              <span>正在恢复对话记录…</span>
            </div>
          )}
          {!historyLoading && chatMessages.length === 0 && (
            <ChatEmptyState onPickSuggestion={(text) => setChatInput(text)} />
          )}
          {chatMessages.map((item) => (
            <article key={item.id} className="chat-item">
              <div className="chat-row chat-row--user">
                <div className="chat-message">
                  <div className="chat-bubble chat-bubble--user">{item.question}</div>
                </div>
                <span className="avatar avatar--user" title="你" aria-hidden="true">
                  你
                </span>
              </div>
              <div className="chat-row chat-row--assistant">
                <span className="avatar avatar--assistant" title="PaperMind" aria-hidden="true">
                  PM
                </span>
                <div className="chat-message">
                  <div className="chat-item__head">
                    <strong>PaperMind</strong>
                    <time dateTime={item.createdAt}>{formatTime(item.createdAt)}</time>
                  </div>
                  <div
                    className={`chat-bubble chat-bubble--assistant answer-content ${
                      item.answer ? "" : "is-streaming"
                    }`}
                  >
                    {item.answer ? (
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>{item.answer}</ReactMarkdown>
                    ) : (
                      <span className="streaming-placeholder">
                        正在思考
                        <StreamingDots />
                      </span>
                    )}
                  </div>
                  {(item.answer || item.contexts.length > 0 || item.used_tools.length > 0) && (
                    <div className="message-actions">
                      <button
                        type="button"
                        className={`message-actions__btn message-actions__btn--sources ${
                          sourcesPanelMessage?.id === item.id ? "is-active" : ""
                        }`}
                        onClick={() => {
                          if (sourcesPanelMessage?.id === item.id) {
                            setSourcesPanelMessage(null)
                          } else {
                            onOpenSources(item)
                          }
                        }}
                        aria-expanded={sourcesPanelMessage?.id === item.id}
                        aria-controls="sources-panel"
                      >
                        <span className="message-actions__icon" aria-hidden="true">
                          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                            <polyline points="14 2 14 8 20 8" />
                          </svg>
                        </span>
                        引用来源
                        {item.contexts.length > 0 && (
                          <span className="message-actions__count">{item.contexts.length}</span>
                        )}
                      </button>
                    </div>
                  )}
                </div>
              </div>
            </article>
          ))}
        </div>

        <form
          className="chat-composer"
          onSubmit={(event) => {
            event.preventDefault()
            onSendMessage()
          }}
        >
          <textarea
            value={chatInput}
            onChange={(event) => setChatInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault()
                onSendMessage()
              }
            }}
            placeholder="输入论文相关问题，Enter 发送，Shift+Enter 换行"
            rows={2}
            disabled={answerBusy}
          />
          <div className="composer-footer">
            <div className="composer-footer__meta">
              <span className={`connection-pill ${healthErr ? "is-error" : health ? "is-ok" : ""}`}>
                <span className="connection-pill__dot" aria-hidden="true" />
                {healthErr ? "服务异常" : health ? "已连接" : "连接中…"}
              </span>
              <div className="composer-settings" aria-label="检索设置">
                <label className="composer-settings__field">
                  <span className="composer-settings__label">范围</span>
                  <select
                    value={scopeTaskId}
                    onChange={(event) => setScopeTaskId(event.target.value)}
                    title={selectedScope?.original_filename ?? "全部已完成论文"}
                  >
                    <option value="">全部（{searchableTasks.length}）</option>
                    {searchableTasks.map((task) => (
                      <option key={task.task_id} value={task.task_id}>
                        {task.original_filename}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="composer-settings__field composer-settings__field--topk">
                  <span className="composer-settings__label">Top K</span>
                  <input
                    type="number"
                    min={1}
                    value={topK}
                    onChange={(event) => {
                      const value = event.target.value
                      if (value === "") {
                        setTopK("")
                        return
                      }
                      setTopK(Math.max(1, Math.floor(Number(value)) || 1))
                    }}
                  />
                </label>
                <button
                  type="button"
                  className="composer-settings__refresh"
                  disabled={!scopeTaskId}
                  onClick={onRefreshOneTask}
                  title="刷新当前论文解析状态"
                >
                  刷新
                </button>
              </div>
            </div>
            <div className="composer-actions">
              <button
                type="submit"
                className="send-button"
                disabled={answerBusy || !chatInput.trim()}
                aria-label="发送消息"
              >
                {answerBusy ? (
                  <span className="send-button__spinner" aria-hidden="true" />
                ) : (
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                    <path
                      d="M5 12h14M13 6l6 6-6 6"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                )}
              </button>
            </div>
          </div>
        </form>
      </div>

      <SourcesPanel
        message={sourcesPanelMessage}
        selectedIndex={sourcesPanelIndex}
        onSelectIndex={setSourcesPanelIndex}
        onClose={() => setSourcesPanelMessage(null)}
      />
    </section>
  )
}
