import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { chatWithAgentStream, deletePaper, deletePapers, fetchHealth, fetchTask, listTasks, uploadPaper } from "./api/papers"
import { SourcesPanel, type SourcesPanelMessage } from "./components/SourcesPanel"
import type { AgentChatResponse, HealthResponse, TaskRecord } from "./types/api"
import { normalizeCitationList } from "./utils/citations"
import "./App.css"

type PageType = "files" | "chat"

interface MenuItem {
  id: PageType | "records" | "tags" | "users" | "profile"
  label: string
  icon: string
  disabled?: boolean
}

interface ChatMessage {
  id: string
  question: string
  answer: string
  createdAt: string
  contexts: AgentChatResponse["contexts"]
  sources: AgentChatResponse["sources"]
  used_tools: AgentChatResponse["used_tools"]
}

/**
 * 根据任务状态返回用于样式类名的后缀。
 */
function statusBadgeClass(status: TaskRecord["status"]): string {
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
function statusLabel(status: TaskRecord["status"]): string {
  const map: Record<TaskRecord["status"], string> = {
    pending: "排队",
    parsing: "处理中",
    indexing: "索引中",
    succeeded: "已完成",
    failed: "失败",
  }
  return map[status] ?? status
}

/**
 * 将时间转成易读格式。
 */
function formatTime(input: string): string {
  const date = new Date(input)
  if (Number.isNaN(date.getTime())) {
    return input
  }
  return date.toLocaleString("zh-CN", { hour12: false })
}

const menuItems: MenuItem[] = [
  { id: "chat", label: "聊天助手", icon: "C" },
  { id: "records", label: "聊天记录", icon: "R", disabled: true },
  { id: "files", label: "知识库", icon: "K" },
  { id: "tags", label: "组织标签", icon: "T", disabled: true },
  { id: "users", label: "用户管理", icon: "U", disabled: true },
  { id: "profile", label: "个人中心", icon: "P", disabled: true },
]

/**
 * 主界面：拆分为文件管理与对话管理两页。
 */
function App() {
  const [activePage, setActivePage] = useState<PageType>("chat")
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [healthErr, setHealthErr] = useState<string | null>(null)
  const [banner, setBanner] = useState<string | null>(null)

  const [tasks, setTasks] = useState<TaskRecord[]>([])
  const [tasksBusy, setTasksBusy] = useState(false)
  const [uploadBusy, setUploadBusy] = useState(false)
  const [uploadProgress, setUploadProgress] = useState("")
  const [fileKeyword, setFileKeyword] = useState("")
  const [selectedTaskIds, setSelectedTaskIds] = useState<string[]>([])
  const [deleteBusy, setDeleteBusy] = useState(false)

  const [scopeTaskId, setScopeTaskId] = useState("")
  const [topK, setTopK] = useState<number | "">(5)
  const [chatInput, setChatInput] = useState("")
  const [answerBusy, setAnswerBusy] = useState(false)
  const [sourcesPanelMessage, setSourcesPanelMessage] = useState<SourcesPanelMessage | null>(null)
  const [sourcesPanelIndex, setSourcesPanelIndex] = useState(0)
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([])
  const chatListRef = useRef<HTMLDivElement | null>(null)

  const topKNum = useMemo(() => {
    if (topK === "") {
      return undefined
    }
    const n = Number(topK)
    return Number.isFinite(n) && n > 0 ? n : undefined
  }, [topK])

  const filteredTasks = useMemo(() => {
    const keyword = fileKeyword.trim().toLowerCase()
    if (!keyword) {
      return tasks
    }
    return tasks.filter((row) => row.original_filename.toLowerCase().includes(keyword))
  }, [tasks, fileKeyword])

  const searchableTasks = useMemo(
    () => tasks.filter((row) => row.status === "succeeded"),
    [tasks],
  )

  const selectedScope = useMemo(
    () => searchableTasks.find((row) => row.task_id === scopeTaskId),
    [searchableTasks, scopeTaskId],
  )

  const selectedTaskIdSet = useMemo(() => new Set(selectedTaskIds), [selectedTaskIds])
  const allVisibleSelected = filteredTasks.length > 0 && filteredTasks.every((task) => selectedTaskIdSet.has(task.task_id))

  const loadHealth = useCallback(async () => {
    setHealthErr(null)
    try {
      const result = await fetchHealth()
      setHealth(result)
    } catch (error) {
      setHealth(null)
      setHealthErr(error instanceof Error ? error.message : String(error))
    }
  }, [])

  /**
   * 拉取任务列表，默认取最新 50 条。
   */
  const loadTasks = useCallback(async () => {
    setTasksBusy(true)
    setBanner(null)
    try {
      const rows = await listTasks(50)
      setTasks(rows)
    } catch (error) {
      setBanner(error instanceof Error ? error.message : String(error))
    } finally {
      setTasksBusy(false)
    }
  }, [])

  useEffect(() => {
    void loadHealth()
    void loadTasks()
  }, [loadHealth, loadTasks])

  useEffect(() => {
    chatListRef.current?.scrollTo({
      top: chatListRef.current.scrollHeight,
      behavior: "smooth",
    })
  }, [chatMessages])

  /**
   * 上传文件并自动设置检索范围。
   */
  const onPickFile = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    event.target.value = ""
    if (!file) {
      return
    }
    setUploadBusy(true)
    setUploadProgress("准备上传...")
    setBanner(null)
    try {
      const result = await uploadPaper(file, (uploaded, total) => {
        setUploadProgress(`上传分片 ${uploaded}/${total}`)
      })
      setScopeTaskId(result.task_id)
      setBanner(`上传成功，任务 ID：${result.task_id}`)
      await loadTasks()
    } catch (error) {
      setBanner(error instanceof Error ? error.message : String(error))
    } finally {
      setUploadBusy(false)
      setUploadProgress("")
    }
  }

  /**
   * 按 task_id 刷新单条任务状态。
   */
  const onRefreshOneTask = async () => {
    const id = scopeTaskId.trim()
    if (!id) {
      setBanner("当前选择的是全部知识库，无需刷新单篇状态。")
      return
    }
    setBanner(null)
    try {
      const item = await fetchTask(id)
      setTasks((prev) => {
        const idx = prev.findIndex((row) => row.task_id === item.task_id)
        if (idx < 0) {
          return [item, ...prev]
        }
        const next = [...prev]
        next[idx] = item
        return next
      })
    } catch (error) {
      setBanner(error instanceof Error ? error.message : String(error))
    }
  }

  /**
   * 切换单个文件的批量选择状态。
   */
  const toggleTaskSelection = (taskId: string) => {
    setSelectedTaskIds((prev) =>
      prev.includes(taskId) ? prev.filter((id) => id !== taskId) : [...prev, taskId],
    )
  }

  /**
   * 切换当前筛选结果的全选状态。
   */
  const toggleSelectVisibleTasks = () => {
    if (allVisibleSelected) {
      const visible = new Set(filteredTasks.map((task) => task.task_id))
      setSelectedTaskIds((prev) => prev.filter((id) => !visible.has(id)))
      return
    }
    setSelectedTaskIds((prev) => Array.from(new Set([...prev, ...filteredTasks.map((task) => task.task_id)])))
  }

  /**
   * 删除单个文件任务及其知识库数据。
   */
  const onDeleteOne = async (task: TaskRecord) => {
    if (!window.confirm(`确认删除「${task.original_filename}」吗？此操作会删除文件、解析结果和索引。`)) {
      return
    }
    setDeleteBusy(true)
    setBanner(null)
    try {
      await deletePaper(task.task_id)
      setSelectedTaskIds((prev) => prev.filter((id) => id !== task.task_id))
      if (scopeTaskId === task.task_id) {
        setScopeTaskId("")
      }
      await loadTasks()
      setBanner("删除成功。")
    } catch (error) {
      setBanner(error instanceof Error ? error.message : String(error))
    } finally {
      setDeleteBusy(false)
    }
  }

  /**
   * 批量删除已勾选的文件任务及其知识库数据。
   */
  const onDeleteSelected = async () => {
    if (selectedTaskIds.length === 0) {
      setBanner("请先勾选要删除的文件。")
      return
    }
    if (!window.confirm(`确认删除已选的 ${selectedTaskIds.length} 个文件吗？此操作不可撤销。`)) {
      return
    }
    setDeleteBusy(true)
    setBanner(null)
    try {
      const result = await deletePapers(selectedTaskIds)
      const failed = result.results.filter((item) => !item.deleted)
      if (scopeTaskId && selectedTaskIds.includes(scopeTaskId)) {
        setScopeTaskId("")
      }
      setSelectedTaskIds([])
      await loadTasks()
      setBanner(failed.length > 0 ? `部分删除失败：${failed.length} 个。` : "批量删除成功。")
    } catch (error) {
      setBanner(error instanceof Error ? error.message : String(error))
    } finally {
      setDeleteBusy(false)
    }
  }

  /**
   * 打开指定消息的引用来源侧栏。
   */
  const openSourcesPanel = (item: ChatMessage) => {
    setSourcesPanelMessage({
      id: item.id,
      question: item.question,
      contexts: normalizeCitationList(item.contexts),
      sources: Array.isArray(item.sources) ? item.sources : [],
      used_tools: Array.isArray(item.used_tools) ? item.used_tools : [],
    })
    setSourcesPanelIndex(0)
  }

  /**
   * 通过 Agent 发送消息并流式获取回答。
   */
  const onSendMessage = async () => {
    const question = chatInput.trim()
    if (!question) {
      setBanner("请输入对话问题。")
      return
    }
    setAnswerBusy(true)
    setBanner(null)
    const messageId = `${Date.now()}`
    setChatMessages((prev) => [
      ...prev,
      {
        id: messageId,
        question,
        answer: "",
        createdAt: new Date().toISOString(),
        contexts: [],
        sources: [],
        used_tools: [],
      },
    ])
    setChatInput("")

    const appendAnswer = (text: string) => {
      setChatMessages((prev) =>
        prev.map((item) =>
          item.id === messageId
            ? { ...item, answer: item.answer + text }
            : item,
        ),
      )
    }

    const finalizeMessage = (data: Record<string, unknown>) => {
      const contexts = normalizeCitationList(data.contexts)
      const sources = Array.isArray(data.sources) ? (data.sources as AgentChatResponse["sources"]) : []
      const used_tools = Array.isArray(data.used_tools) ? (data.used_tools as string[]) : []
      setChatMessages((prev) =>
        prev.map((item) =>
          item.id === messageId
            ? { ...item, contexts, sources, used_tools }
            : item,
        ),
      )
      setSourcesPanelMessage((prev) =>
        prev?.id === messageId ? { ...prev, contexts, sources, used_tools } : prev,
      )
    }

    try {
      await chatWithAgentStream(
        { query: question, top_k: topKNum, task_id: scopeTaskId.trim() || undefined },
        (event) => {
          const type = event.type as string
          if (type === "delta" && event.text) {
            appendAnswer(String(event.text))
          } else if (type === "done") {
            finalizeMessage(event)
          }
          // status events (routing, planning) are received but not displayed yet
        },
      )
    } catch (error) {
      setChatMessages((prev) =>
        prev.map((item) =>
          item.id === messageId && !item.answer
            ? { ...item, answer: `请求失败：${error instanceof Error ? error.message : String(error)}` }
            : item,
        ),
      )
    } finally {
      setAnswerBusy(false)
    }
  }

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand__mark" aria-hidden="true">PM</span>
          <span>PaperMind</span>
        </div>
        <nav className="menu">
          {menuItems.map((item) => (
            <button
              key={item.id}
              type="button"
              className={`menu__item ${activePage === item.id ? "is-active" : ""}`}
              disabled={item.disabled}
              onClick={() => {
                if (item.id === "chat" || item.id === "files") {
                  setActivePage(item.id)
                }
              }}
            >
              <span className="menu__icon" aria-hidden="true">{item.icon}</span>
              <span>{item.label}</span>
            </button>
          ))}
        </nav>
        <button type="button" className="sidebar__collapse" aria-label="收起侧边栏">
          =
        </button>
      </aside>

      <main className="main">
        {activePage === "files" && (
          <header className="topbar topbar--files">
            <div className="topbar-query topbar-query--files">
              <input
                type="text"
                value={fileKeyword}
                onChange={(event) => setFileKeyword(event.target.value)}
                placeholder="检索知识库文件名"
                autoComplete="off"
              />
            </div>
          </header>
        )}

        {banner && <div className="error-box">{banner}</div>}

        {activePage === "files" ? (
          <section className="panel">
            <div className="toolbar">
              <button type="button" className="btn btn--primary" disabled={uploadBusy}>
                <label className="upload-label">
                  {uploadBusy ? uploadProgress || "上传中..." : "新增"}
                  <input type="file" accept=".pdf,application/pdf" disabled={uploadBusy} onChange={(event) => void onPickFile(event)} />
                </label>
              </button>
              <button type="button" className="btn" disabled={tasksBusy} onClick={() => void loadTasks()}>
                {tasksBusy ? "刷新中..." : "刷新"}
              </button>
              <button type="button" className="btn btn--danger-solid" disabled={deleteBusy || selectedTaskIds.length === 0} onClick={() => void onDeleteSelected()}>
                {deleteBusy ? "删除中..." : `批量删除${selectedTaskIds.length ? ` (${selectedTaskIds.length})` : ""}`}
              </button>
            </div>

            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>
                      <input
                        type="checkbox"
                        checked={allVisibleSelected}
                        disabled={filteredTasks.length === 0}
                        onChange={toggleSelectVisibleTasks}
                        aria-label="选择当前列表全部文件"
                      />
                    </th>
                    <th className="col-filename">文件名</th>
                    <th>上传状态</th>
                    <th>任务标签</th>
                    <th>是否公开</th>
                    <th>上传时间</th>
                    <th>操作</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredTasks.length === 0 && (
                    <tr>
                      <td colSpan={7} className="muted">
                        暂无文件
                      </td>
                    </tr>
                  )}
                  {filteredTasks.map((task) => (
                    <tr key={task.task_id}>
                      <td>
                        <input
                          type="checkbox"
                          checked={selectedTaskIdSet.has(task.task_id)}
                          onChange={() => toggleTaskSelection(task.task_id)}
                          aria-label={`选择 ${task.original_filename}`}
                        />
                      </td>
                      <td className="col-filename">
                        <span className="filename-ellipsis" data-full-name={task.original_filename} title={task.original_filename}>
                          {task.original_filename}
                        </span>
                      </td>
                      <td>
                        <span className={`badge ${statusBadgeClass(task.status)}`}>{statusLabel(task.status)}</span>
                      </td>
                      <td>{task.task_id.slice(0, 8)}</td>
                      <td>
                        <span className="chip">私有</span>
                      </td>
                      <td>{formatTime(task.created_at)}</td>
                      <td>
                        <div className="table-actions">
                          <button
                            type="button"
                            className="btn btn--primary"
                            disabled={task.status !== "succeeded"}
                            onClick={() => {
                              setScopeTaskId(task.task_id)
                              setActivePage("chat")
                              setBanner(`已将检索范围切换为：${task.original_filename}`)
                            }}
                          >
                            {task.status === "succeeded" ? "用于检索" : "不可检索"}
                          </button>
                          <button
                            type="button"
                            className="btn btn--danger-solid"
                            disabled={deleteBusy}
                            onClick={() => void onDeleteOne(task)}
                          >
                            删除
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        ) : (
          <section className={`chat-page ${sourcesPanelMessage ? "chat-page--with-sources" : ""}`}>
            <div className="chat-shell">
              <div className="chat-list" ref={chatListRef}>
                {chatMessages.length === 0 && (
                  <div className="chat-empty">
                    <span className="avatar avatar--assistant" aria-hidden="true">PM</span>
                    <h2>你好，我是 PaperMind</h2>
                    <p>你可以直接提问论文内容，我会结合知识库给出回答并附带引用来源。</p>
                  </div>
                )}
                {chatMessages.map((item) => (
                  <article key={item.id} className="chat-item">
                    <div className="chat-row chat-row--user">
                      <div className="chat-message">
                        <div className="chat-item__head">{formatTime(item.createdAt)}</div>
                        <div className="chat-bubble chat-bubble--user">{item.question}</div>
                      </div>
                      <span className="avatar avatar--user" aria-hidden="true">你</span>
                    </div>
                    <div className="chat-row chat-row--assistant">
                      <span className="avatar avatar--assistant" aria-hidden="true">PM</span>
                      <div className="chat-message">
                        <div className="chat-item__head">
                          <strong>PaperMind</strong>
                          <span>{formatTime(item.createdAt)}</span>
                        </div>
                        <div className={`chat-bubble chat-bubble--assistant answer-content ${item.answer ? "" : "is-streaming"}`}>
                          {item.answer ? <ReactMarkdown remarkPlugins={[remarkGfm]}>{item.answer}</ReactMarkdown> : "正在生成回答…"}
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
                                  openSourcesPanel(item)
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
                              来源
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
                  void onSendMessage()
                }}
              >
                <textarea
                  value={chatInput}
                  onChange={(event) => setChatInput(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && !event.shiftKey) {
                      event.preventDefault()
                      void onSendMessage()
                    }
                  }}
                  placeholder="给 PaperMind 发送消息"
                />
                <div className="composer-footer">
                  <div className="composer-footer__meta">
                    <span className="connection-state">
                      连接
                      <span className={healthErr ? "connection-state__bad" : "connection-state__ok"}>
                        {healthErr ? "异常" : health ? "正常" : "…"}
                      </span>
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
                        <span className="composer-settings__label">top_k</span>
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
                        onClick={() => void onRefreshOneTask()}
                        title="刷新当前论文解析状态"
                      >
                        刷新
                      </button>
                    </div>
                  </div>
                  <div className="composer-actions">
                    <button type="submit" className="send-button" disabled={answerBusy || !chatInput.trim()}>
                      {answerBusy ? "..." : "➤"}
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
        )}
      </main>
    </div>
  )
}

export default App
