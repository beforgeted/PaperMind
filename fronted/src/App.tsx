import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import {
  chatWithAgentStream,
  createSession,
  deletePaper,
  deletePapers,
  deleteSession,
  fetchHealth,
  fetchSessionHistory,
  fetchTask,
  listSessions,
  listTasks,
  uploadPaper,
} from "./api/papers"
import { ChatPanel } from "./components/ChatPanel"
import { KnowledgeBasePanel } from "./components/KnowledgeBasePanel"
import { RecordsPanel } from "./components/RecordsPanel"
import type { SourcesPanelMessage } from "./components/SourcesPanel"
import type { AgentChatResponse, HealthResponse, SessionRecord, TaskRecord } from "./types/api"
import { normalizeCitationList } from "./utils/citations"
import { historyToChatMessages } from "./utils/sessionHistory"
import "./App.css"

type PageType = "files" | "chat" | "records"

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
  { id: "records", label: "聊天记录", icon: "R" },
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

  // Session management
  const [sessions, setSessions] = useState<SessionRecord[]>([])
  const [currentSessionId, setCurrentSessionId] = useState<string>("")
  const [sessionsBusy, setSessionsBusy] = useState(false)
  const [historyLoading, setHistoryLoading] = useState(false)

  const topKNum = useMemo(() => {
    if (topK === "") {
      return undefined
    }
    const n = Number(topK)
    return Number.isFinite(n) && n > 0 ? n : undefined
  }, [topK])

  const searchableTasks = useMemo(
    () => tasks.filter((row) => row.status === "succeeded"),
    [tasks],
  )

  const selectedScope = useMemo(
    () => searchableTasks.find((row) => row.task_id === scopeTaskId),
    [searchableTasks, scopeTaskId],
  )

  const selectedTaskIdSet = useMemo(() => new Set(selectedTaskIds), [selectedTaskIds])

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
    void loadSessions()
  }, [loadHealth, loadTasks])

  const loadSessions = useCallback(async () => {
    setSessionsBusy(true)
    try {
      const result = await listSessions()
      setSessions(result.sessions || [])
    } catch {
      // Sessions may not be available yet
    } finally {
      setSessionsBusy(false)
    }
  }, [])

  const onNewSession = useCallback(async () => {
    try {
      const session = await createSession()
      setSessions((prev) => [session, ...prev])
      setCurrentSessionId(session.session_id)
      setChatMessages([])
    } catch (err) {
      setBanner(err instanceof Error ? err.message : String(err))
    }
  }, [])

  const onSwitchSession = useCallback(async (sessionId: string) => {
    if (!sessionId) {
      return
    }
    setCurrentSessionId(sessionId)
    setSourcesPanelMessage(null)
    setHistoryLoading(true)
    setChatMessages([])
    try {
      const data = await fetchSessionHistory(sessionId)
      setChatMessages(historyToChatMessages(data.messages))
      if (data.title) {
        setSessions((prev) =>
          prev.map((s) => (s.session_id === sessionId ? { ...s, title: data.title } : s)),
        )
      }
    } catch (error) {
      setBanner(error instanceof Error ? error.message : String(error))
      setChatMessages([])
    } finally {
      setHistoryLoading(false)
    }
  }, [])

  const onDeleteSession = useCallback(async (sessionId: string, event: React.MouseEvent) => {
    event.stopPropagation()
    if (!window.confirm("确认删除此会话？")) return
    try {
      await deleteSession(sessionId)
      setSessions((prev) => prev.filter((s) => s.session_id !== sessionId))
      if (currentSessionId === sessionId) {
        setCurrentSessionId("")
        setChatMessages([])
      }
    } catch (err) {
      setBanner(err instanceof Error ? err.message : String(err))
    }
  }, [currentSessionId])

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
  const toggleSelectVisibleTasks = (visibleIds: string[]) => {
    const visible = new Set(visibleIds)
    const allSelected =
      visibleIds.length > 0 && visibleIds.every((id) => selectedTaskIdSet.has(id))
    if (allSelected) {
      setSelectedTaskIds((prev) => prev.filter((id) => !visible.has(id)))
      return
    }
    setSelectedTaskIds((prev) => Array.from(new Set([...prev, ...visibleIds])))
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

    // Auto-create session if not already in one
    let sid = currentSessionId
    if (!sid) {
      try {
        const session = await createSession()
        setSessions((prev) => [session, ...prev])
        sid = session.session_id
        setCurrentSessionId(sid)
      } catch (err) {
        setBanner(err instanceof Error ? err.message : String(err))
      }
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
        { query: question, top_k: topKNum, task_id: scopeTaskId.trim() || undefined, session_id: sid || undefined },
        (event) => {
          const type = event.type as string
          if (type === "delta" && event.text) {
            appendAnswer(String(event.text))
          } else if (type === "done") {
            finalizeMessage(event)
          } else if (type === "status" && event.phase === "title" && event.title) {
            // Auto-generated title from first turn
            const title = String(event.title)
            setSessions((prev) =>
              prev.map((s) => (s.session_id === sid ? { ...s, title } : s)),
            )
          }
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
                if (item.id === "chat" || item.id === "files" || item.id === "records") {
                  setActivePage(item.id)
                  if (item.id === "records") void loadSessions()
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
        {banner && <div className="error-box">{banner}</div>}

        {activePage === "records" ? (
          <RecordsPanel
            sessions={sessions}
            currentSessionId={currentSessionId}
            busy={sessionsBusy}
            onNewSession={() => {
              void onNewSession()
              setActivePage("chat")
            }}
            onOpenSession={(sessionId) => {
              void onSwitchSession(sessionId)
              setActivePage("chat")
            }}
            onDeleteSession={(sessionId, e) => void onDeleteSession(sessionId, e)}
          />
        ) : activePage === "files" ? (
          <KnowledgeBasePanel
            tasks={tasks}
            busy={tasksBusy}
            uploadBusy={uploadBusy}
            uploadProgress={uploadProgress}
            deleteBusy={deleteBusy}
            selectedTaskIds={selectedTaskIds}
            onToggleTask={toggleTaskSelection}
            onToggleSelectVisible={toggleSelectVisibleTasks}
            onUpload={(event) => void onPickFile(event)}
            onRefresh={() => void loadTasks()}
            onDeleteSelected={() => void onDeleteSelected()}
            onDeleteOne={(task) => void onDeleteOne(task)}
            onUseForSearch={(task) => {
              setScopeTaskId(task.task_id)
              setActivePage("chat")
              setBanner(`已将检索范围切换为：${task.original_filename}`)
            }}
          />
        ) : (
          <ChatPanel
            chatMessages={chatMessages}
            historyLoading={historyLoading}
            chatInput={chatInput}
            setChatInput={setChatInput}
            answerBusy={answerBusy}
            onSendMessage={() => void onSendMessage()}
            chatListRef={chatListRef}
            sourcesPanelMessage={sourcesPanelMessage}
            sourcesPanelIndex={sourcesPanelIndex}
            setSourcesPanelIndex={setSourcesPanelIndex}
            setSourcesPanelMessage={setSourcesPanelMessage}
            onOpenSources={openSourcesPanel}
            formatTime={formatTime}
            sessions={sessions}
            currentSessionId={currentSessionId}
            onNewSession={() => void onNewSession()}
            onSwitchSession={(id) => void onSwitchSession(id)}
            onOpenRecords={() => {
              void loadSessions()
              setActivePage("records")
            }}
            scopeTaskId={scopeTaskId}
            setScopeTaskId={setScopeTaskId}
            topK={topK}
            setTopK={setTopK}
            searchableTasks={searchableTasks}
            selectedScope={selectedScope}
            onRefreshOneTask={() => void onRefreshOneTask()}
            health={health}
            healthErr={healthErr}
          />
        )}
      </main>
    </div>
  )
}

export default App
