import { useMemo, useRef, useState } from "react"
import type { ChangeEvent } from "react"
import type { TaskRecord } from "../types/api"
import {
  filterTasks,
  formatTaskTime,
  isTaskProcessing,
  statusBadgeClass,
  statusLabel,
  type TaskStatusFilter,
} from "../utils/taskStatus"

interface KnowledgeBasePanelProps {
  tasks: TaskRecord[]
  busy?: boolean
  uploadBusy?: boolean
  uploadProgress?: string
  deleteBusy?: boolean
  selectedTaskIds: string[]
  onToggleTask: (taskId: string) => void
  onToggleSelectVisible: (visibleIds: string[]) => void
  onUpload: (event: ChangeEvent<HTMLInputElement>) => void
  onRefresh: () => void
  onDeleteSelected: () => void
  onDeleteOne: (task: TaskRecord) => void
  onUseForSearch: (task: TaskRecord) => void
}

const STATUS_FILTERS: { id: TaskStatusFilter; label: string }[] = [
  { id: "all", label: "全部" },
  { id: "ready", label: "可检索" },
  { id: "processing", label: "处理中" },
  { id: "failed", label: "失败" },
]

/**
 * 知识库管理页：上传、筛选、批量操作与论文卡片列表。
 */
export function KnowledgeBasePanel({
  tasks,
  busy = false,
  uploadBusy = false,
  uploadProgress = "",
  deleteBusy = false,
  selectedTaskIds,
  onToggleTask,
  onToggleSelectVisible,
  onUpload,
  onRefresh,
  onDeleteSelected,
  onDeleteOne,
  onUseForSearch,
}: KnowledgeBasePanelProps) {
  const [keyword, setKeyword] = useState("")
  const [statusFilter, setStatusFilter] = useState<TaskStatusFilter>("all")
  const uploadInputRef = useRef<HTMLInputElement>(null)

  const displayedTasks = useMemo(
    () => filterTasks(tasks, keyword, statusFilter),
    [tasks, keyword, statusFilter],
  )

  const selectedSet = useMemo(() => new Set(selectedTaskIds), [selectedTaskIds])
  const allVisibleSelected =
    displayedTasks.length > 0 && displayedTasks.every((t) => selectedSet.has(t.task_id))

  const stats = useMemo(() => {
    const ready = tasks.filter((t) => t.status === "succeeded").length
    const processing = tasks.filter((t) => isTaskProcessing(t.status)).length
    const failed = tasks.filter((t) => t.status === "failed").length
    return { total: tasks.length, ready, processing, failed }
  }, [tasks])

  const openUpload = () => {
    if (!uploadBusy) {
      uploadInputRef.current?.click()
    }
  }

  return (
    <section className="panel kb-page" aria-labelledby="kb-page-title">
      <header className="kb-page__header">
        <div className="kb-page__intro">
          <p className="kb-page__eyebrow">论文知识库</p>
          <h1 id="kb-page-title">知识库</h1>
          <p className="kb-page__subtitle">
            上传 PDF 后自动解析、切分并建立索引，可在聊天中限定检索范围。
          </p>
        </div>
        <div className="kb-page__header-actions">
          <button type="button" className="btn btn--ghost btn--sm" disabled={busy} onClick={onRefresh}>
            {busy ? "刷新中…" : "刷新列表"}
          </button>
          <button
            type="button"
            className="btn btn--danger-solid btn--sm"
            disabled={deleteBusy || selectedTaskIds.length === 0}
            onClick={onDeleteSelected}
          >
            {deleteBusy ? "删除中…" : `删除${selectedTaskIds.length ? ` (${selectedTaskIds.length})` : ""}`}
          </button>
        </div>
      </header>

      <div className="kb-page__stats" aria-live="polite">
        <div className="kb-stat">
          <span className="kb-stat__value">{stats.total}</span>
          <span className="kb-stat__label">全部文件</span>
        </div>
        <div className="kb-stat kb-stat--ready">
          <span className="kb-stat__value">{stats.ready}</span>
          <span className="kb-stat__label">可检索</span>
        </div>
        <div className="kb-stat kb-stat--processing">
          <span className="kb-stat__value">{stats.processing}</span>
          <span className="kb-stat__label">处理中</span>
        </div>
        <div className="kb-stat kb-stat--failed">
          <span className="kb-stat__value">{stats.failed}</span>
          <span className="kb-stat__label">失败</span>
        </div>
      </div>

      <div
        className={`kb-upload ${uploadBusy ? "kb-upload--busy" : ""}`}
        role="button"
        tabIndex={0}
        onClick={openUpload}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault()
            openUpload()
          }
        }}
      >
        <input
          ref={uploadInputRef}
          type="file"
          accept=".pdf,application/pdf"
          className="kb-upload__input"
          disabled={uploadBusy}
          onChange={onUpload}
          aria-hidden="true"
        />
        <span className="kb-upload__icon" aria-hidden="true">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75">
            <path d="M12 16V4M12 4l4 4M12 4L8 8" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M4 14v4a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-4" strokeLinecap="round" />
          </svg>
        </span>
        <span className="kb-upload__text">
          <strong>{uploadBusy ? uploadProgress || "上传中…" : "点击或拖拽上传 PDF"}</strong>
          <span>支持单文件上传，解析完成后即可用于 Agent 检索</span>
        </span>
        {!uploadBusy && (
          <span className="btn btn--primary kb-upload__btn">选择文件</span>
        )}
      </div>

      <div className="kb-toolbar">
        <div className="kb-page__search">
          <span className="kb-page__search-icon" aria-hidden="true">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="11" cy="11" r="7" />
              <path d="M20 20l-3-3" />
            </svg>
          </span>
          <input
            type="search"
            value={keyword}
            onChange={(e) => setKeyword(e.target.value)}
            placeholder="搜索文件名…"
            aria-label="搜索知识库文件"
            autoComplete="off"
          />
          {keyword.trim() && (
            <button
              type="button"
              className="kb-page__search-clear"
              onClick={() => setKeyword("")}
              aria-label="清除搜索"
            >
              ×
            </button>
          )}
        </div>

        <div className="kb-filters" role="tablist" aria-label="按状态筛选">
          {STATUS_FILTERS.map((f) => (
            <button
              key={f.id}
              type="button"
              role="tab"
              aria-selected={statusFilter === f.id}
              className={`kb-filter ${statusFilter === f.id ? "is-active" : ""}`}
              onClick={() => setStatusFilter(f.id)}
            >
              {f.label}
            </button>
          ))}
        </div>

        {displayedTasks.length > 0 && (
          <label className="kb-select-all">
            <input
              type="checkbox"
              checked={allVisibleSelected}
              onChange={() => onToggleSelectVisible(displayedTasks.map((t) => t.task_id))}
            />
            <span>全选当前列表 ({displayedTasks.length})</span>
          </label>
        )}
      </div>

      {busy && tasks.length === 0 ? (
        <ul className="kb-grid kb-grid--loading" aria-busy="true">
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <li key={i}>
              <div className="kb-card kb-card--skeleton" />
            </li>
          ))}
        </ul>
      ) : displayedTasks.length === 0 ? (
        <div className="kb-empty">
          <div className="kb-empty__visual" aria-hidden="true">
            <span className="kb-empty__glow" />
            <svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.25">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
              <path d="M14 2v6h6M8 13h8M8 17h5" strokeLinecap="round" />
            </svg>
          </div>
          <h2>{keyword.trim() || statusFilter !== "all" ? "没有匹配的文件" : "知识库还是空的"}</h2>
          <p>
            {keyword.trim() || statusFilter !== "all"
              ? "试试调整筛选条件，或清空搜索查看全部文件。"
              : "上传第一篇 PDF，系统会自动解析并建立可检索索引。"}
          </p>
          {!keyword.trim() && statusFilter === "all" && (
            <button type="button" className="btn btn--soft" onClick={openUpload}>
              上传 PDF
            </button>
          )}
        </div>
      ) : (
        <ul className="kb-grid">
          {displayedTasks.map((task) => {
            const selected = selectedSet.has(task.task_id)
            const canSearch = task.status === "succeeded"
            return (
              <li key={task.task_id}>
                <article className={`kb-card ${selected ? "is-selected" : ""}`}>
                  <div className="kb-card__top">
                    <label className="kb-card__check">
                      <input
                        type="checkbox"
                        checked={selected}
                        onChange={() => onToggleTask(task.task_id)}
                        aria-label={`选择 ${task.original_filename}`}
                      />
                    </label>

                    <div className="kb-card__icon" aria-hidden="true">
                      <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                        <path d="M14 2v6h6" />
                      </svg>
                      <span className="kb-card__icon-label">PDF</span>
                    </div>

                    <div className="kb-card__body">
                    <h3 className="kb-card__title" title={task.original_filename}>
                      {task.original_filename}
                    </h3>
                    <div className="kb-card__meta">
                      <span className={`badge ${statusBadgeClass(task.status)}`}>
                        {statusLabel(task.status)}
                      </span>
                      <span className="kb-card__id" title={task.task_id}>
                        {task.task_id.slice(0, 8)}
                      </span>
                    </div>
                    <p className="kb-card__time">{formatTaskTime(task.created_at)}</p>
                    {(task.num_parents != null || task.num_children != null) && task.status === "succeeded" && (
                      <p className="kb-card__index-hint">
                        父块 {task.num_parents ?? "—"} · 子块 {task.num_children ?? "—"}
                      </p>
                    )}
                    {task.message && task.status === "failed" && (
                      <p className="kb-card__error" title={task.message}>
                        {task.message}
                      </p>
                    )}
                    </div>
                  </div>

                  <div className="kb-card__actions">
                    <button
                      type="button"
                      className="btn btn--primary btn--sm"
                      disabled={!canSearch}
                      onClick={() => onUseForSearch(task)}
                    >
                      {canSearch ? "用于检索" : "等待就绪"}
                    </button>
                    <button
                      type="button"
                      className="btn btn--ghost btn--sm btn--danger-text"
                      disabled={deleteBusy}
                      onClick={() => onDeleteOne(task)}
                    >
                      删除
                    </button>
                  </div>
                </article>
              </li>
            )
          })}
        </ul>
      )}
    </section>
  )
}
