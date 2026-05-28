const SUGGESTIONS = [
  "比较 LCDNet 和 U-shape 在水下图像增强上的方法差异",
  "LCDNet 的网络结构是怎么设计的？",
  "知识库里关于水下图像增强有哪些论文？",
  "帮我总结几篇基于 Transformer 的 UIE 方法",
]

interface ChatEmptyStateProps {
  onPickSuggestion: (text: string) => void
}

/**
 * 无消息时的欢迎区与示例问题。
 */
export function ChatEmptyState({ onPickSuggestion }: ChatEmptyStateProps) {
  return (
    <div className="chat-empty">
      <div className="chat-empty__hero">
        <span className="chat-empty__glow" aria-hidden="true" />
        <span className="avatar avatar--assistant avatar--lg" aria-hidden="true">
          PM
        </span>
        <h2>你好，我是 PaperMind</h2>
        <p>基于论文知识库的学术问答助手。我会检索证据、标注来源，并支持多轮对话记忆。</p>
      </div>
      <div className="chat-empty__suggestions">
        <p className="chat-empty__suggestions-label">试试这些问题</p>
        <div className="suggestion-grid">
          {SUGGESTIONS.map((text) => (
            <button
              key={text}
              type="button"
              className="suggestion-chip"
              onClick={() => onPickSuggestion(text)}
            >
              {text}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
