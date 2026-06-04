import type { AgentChatResponse } from "../types/api"

export interface SessionHistoryMessageDto {
  id: string
  question: string
  answer: string
  created_at: string
  used_tools: string[]
}

export interface SessionHistoryResponseDto {
  session_id: string
  title: string
  messages: SessionHistoryMessageDto[]
  source: string
}

export interface ChatMessageFromHistory {
  id: string
  question: string
  answer: string
  createdAt: string
  contexts: AgentChatResponse["contexts"]
  sources: AgentChatResponse["sources"]
  used_tools: AgentChatResponse["used_tools"]
}

/**
 * 将后端会话历史转为聊天列表项。
 */
export function historyToChatMessages(messages: SessionHistoryMessageDto[]): ChatMessageFromHistory[] {
  return messages.map((m) => ({
    id: m.id || `hist-${m.created_at}`,
    question: m.question,
    answer: m.answer,
    createdAt: m.created_at || new Date().toISOString(),
    contexts: [],
    sources: [],
    used_tools: Array.isArray(m.used_tools) ? m.used_tools : [],
  }))
}
