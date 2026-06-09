"""结构化日志事件名常量。"""

# HTTP
HTTP_REQUEST_STARTED = "http_request_started"
HTTP_REQUEST_COMPLETED = "http_request_completed"
HTTP_REQUEST_FAILED = "http_request_failed"

# Agent API
AGENT_CHAT_STARTED = "agent_chat_started"
AGENT_CHAT_COMPLETED = "agent_chat_completed"
AGENT_CHAT_FAILED = "agent_chat_failed"

# Chat workflow
CHAT_STREAM_STARTED = "chat_stream_started"
CHAT_STREAM_COMPLETED = "chat_stream_completed"
CHAT_STREAM_FAILED = "chat_stream_failed"

# Orchestrator
RUN_CREATED = "run_created"
RUN_STATUS_CHANGED = "run_status_changed"
GRAPH_NODE_UPDATED = "graph_node_updated"
MAIN_AGENT_DECISION = "main_agent_decision"
SUB_AGENT_STEP_COMPLETED = "sub_agent_step_completed"
MCP_TASK_SUBMITTED = "mcp_task_submitted"

# Agent executor
TOOL_CALL_COMPLETED = "tool_call_completed"
AGENT_EXECUTION_STARTED = "agent_execution_started"
AGENT_EXECUTION_COMPLETED = "agent_execution_completed"

# Context builder
CONTEXT_BUILT = "context_built"

# MCP audit
MCP_TOOL_CALL = "mcp_tool_call"

# 写入 agent.json.log 的业务事件（按 event 字段过滤）
AGENT_BUSINESS_EVENTS = frozenset(
    {
        AGENT_CHAT_STARTED,
        AGENT_CHAT_COMPLETED,
        AGENT_CHAT_FAILED,
        CHAT_STREAM_STARTED,
        CHAT_STREAM_COMPLETED,
        CHAT_STREAM_FAILED,
        RUN_CREATED,
        RUN_STATUS_CHANGED,
        GRAPH_NODE_UPDATED,
        MAIN_AGENT_DECISION,
        SUB_AGENT_STEP_COMPLETED,
        MCP_TASK_SUBMITTED,
        TOOL_CALL_COMPLETED,
        AGENT_EXECUTION_STARTED,
        AGENT_EXECUTION_COMPLETED,
        CONTEXT_BUILT,
    }
)
