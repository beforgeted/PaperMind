# PaperMind Multi-Agent

PaperMind 是面向学术研究的 **Multi-Agent 知识库系统**：支持 PDF 上传与异步解析、Elasticsearch 混合检索、三层记忆（Redis + ES）、基于 LangGraph 的多 Agent 编排，以及 React 前端的 SSE 流式对话。

---

## 架构概览

### 用户视角：1 个主 Agent + 4 个子 Agent

主 Agent **只做意图识别与调度**，不直接调用工具。子 Agent 按流水线串行执行，最终由 **汇总节点** 合成一条面向用户的流式回答。


| 角色 | `agent_id` | 用途 | 典型问题 |
|------|------------|------|----------|
| **主 Agent** | — | 理解意图、拆任务、输出路由计划 | 所有对话入口（自动） |
| **检索 Agent** | `retrieval-agent` | 知识库检索、证据、RAG；可接外网文献 MCP | 「找论文」「知识库里有什么」 |
| **发现 Agent** | `profile-agent` | 论文列表、单篇画像 | 「这篇讲什么」「推荐相关论文」 |
| **综述 Agent** | `summary-agent` | 基于证据写文献综述 | 「写综述」「梳理研究现状」 |
| **写作 Agent** | `writing-agent` | 润色、审稿、写作辅助 | 「润色这段」「按大纲写」 |

常见组合：**综述** ≈ 检索 → 综述；**查资料再写** ≈ 检索 → 写作。闲聊/问候走 `direct_answer`，不跑子 Agent。

### 技术分层

```
┌─ 前端 ─────────────────────────────────────────────────┐
│  React / Vite — SSE 流式对话、知识库、会话历史          │
└──────────────────────────┬───────────────────────────┘
                           │  FastAPI  /api/v1/*
┌─ Agent 编排 ─────────────▼───────────────────────────┐
│  GraphState（状态黑板）                               │
│    query / history_messages / decision /              │
│    conversation_context / agent_results /             │
│    evidence_packets                                   │
│  ContextBuilder（上下文装配器）                        │
│    → 按阶段/Agent 策略组装 LLM messages               │
│  LangGraph: main_agent → sub_agents* → summarize      │
└──────────────────────────┬───────────────────────────┘
                           │
┌─ MCP 工具层 ─────────────▼───────────────────────────┐
│  mcp_gateway → papermind-retrieval（本地 ES）         │
│              → paper-search（外网，仅检索 Agent）     │
└──────────────────────────┬───────────────────────────┘
                           │
┌─ 数据与基础设施 ─────────▼───────────────────────────┐
│  检索 / QA / 论文服务 / 三层记忆                       │
│  Kafka Worker → Docling → 切分 → Embedding → ES       │
│  MinIO · Redis · Elasticsearch · MySQL（可选）        │
└──────────────────────────────────────────────────────┘
```

**核心能力**

- **Multi-Agent 编排**：Main Agent 路由 + 子 Agent 串行执行 + 汇总合成。
- **ContextBuilder**：GraphState 存事实，ContextBuilder 在每次 LLM 调用前按 Agent 策略裁剪并组装 `messages`（不把 prompt 长期塞进 State）。
- **Tool-Calling（MCP 模式 A）**：子 Agent 经 `mcp_gateway` 调用本地/外网 MCP；工具白名单见 `mcp_server_service.py`。
- **会话记忆**：Redis 热缓存近 10 轮 + ES Episodic 全量归档；加载时 Redis 优先，缺失则 ES 回填。
- **证据传递**：检索类工具结果写入 `evidence_packets`，供综述子 Agent 与最终汇总使用。
- **PDF 流水线**：上传 → Kafka → Worker（Docling 解析、Parent-Child 切分、索引）。

---

## 上下文与记忆如何传递

### GraphState（黑板）存什么

| 字段 | 含义 |
|------|------|
| `query` | 当前用户问题（原始一句） |
| `history_messages` | 本轮之前的会话 `[{role, content}, ...]`（最多 10 条 message） |
| `conversation_context` | 路由后写入：`standalone_query`、`resolved_entities`、`context_requirements` |
| `decision` | 主 Agent JSON：目标子 Agent、各阶段任务、调度原因等 |
| `agent_results` | 已执行子 Agent 的输出列表（供下游与汇总） |
| `evidence_packets` | 检索工具返回的结构化证据 |
| `context` | 运行参数：`session_id`、`task_id`、`top_k` 等 |

**不**在 State 里持久化拼好的 `router_messages` / `summary_messages`（仅为 LLM 临时产物）。

### ContextBuilder 各阶段默认策略

| 阶段 / Agent | 会话历史条数 | standalone_query | 上游 agent_results | evidence_packets |
|--------------|-------------|------------------|-------------------|------------------|
| Router | **10** | — | — | — |
| retrieval-agent | 0 | ✅ | ❌ | ❌ |
| writing-agent | 2 | ✅ | ✅ | ❌ |
| summary-agent（子） | 0 | ❌ | ✅ | ✅ |
| profile-agent | 0 | ✅ | ❌ | ❌ |
| **最终汇总** | **4** | 展示在 Human 中 | ✅ 全部 | ✅ |

主 Agent 可在 `context_requirements` 里为各子 Agent **建议**覆盖项；实际裁剪由 `ContextBuilder` + `policies.py` 合并执行。

### 记忆读写路径（Agent 对话）

```
请求带 session_id
  → SessionStore.load_history_messages（Redis 近 10 轮，完整 content）
  → Redis 空则 ES Episodic 回填 Redis
  → 写入 state.history_messages + context.history_messages
  → 各阶段 ContextBuilder 按策略注入 messages
  → 对话结束写回 Redis + ES（user/assistant 各一条 turn）
```

Agent 路径 **不使用** MySQL `ChatHistoryService`（该模块为可选/遗留 API，默认未接入 `/agent/chat`）。

### 子 Agent 之间如何传递

串行流水线：后一个子 Agent 的 Human 消息中，在策略允许时包含 **上游 `agent_results` 全文**；检索工具结果 additionally 进入 **`evidence_packets`**。用户 SSE 仅展示 **汇总节点** 的流式输出，不展示各子 Agent 中间原文。

---

## LangGraph 工作流

```
START → main_agent_node
           │  写入 decision + conversation_context
           ├─ direct_answer → direct_answer_node → END
           └─ dispatch → execute_sub_agent_node（按 CANONICAL_AGENT_ORDER 循环）
                              │
                              ├─ 还有待执行 → 继续 execute_sub_agent_node
                              ├─ pending_task → suspend_node → END
                              └─ 完成 → summarize_node → END
```

子 Agent 固定顺序：`retrieval-agent` → `writing-agent` → `summary-agent` → `profile-agent`（仅执行 `target_agents` 中的项）。

---

## 目录结构

```
PaperMind_multiAgent/
├── app/
│   ├── main.py                      # FastAPI 入口
│   ├── core/                        # 配置、日志、通用 schema
│   ├── middleware/                  # 请求日志中间件
│   ├── observability/               # 业务事件枚举等
│   ├── agent/
│   │   ├── agents/                  # main + 4 子 Agent 定义
│   │   ├── context/                 # ContextBuilder、策略、格式化
│   │   │   ├── builder.py
│   │   │   ├── policies.py
│   │   │   ├── formatters.py
│   │   │   └── types.py
│   │   ├── graphs/
│   │   │   ├── paper_mind_graph.py  # LangGraph 装配
│   │   │   └── graph_state.py       # PaperMindState
│   │   └── tools/                   # MCP → LangChain 适配、工具注册
│   ├── mcp_gateway/                 # MCP 网关
│   ├── mcp_servers/                 # 本地 papermind-retrieval 等
│   ├── api/v1/                      # agent、sessions、papers、task、mcp
│   ├── services/
│   │   ├── chat_workflow_service.py # SSE 门面 + 图执行
│   │   ├── agent_executor_service.py
│   │   ├── summary_service.py
│   │   ├── orchestrator_service.py
│   │   ├── memory/                  # session_store、episodic、semantic
│   │   ├── storage/                 # es、redis、kafka、embedding
│   │   ├── papers/                  # 搜索、画像、索引
│   │   ├── retrieval.py / qa.py
│   │   └── ...
│   └── workers/consumer.py          # PDF 解析 Worker
├── frontend/                        # Vite + React
├── scripts/
│   ├── test_context_builder.py      # ContextBuilder 单元测试
│   └── test_memory_injection.py     # Redis/ES 记忆集成测试
├── log/                             # 按日期分目录的 JSON 日志
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## API 入口

| Endpoint | 用途 |
|----------|------|
| `GET /health` | 健康检查（ES / Redis / Kafka / MinIO / MySQL） |
| `POST /api/v1/sessions` | 创建会话 |
| `GET /api/v1/sessions` | 会话列表 |
| `GET /api/v1/sessions/{id}/history` | 恢复对话历史 |
| `DELETE /api/v1/sessions/{id}` | 删除会话 |
| `POST /api/v1/agent/chat` | Agent 对话（JSON） |
| `POST /api/v1/agent/chat/stream` | Agent SSE 流式 |
| `POST /api/v1/papers/upload` | 上传 PDF |
| `GET /api/v1/papers/{task_id}` | 任务状态 |
| `GET /api/v1/mcp/servers` | MCP Server 列表 |
| `POST /api/v1/mcp/tools/call` | 调试调用 MCP 工具 |

Swagger：`http://localhost:2222/docs`

---

## 子 Agent 工具能力（摘要）

配置：`app/services/mcp_server_service.py` → `AGENT_TOOL_ALLOWLIST`。


| 子 Agent | MCP | 能力摘要 |
|----------|-----|----------|
| retrieval-agent | 本地 + **外网** | `search_papers`、`retrieve_evidence`、`answer_with_rag`、arXiv/PubMed 等 |
| profile-agent | 本地 | `search_papers`、`get_paper_profile` |
| summary-agent | 本地 | `retrieve_evidence`、`answer_with_rag` |
| writing-agent | 本地 | `retrieve_evidence`、`answer_with_rag`（事实核查） |

外网 MCP（`paper-search`）**仅**分配给检索 Agent。

---

## Quick Start

### 1. 基础设施

```bash
docker compose up -d
```

| 服务 | 宿主机端口 | 说明 |
|------|-----------|------|
| MinIO | 9000 / 控制台 9001 | PDF 对象存储 |
| Redis | **6380** → 容器 6379 | 会话热记忆（避免与本机 6379 冲突） |
| Kafka | 9092 | 解析任务队列 |
| Elasticsearch | **9201** → 容器 9200 | 检索 + 记忆索引 |
| MySQL | 3307 → 容器 3306 | 可选持久化 |

### 2. Python 环境（推荐 conda `papermind`）

```bash
conda activate papermind
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env：REDIS_URL=redis://localhost:6380/2、DASHSCOPE_API_KEY、ES_HOSTS 等
```

### 3. 启动服务

```bash
# 后端（默认端口 2222，见 app/core/config.py）
uvicorn app.main:app --reload --host 0.0.0.0 --port 2222

# 前端
cd frontend && npm install && npm run dev

# PDF 解析 Worker（另开终端）
python -m app.workers.consumer
```

前端：`http://localhost:5173`

### 4. 自检脚本

```bash
conda activate papermind

# ContextBuilder 逻辑（无需 Redis）
python scripts/test_context_builder.py

# 记忆 Redis → ES 回填（需 Redis + ES）
python scripts/test_memory_injection.py
```

---

## 演示请求

```bash
# 创建会话
curl -X POST http://localhost:2222/api/v1/sessions

# 检索对话（替换 session_id）
curl -X POST http://localhost:2222/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"知识库里有哪些关于水下图像增强的论文？","session_id":"<sid>","top_k":5}'

# 多轮追问（依赖 session 记忆 + standalone_query）
curl -X POST http://localhost:2222/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"它的方法有什么不足？","session_id":"<sid>"}'
```

流式：

```bash
curl -N -X POST http://localhost:2222/api/v1/agent/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query":"你好","session_id":"<sid>"}'
```

---

## 日志

双通道输出（见 `app/core/logging.py`）：

- **stdout**：开发可读文本
- **`log/YYYY/MM/DD/`**：`app.json.log`、`agent.json.log`、`audit.json.log`

环境变量（`.env`）：

| 变量 | 说明 |
|------|------|
| `LOG_DIR` | 默认 `log` |
| `LOG_JSON_TO_FILE` | 是否写 JSON 文件 |
| `LOG_AGENT_BUSINESS_FILE` | 是否写 agent 业务日志 |
| `LOG_LLM_PAYLOADS` | 是否记录 LLM 请求体（调试用） |

---

## 接入新 MCP Server

1. 在 `app/mcp_gateway/server_manager.py` 注册 `SERVER_MODULES`
2. 在 `app/services/mcp_server_service.py` 增加 `MCPServerConfig` 与 `AGENT_TOOL_ALLOWLIST`
3. 重启 API；`lifespan` 会 `mcp_server_manager.refresh()` 发现工具

---

## 仓库分支


| 分支 | 说明 |
|------|------|
| `main` | 当前主开发分支（Multi-Agent + MCP + ContextBuilder） |
| `legacy-main` | 升级前单 Agent 备份，仅供对照 |

```bash
git clone https://github.com/beforgeted/PaperMind.git
cd PaperMind
# 默认 main；查看旧版：git checkout legacy-main
```

两套分支历史相互独立，请勿随意 `merge`。

---

## 相关文档

- 设计规格：`docs/superpowers/specs/2026-06-04-papermind-multi-agent-transform-design.md`
- 实施计划：`docs/superpowers/plans/2026-06-04-papermind-multi-agent-transform-plan.md`
