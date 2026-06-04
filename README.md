# PaperMind Multi-Agent

PaperMind 是一个面向学术研究的 **Multi-Agent 知识库系统**。支持 PDF 上传、异步解析、Parent-Child 切分、Elasticsearch 混合检索、三层记忆系统（Redis + ES）、以及一个基于 LangGraph 编排器的 **多 Agent 对话入口**。

## 架构概览

PaperMind 的核心目标是把"论文知识库"包装成一个可持续对话的研究助手，并通过多个专业 Agent 分工协作来提升回答质量。

### 子 Agent 分工（用户视角）

系统由 **1 个主 Agent + 4 个专业子 Agent** 协作；主 Agent 只做意图识别与调度，**不直接调工具**。子 Agent 按任务类型选用，最终由 **汇总节点** 合成一条面向用户的回答。


| 子 Agent        | `agent_id`        | 你能用它做什么                                   | 典型触发场景                  |
| -------------- | ----------------- | ----------------------------------------- | ----------------------- |
| **主 Agent**    | —（编排器）            | 理解问题、拆任务、决定派谁                             | 所有对话入口（自动运行，用户无感）       |
| **检索 Agent**   | `retrieval-agent` | 查知识库、找证据、RAG 问答；**唯一**可调外网 arXiv/PubMed 等 | 「找论文」「知识库里有什么」「某主题相关片段」 |
| **论文发现 Agent** | `profile-agent`   | 论文列表推荐、单篇画像（方法/贡献/摘要）                     | 「有哪些相关论文」「这篇论文讲了什么」     |
| **综述 Agent**   | `summary-agent`   | 基于检索结果写文献综述、梳理研究脉络                        | 「写综述」「总结研究现状」「领域进展」     |
| **写作 Agent**   | `writing-agent`   | 学术润色、同行评审、草稿与写作辅助                         | 「润色这段」「审稿」「按大纲写一段」      |


常见组合：**综述** ≈ 检索 Agent → 综述 Agent；**写作前查资料** ≈ 检索 Agent → 写作 Agent。简单闲聊由主 Agent **直接回答**，不经过子 Agent 工具链。

### 系统分层（技术视角）

```
┌─ 前端体验层 ─────────────────────────────────────────────┐
│  React / Vite — SSE 流式对话、知识库、会话历史             │
└────────────────────────────┬─────────────────────────────┘
                             │ FastAPI (/api/v1/agent, sessions, papers, mcp)
┌─ Agent 编排层 ─────────────▼─────────────────────────────┐
│  Main Agent（意图路由）                                    │
│    ├─ Retrieval Agent  → 论文检索、证据、RAG、外网文献      │
│    ├─ Profile Agent    → 论文列表 / 单篇画像               │
│    ├─ Summary Agent    → 文献综述                          │
│    └─ Writing Agent    → 润色 / 审稿 / 写作辅助             │
│  → summarize_node 汇总 → 一条 SSE 流式回答                  │
└────────────────────────────┬─────────────────────────────┘
                             │ 
┌─ MCP 工具层 ───────────────▼─────────────────────────────┐
│  mcp_gateway → papermind-retrieval（本地 ES 检索）        │
│              → paper-search（外网，仅 Retrieval Agent）   │
└────────────────────────────┬─────────────────────────────┘
                             │
┌─ 数据与基础设施 ───────────▼─────────────────────────────┐
│  领域服务：retrieval / qa / papers / 三层记忆               │
│  入库流水线：Kafka Worker → Docling → 切分 → ES            │
│  MinIO · Redis · Elasticsearch · MySQL(可选)              │
└──────────────────────────────────────────────────────────┘
```

**核心亮点：**

- **Multi-Agent 编排**: Main Agent 负责意图识别与任务拆解，4 个子 Agent 分工执行（检索、写作、综述、画像发现），最终由汇总节点合成统一回答。
- **Tool-Calling 检索（模式 A / MCP）**: Agent 工具由 MCP Server 暴露，经 `mcp_gateway` 统一调用；本地 `papermind-retrieval` + 可选外部 `paper-search` MCP，便于后续接入第三方 MCP。
- **文件上传与异步解析**: PDF 上传后进入 Kafka + Worker 的异步解析流水线，Docling 解析、切分、embedding 与索引由 Worker 后台完成。
- **三层记忆系统**: Redis SessionStore 管理热会话滑动窗口；ES Episodic 永久存档完整对话；ES Semantic 存储长期知识记忆。
- **前后端分离**: React / Vite 前端提供流式对话、会话切换、知识库管理和引用来源查看。

## 目录结构

```
PaperMind_multiAgent/
├── app/
│   ├── main.py                         # FastAPI 入口
│   ├── core/
│   │   ├── config.py                   # 统一配置
│   │   ├── logging.py                  # 日志配置
│   │   └── schemas.py                  # 通用数据模型
│   ├── agent/                          # Agent 系统
│   │   ├── agents/                     # 6 个 Agent 定义
│   │   │   ├── main_agent.py           # 意图路由编排器
│   │   │   ├── retrieval_agent.py      # 论文检索 Agent
│   │   │   ├── writing_agent.py        # 学术写作 Agent
│   │   │   ├── summary_agent.py        # 文献综述 Agent
│   │   │   ├── profile_agent.py        # 论文发现 Agent
│   │   │   └── chat_agent.py           # 统一回答合成 Agent
│   │   ├── graphs/                     # LangGraph 工作流
│   │   │   ├── paper_mind_graph.py     # StateGraph 装配
│   │   │   └── graph_state.py          # AgentState TypedDict
│   │   ├── schemas/                    # Agent 专用 schema
│   │   └── tools/                      # Agent 可调用的工具
│   │       ├── mcp_adapter.py          # MCP → LangChain 工具适配
│   │       └── registry.py             # Agent 工具白名单（模式 A）
│   ├── mcp_servers/
│   │   └── papermind_retrieval/        # 本地检索 MCP Server
│   ├── mcp_gateway/                    # MCP 网关（鉴权/路由/执行）
│   ├── api/
│   │   ├── v1/
│   │   │   ├── router.py               # /api/v1 路由聚合
│   │   │   ├── agent_routes.py         # Agent 对话 (chat + SSE)
│   │   │   ├── sessions.py             # 会话 CRUD
│   │   │   ├── upload.py               # PDF 上传 (含分片)
│   │   │   └── task_routes.py          # 任务状态查询
│   │   ├── chat_api.py                 # 聊天接口
│   │   ├── task_api.py                 # 任务管理
│   │   └── workflow_api.py             # 工作流管理
│   ├── services/
│   │   ├── storage/                    # 基础设施封装
│   │   │   ├── es.py                   # Elasticsearch 客户端
│   │   │   ├── redis.py                # Redis 客户端
│   │   │   ├── kafka.py                # Kafka 生产者
│   │   │   ├── embedding.py            # Embedding 后端
│   │   │   └── docstore.py             # ES 文档存储
│   │   ├── memory/                     # 三层记忆系统
│   │   │   ├── session_store.py        # Redis 会话记忆
│   │   │   ├── episodic.py             # ES 情景记忆
│   │   │   ├── semantic.py             # ES 语义记忆
│   │   │   ├── consolidation.py        # 记忆固化
│   │   │   └── manager.py              # 统一编排
│   │   ├── papers/                     # 论文服务
│   │   │   ├── search.py               # 论文搜索
│   │   │   ├── profile.py              # 论文画像
│   │   │   └── index.py                # 论文索引
│   │   ├── retrieval.py                # 混合检索 BM25+kNN+RRF
│   │   ├── qa.py                       # RAG 问答链
│   │   ├── docling.py                  # PDF 解析
│   │   ├── indexing.py                 # 切分与索引
│   │   ├── agent_executor_service.py   # 子 Agent 执行 (tool-calling)
│   │   ├── chat_workflow_service.py    # 对话工作流 + SSE 门面
│   │   ├── mcp_server_service.py       # MCP Server 目录 + Agent 工具白名单
│   │   ├── mcp_bridge.py               # MCP 桥接 (兼容层)
│   │   ├── orchestrator_service.py     # Run / 异步任务编排
│   │   └── llm_service.py              # LLM 客户端工厂
│   ├── shared/                         # 工具函数
│   └── workers/
│       └── consumer.py                 # Kafka 消费 Worker
├── frontend/                           # Vite + React 前端
│   └── src/
│       ├── api/papers.ts               # 所有 API 调用封装
│       ├── types/api.ts                # TypeScript 类型定义
│       ├── components/
│       │   ├── ChatPanel.tsx           # 聊天助手面板
│       │   ├── KnowledgeBasePanel.tsx  # 知识库管理
│       │   ├── RecordsPanel.tsx        # 聊天记录面板
│       │   └── SourcesPanel.tsx        # 引用来源侧栏
│       └── utils/                      # 工具函数
├── docker-compose.yml                  # 本地基础设施
├── requirements.txt
└── .env.example
```

## API 入口


| Endpoint                                 | 用途                                    |
| ---------------------------------------- | ------------------------------------- |
| `GET /health`                            | 健康检查（含 ES/Redis/Kafka/MinIO/MySQL 探活） |
| `POST /api/v1/sessions`                  | 创建新会话                                 |
| `GET /api/v1/sessions`                   | 会话列表                                  |
| `DELETE /api/v1/sessions/{id}`           | 删除会话                                  |
| `GET /api/v1/sessions/{id}/history`      | 恢复会话完整对话历史                            |
| `POST /api/v1/agent/chat`                | **Agent 研究助手（统一问答入口）**                |
| `POST /api/v1/agent/chat/stream`         | Agent SSE 流式回答                        |
| `POST /api/v1/papers/upload`             | 上传 PDF 并投递解析任务                        |
| `POST /api/v1/papers/upload/multipart/`* | 分片上传大 PDF                             |
| `GET /api/v1/papers`                     | 查看最近任务                                |
| `GET /api/v1/papers/{task_id}`           | 查询任务状态                                |
| `DELETE /api/v1/papers/{task_id}`        | 删除任务及相关存储                             |
| `DELETE /api/v1/papers/batch`            | 批量删除任务                                |
| `GET /api/v1/mcp/servers`                | 列出已配置的 MCP Server                     |
| `GET /api/v1/mcp/tools`                  | 列出已发现的 MCP 工具                         |
| `POST /api/v1/mcp/tools/call`            | 经 Gateway 调用任意 MCP 工具（调试/集成）          |


## Agent 架构

### 一次对话里发生了什么

```
用户提问
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ Main Agent：判断意图，生成执行计划（派哪些子 Agent、顺序）   │
└────────────────────────────┬────────────────────────────┘
                             │
     ┌───────────────────────┼───────────────────────┐
     ▼                       ▼                       ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    …
│ 检索 Agent   │    │ 发现 Agent   │    │ 综述 Agent   │
│ 找论文/证据  │    │ 列表+画像    │    │ 写综述       │
│ (+外网可选)  │    │              │    │              │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                  │
       └──────────────────┴──────────────────┘
                             │
                             ▼
              summarize_node：合并各阶段结果
                             │
                             ▼
              用户看到一条流式最终回答（SSE）
```

- **检索 Agent**：知识库混合检索（BM25 + 向量 + RRF）、RAG 问答；需要时调用外网 MCP（arXiv / PubMed 等）。
- **发现 Agent**：`search_papers` / `get_paper_profile`，偏「有哪些论文、某篇讲什么」。
- **综述 Agent**：在上游检索证据基础上组织综述结构；证据不足时可自行补检索。
- **写作 Agent**：润色与审稿为主；需要事实依据时再调检索类工具。
- **直接回答**：问候、闲聊等与学术任务无关时，主 Agent 不派子 Agent。

### LangGraph 图结构

```
START → main_agent_node
            │
            ├─ dispatch → execute_sub_agent_node（按 selected_agents 循环）
            │                  │
            │                  ├─ 还有待执行 Agent → 继续 execute_sub_agent_node
            │                  ├─ suspend → suspend_node → END
            │                  └─ 完成 → summarize_node → END
            │
            ├─ direct_answer → direct_answer_node → END
            └─ summarize → summarize_node → END
```

### 子 Agent 与工具能力对照

下表说明各子 Agent **实际能调用的 MCP 工具**（模式 A：经 `mcp_gateway` 统一执行）。配置入口：`app/services/mcp_server_service.py` → `AGENT_TOOL_ALLOWLIST`。


| 子 Agent      | 主要职责（给用户）            | MCP Server  | 工具能力摘要                                                                                                                                                                                                                 |
| ------------ | -------------------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **检索 Agent** | 知识库检索、证据片段、RAG；外网查文献 | 本地 + **外网** | 本地：`retrieve_evidence`、`retrieve_with_mqe`、`answer_with_rag`、`search_papers`、`deep_search_papers`、`get_paper_profile`；外网：`search_arxiv`、`search_pubmed`、`search_biorxiv`、`search_google_scholar`、`download_`*、`read_*` |
| **发现 Agent** | 推荐论文列表、单篇结构化画像       | 仅本地         | `search_papers`、`deep_search_papers`、`get_paper_profile`                                                                                                                                                               |
| **综述 Agent** | 文献综述与领域梳理            | 仅本地         | `retrieve_evidence`、`answer_with_rag`                                                                                                                                                                                  |
| **写作 Agent** | 润色、审稿、写作辅助           | 仅本地         | `retrieve_evidence`、`answer_with_rag`（作事实核查）                                                                                                                                                                           |
| **主 Agent**  | 路由与拆任务               | —           | 不绑定工具                                                                                                                                                                                                                  |


> **外网 MCP**（`paper-search`）只分配给 **检索 Agent**，其它子 Agent 仅访问已上传并索引到 ES 的知识库。

**工具调用路径（实现细节）：** 子 Agent `bind_tools` → `mcp_adapter` → `mcp_gateway` → `papermind-retrieval` / `paper-search` → 领域服务 / 外部 API → 结果回填 LLM。

**事实性约束：** 涉及论文数据时必须走工具，禁止编造；汇总节点只向用户推送**一条**最终流式回答，避免检索阶段与汇总阶段重复刷屏。

### 接入新的 MCP Server

1. 在 `app/mcp_gateway/server_manager.py` 的 `SERVER_MODULES` 注册模块路径。
2. 在 `app/services/mcp_server_service.py` 增加 `MCPServerConfig` 与 `AGENT_TOOL_ALLOWLIST`。
3. 重启服务；`lifespan` 会 `mcp_server_manager.refresh()` 自动发现工具。

## 演示请求

```bash
# 1. 创建新会话
curl -X POST http://localhost:2222/api/v1/sessions
# → {"session_id": "abc123", ...}

# 2. 多 Agent 检索对话
curl -X POST http://localhost:2222/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"帮我找几篇关于水下图像增强的论文，并说明依据","session_id":"abc123","top_k":5}'

# 3. 多轮追问
curl -X POST http://localhost:2222/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"第一篇的方法是什么？","session_id":"abc123"}'

# 4. 学术润色
curl -X POST http://localhost:2222/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"请帮我润色以下学术文本：...","session_id":"abc123"}'
```

响应示例：

```json
{
  "query": "帮我找几篇关于水下图像增强的论文",
  "answer": "根据知识库检索结果，找到以下相关论文：\n\n1. ...",
  "contexts": [...],
  "sources": [{ "paper_id": "...", "title": "...", "score": 0.92 }],
  "used_tools": ["search_papers", "retrieve_evidence"]
}
```

## 基础设施


| 服务            | 宿主机地址               | 说明                          |
| ------------- | ------------------- | --------------------------- |
| MinIO         | `9000` / 控制台 `9001` | PDF / Markdown 对象存储         |
| Redis         | `6379`              | 会话记忆 + 上传状态                 |
| Kafka         | `9092`              | 解析任务队列 (KRaft, 无 Zookeeper) |
| Elasticsearch | `9201` → 容器 `9200`  | 父/子块 + 论文索引 + 记忆            |
| MySQL         | `3307` → 容器 `3306`  | 聊天历史 (可选，默认启用)              |


## Quick Start

```bash
# 1. 启动基础设施
docker compose up -d

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env：填入 DASHSCOPE_API_KEY、OPENAI_API_KEY 等

# 4. 启动后端
uvicorn app.main:app --reload --host 0.0.0.0 --port 2222

# 5. 启动前端 (新终端)
cd frontend && npm install && npm run dev

# 6. 启动 Worker (新终端，用于 PDF 解析)
python -m app.workers.consumer
```

访问 `http://localhost:5173` 进入前端界面。

## 仓库分支说明

本仓库在 GitHub 上可能同时存在以下分支，用途不同：


| 分支            | 说明                                                                 |
| ------------- | ------------------------------------------------------------------ |
| `main`        | **当前主开发分支**，包含 Multi-Agent 编排、MCP 模式 A、前端流式对话等最新代码。克隆仓库后默认即为此分支。   |
| `legacy-main` | **历史备份分支**，保留升级前的单 Agent 版 PaperMind 代码与提交历史，仅供对照或回查，**不是**日常开发起点。 |


### 克隆与切换

```bash
# 默认克隆 main（推荐）
git clone https://github.com/beforgeted/PaperMind.git
cd PaperMind

# 如需查看旧版单 Agent 代码
git fetch origin
git checkout legacy-main
```

### 注意事项

- `git clone` 只会检出默认分支（`main`），不会自动下载 `legacy-main` 的工作区内容；但在 GitHub 网页的 **Branches** 列表中仍能看到 `legacy-main`。
- 两套分支的 Git 历史相互独立（无共同祖先），请勿对 `main` 执行与 `legacy-main` 的普通 `merge`，除非明确要做历史合并。
- 新贡献者请以 `main` 为准阅读 README 与提交 PR。

