# PaperMind Multi-Agent

PaperMind 是一个面向学术研究的 **Multi-Agent 知识库系统**。支持 PDF 上传、异步解析、Parent-Child 切分、Elasticsearch 混合检索、三层记忆系统（Redis + ES）、以及一个基于 LangGraph 编排器的 **多 Agent 对话入口**。

## 架构概览

PaperMind 的核心目标是把"论文知识库"包装成一个可持续对话的研究助手，并通过多个专业 Agent 分工协作来提升回答质量。

**架构层次：**

```
前端体验层    React / Vite 聊天助手、知识库管理、会话记录
    │
API 接入层    FastAPI REST + SSE 流式，统一 /api/v1 前缀
    │
Agent 编排层  LangGraph StateGraph，Main Agent 意图路由 + 子 Agent 分发
    │         ├── Retrieval Agent  → 论文检索 (ES BM25+kNN+RRF)
    │         ├── Writing Agent    → 学术润色、同行评审
    │         ├── Summary Agent    → 文献综述生成
    │         └── Profile Agent    → 论文画像发现
    │
知识处理层    Docling PDF 解析 → Parent-Child 切分 → Embedding → ES 索引
    │
基础设施层    MinIO / Kafka / Elasticsearch / Redis / MySQL (可选)
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
│   │   ├── agent_executor_service.py   # Agent 执行引擎 (tool-calling)
│   │   ├── chat_workflow_service.py    # 对话工作流门面
│   │   ├── orchestrator_service.py     # 状态编排
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

| Endpoint | 用途 |
|----------|------|
| `GET /health` | 健康检查（含 ES/Redis/Kafka/MinIO/MySQL 探活） |
| `POST /api/v1/sessions` | 创建新会话 |
| `GET /api/v1/sessions` | 会话列表 |
| `DELETE /api/v1/sessions/{id}` | 删除会话 |
| `GET /api/v1/sessions/{id}/history` | 恢复会话完整对话历史 |
| `POST /api/v1/agent/chat` | **Agent 研究助手（统一问答入口）** |
| `POST /api/v1/agent/chat/stream` | Agent SSE 流式回答 |
| `POST /api/v1/papers/upload` | 上传 PDF 并投递解析任务 |
| `POST /api/v1/papers/upload/multipart/*` | 分片上传大 PDF |
| `GET /api/v1/papers` | 查看最近任务 |
| `GET /api/v1/papers/{task_id}` | 查询任务状态 |
| `DELETE /api/v1/papers/{task_id}` | 删除任务及相关存储 |
| `DELETE /api/v1/papers/batch` | 批量删除任务 |

## Agent 架构

### Multi-Agent 编排流程

```
用户 query
    │
    ▼
Main Agent ── 意图识别 + 选择子 Agent
    │
    ├── retrieval 意图 → Retrieval Agent ──调用──→ ES 混合检索
    ├── writing 意图   → Writing Agent   ──调用──→ LLM 润色/审稿
    ├── summary 意图   → Summary Agent   ──调用──→ 基于检索的综述
    ├── profile 意图   → Profile Agent   ──调用──→ 论文搜索/画像
    └── chat 意图     → 直接回答
    │
    ▼
Summarize Node ── 汇总所有子 Agent 输出 → 合成最终回答
```

### LangGraph 图结构

```
START → main_agent_node ── 条件路由 ──→ execute_sub_agent_node (循环)
                        │                      │
                        ├→ direct_answer       ├→ suspend_node → END
                        └→ summarize_node      └→ summarize_node → END
```

### Agent 与工具绑定

| Agent | 可用工具 |
|-------|---------|
| Retrieval Agent | `retrieve_evidence`, `retrieve_with_mqe`, `answer_with_rag`, `search_papers`, `deep_search_papers`, `get_paper_profile` |
| Profile Agent | `search_papers`, `deep_search_papers`, `get_paper_profile` |
| Summary Agent | `retrieve_evidence`, `answer_with_rag` |
| Writing Agent | `retrieve_evidence`, `answer_with_rag` |
| Chat Agent | `retrieve_evidence`, `answer_with_rag` |

### 检索流程

Agent 不会凭空编造论文数据。当需要事实性信息时：

1. Agent 通过 LangChain `bind_tools()` 决定调用哪个检索工具
2. 工具调用真实的 ES 后端：BM25 + kNN + RRF 混合检索
3. 检索结果以 `ToolMessage` 形式返回给 Agent
4. Agent 基于真实数据生成带引用的回答

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

| 服务 | 宿主机地址 | 说明 |
|------|-----------|------|
| MinIO | `9000` / 控制台 `9001` | PDF / Markdown 对象存储 |
| Redis | `6379` | 会话记忆 + 上传状态 |
| Kafka | `9092` | 解析任务队列 (KRaft, 无 Zookeeper) |
| Elasticsearch | `9201` → 容器 `9200` | 父/子块 + 论文索引 + 记忆 |
| MySQL | `3307` → 容器 `3306` | 聊天历史 (可选，默认启用) |

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
