# PaperMind

PaperMind 是一个面向科研论文的 Agent-first 知识库系统。当前版本支持 PDF 上传、异步解析、Parent-Child 切分、Qwen/DashScope embedding、Elasticsearch 混合检索、论文级画像检索，以及一个可自主调用 18 种工具的研究 Agent。**系统支持接入外部 Claude Code Skill 包以扩展写作与评审能力。**

## 架构概览

```mermaid
flowchart TD
    User[User] --> FastAPI[FastAPI API]
    FastAPI --> PapersAPI[Papers API]
    FastAPI --> AgentAPI[Agent Chat API]

    PapersAPI --> Upload[Upload Tasks]

    AgentAPI --> ResearchAgent[Research Agent]
    ResearchAgent --> ToolRegistry[Tool Registry]
    ToolRegistry --> RagTools[RAG Tools]
    ToolRegistry --> PaperTools[Paper Search Tools]
    ToolRegistry --> TaskTools[Task Tools]
    ToolRegistry --> MemoryTools[Memory Tools]
    ToolRegistry --> WritingTools[Writing Tools]
    ToolRegistry --> SkillTools[Skill Tools]
    WritingTools --> LLMService[LLM Service]
    SkillTools --> SkillLoader[Skill Loader]
    SkillLoader --> InstalledSkills[Installed Skills]

    RagTools --> RetrievalService[Retrieval Service]
    RagTools --> QAService[QA Service]
    PaperTools --> PaperServices[Paper Search Services]
    TaskTools --> TaskStatus[Task Status Service]
    MemoryTools --> MemoryStore[JSON Memory Store]

    Upload --> Kafka[Kafka]
    Kafka --> Worker[Worker]
    Worker --> Docling[Docling]
    Worker --> Elasticsearch[Elasticsearch]
    Worker --> MinIO[MinIO]
```

核心原则：

- **Agent-first**：所有问答请求统一走 Agent，由 Agent 自主选择工具、获取证据、组织回答。
- **Skill 扩展**：Agent 可动态加载外部 Claude Code Skill 包（nature-skills、academic-research-skills），获取学术写作、润色、评审等领域的专业规则。
- **RAG**：检索证据、RAG 问答、论文对比等能力以 Tool 形式被 Agent 调用。
- **记忆**：记忆负责稳定读写，不负责判断事实真伪。Agent 通过 `remember_fact` / `recall_memory` 工具自行管理。
- **Agent 只做调度与汇总**：理解用户目标、选择工具、组织答案，论文事实必须来自工具返回结果。

## 目录结构

```text
.
├── app/
│   ├── main.py                          # FastAPI entry + lifespan
│   ├── api/v1/
│   │   ├── router.py                    # aggregates /api/v1/*
│   │   ├── _helpers.py                  # upload helpers
│   │   ├── upload.py                    # PDF / multipart upload
│   │   └── tasks.py                     # task status & delete
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── api.py                       # POST /api/v1/agent/chat endpoint
│   │   ├── runtime.py                   # Agent lifecycle, message extraction
│   │   ├── prompts.py                   # system prompt sections
│   │   ├── session.py                   # request/session constraints
│   │   ├── registry.py                  # lazy tool registry (5 profiles)
│   │   ├── skills/                      # installed external skill packs
│   │   │   ├── nature-skills/           # nature-polishing, nature-writing
│   │   │   └── academic-research-skills/# academic-paper, academic-paper-reviewer
│   │   └── tools/
│   │       ├── contracts.py             # unified ToolResult JSON contract
│   │       ├── decorators.py            # LangChain @tool shim
│   │       ├── paper_serializers.py     # result formatting helpers
│   │       ├── rag_tools.py             # retrieve_evidence / answer_with_rag
│   │       ├── paper_search_tools.py    # paper discovery / profile tools
│   │       ├── task_tools.py            # get_task_status
│   │       ├── memory_tools.py          # session/user/project/workspace memory
│   │       ├── academic_writing_tools.py# LLM-powered review/polish/peer-review
│   │       ├── skill_tools.py           # list_skills / load_skill / use_skill_reference
│   │       └── references/              # curated writing rule extracts for tools
│   ├── services/
│   │   ├── retrieval_service.py         # BM25 + vector kNN + app-side RRF
│   │   ├── qa_service.py                # retrieve -> generate RAG chain
│   │   ├── skill_loader.py              # SKILL.md parser + skill cache
│   │   ├── vectorstore_service.py       # Elasticsearch dense_vector store
│   │   ├── memory_store.py              # JSON-backed agent memory store
│   │   └── ...                          # MinIO, Kafka, Docling, embedding, paper index
│   ├── core/                            # config, logging, Pydantic schemas
│   ├── utils/                           # chunking, paper structure parsing
│   └── workers/                         # Kafka consumer, index rebuild CLI
├── fronted/                             # Vite + React frontend
├── tests/
│   └── unit/                            # agent/tool/memory unit tests
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## API 入口

| Endpoint                                    | 用途                     |
| ------------------------------------------- | ------------------------ |
| `GET /health`                               | 健康检查                 |
| `POST /api/v1/agent/chat`                   | **Agent 研究助手（唯一问答入口）** |
| `POST /api/v1/papers/upload`                | 上传单个 PDF 并投递解析任务     |
| `POST /api/v1/papers/upload/multipart/*`    | 分片上传大 PDF               |
| `GET /api/v1/papers`                        | 查看最近任务                 |
| `GET /api/v1/papers/{task_id}`              | 查询任务状态                 |
| `DELETE /api/v1/papers/{task_id}`           | 删除任务及相关存储              |
| `DELETE /api/v1/papers/batch`               | 批量删除任务                 |

## Agent 工具一览

所有工具由 Agent 根据用户问题自动选择，用户无需手动指定。

### 检索与知识库（6 个）

| 工具                       | 功能                                     |
| -------------------------- | ---------------------------------------- |
| `retrieve_evidence`        | BM25+kNN+RRF 混合检索细粒度证据           |
| `answer_with_rag`          | 检索 + LLM 生成答案 + 引用标注            |
| `search_paper_chunks`      | retrieve_evidence 别名                    |
| `search_paper_profiles`    | 论文级画像检索（发现、筛选、推荐）         |
| `deep_search_papers`       | 论文检索 + 附带证据依据                   |
| `get_paper_profile`        | 单篇论文画像（摘要、方法、贡献等）         |

### 任务与记忆（5 个）

| 工具                       | 功能                                     |
| -------------------------- | ---------------------------------------- |
| `get_task_status`          | 查询上传/解析/入库状态                    |
| `remember_fact`            | 存储 session/user/project 三级记忆        |
| `recall_memory`            | 读取三类记忆                               |
| `update_workspace_state`   | 更新研究工作流状态                         |
| `get_workspace_state`      | 读取当前工作流状态                         |

### 写作辅助（4 个，LLM 驱动 + Skill 规则增强）

| 工具                       | 功能                                     |
| -------------------------- | ---------------------------------------- |
| `generate_review_outline`  | 生成结构化综述大纲（主题分组、空白识别）     |
| `generate_review_section`  | 按 Nature 写作标准起草综述段落             |
| `polish_academic_text`     | 按 Nature 期刊标准润色学术文本             |
| `peer_review_draft`        | 结构化同行评审（评分 + 改进建议）           |

### Skill 管理（3 个）

| 工具                       | 功能                                     |
| -------------------------- | ---------------------------------------- |
| `list_available_skills`    | 列出所有已安装的 Skill 包                 |
| `load_skill`               | 加载 Skill 的完整规则到上下文              |
| `use_skill_reference`      | 读取 Skill 中的特定参考文件                |

### Tool Profile

| Profile    | 工具数 | 适用场景                               |
| ---------- | ------ | -------------------------------------- |
| `basic`    | 6      | 检索 + 论文搜索 + 任务状态              |
| `research` | 13     | basic + 四类记忆 + skill 管理           |
| `writing`  | 12     | 检索 + 记忆 + LLM 写作 + skill 管理     |
| `workflow` | 15     | research + LLM 写作 + skill 管理        |
| `all`      | 18     | 全部工具（当前默认）                    |

## Skill 系统

PaperMind 支持接入外部 Claude Code Skill 包。已预装 4 个 Skill：

| Skill                         | 来源                  | 用途                             |
| ----------------------------- | --------------------- | -------------------------------- |
| `nature-polishing` v5.0.2     | Yuan1z0825/nature-skills | Nature 期刊标准学术润色          |
| `nature-writing` v0.2.0       | Yuan1z0825/nature-skills | Nature 风格学术写作（各章节模式） |
| `academic-paper` v3.1.2       | Imbad0202/academic-research-skills | 12-Agent 论文写作流程           |
| `academic-paper-reviewer`     | Imbad0202/academic-research-skills | 多视角同行评审                   |

**工作流示例** — 用户问"帮我写一篇综述"：

```
list_available_skills → 发现 4 个可用 Skill
load_skill("academic-paper") → 加载论文写作规则
load_skill("nature-polishing") → 加载润色规则
deep_search_papers → 找论文
retrieve_evidence → 检索细节
generate_review_outline → 按 ARS 模板生成大纲
generate_review_section → 按 Nature 标准起草
peer_review_draft → 自我评审
polish_academic_text → 最终润色
```

**安装新 Skill**：将 Skill 目录放入 `app/agent/skills/<vendor>/<skill-name>/`，重启 API 即可。Agent 会通过 `list_available_skills` 自动发现。

## Tool 返回协议

所有 Agent tools 统一返回 JSON 字符串：

```json
{
  "tool_name": "retrieve_evidence",
  "query": "frequency domain enhancement",
  "result_count": 1,
  "results": [],
  "sources": [],
  "error": null,
  "confidence": 0.8,
  "metadata": {}
}
```

## Quick Start

```bash
# 1. infra
cp .env.example .env
docker compose up -d

# MinIO console : http://localhost:9001  (papermind / papermind123)
# Elasticsearch : http://localhost:9201
# Kafka         : localhost:9092

# 2. python deps
python -m venv .venv && source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. run API
uvicorn app.main:app --reload
# http://localhost:8000/health
# http://localhost:8000/docs
```

## Windows 启动命令

```powershell
# Terminal 1: Backend API
conda activate papermind
cd D:\Project\PaperMind
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

```powershell
# Terminal 2: Kafka consumer worker
conda activate papermind
cd D:\Project\PaperMind
python -m app.workers.consumer
```

```powershell
# Terminal 3: Frontend (Vite)
cd D:\Project\PaperMind\fronted
npm run dev
```

推荐顺序：

1. 先启动基础设施：`docker compose up -d`
2. 再启动后端 API：`uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`
3. 再启动 worker：`python -m app.workers.consumer`
4. 最后启动前端：`cd fronted && npm run dev`

## 示例请求

Agent 研究助手：

```bash
curl -X POST http://localhost:8000/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"帮我找几篇和水下图像增强相关的论文，并说明依据","top_k":5}'
```

写综述时，Agent 会自动加载 Skill：

```bash
curl -X POST http://localhost:8000/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"基于知识库里的论文，帮我写一篇关于水下图像增强方法的综述"}'
```

响应示例：

```json
{
  "query": "帮我找几篇和水下图像增强相关的论文，并说明依据",
  "answer": "根据知识库检索结果，找到以下水下图像增强相关论文：\n\n1. ...",
  "contexts": [...],
  "sources": [{ "paper_id": "...", "title": "...", "score": 0.92 }],
  "used_tools": ["list_available_skills", "load_skill", "deep_search_papers", "retrieve_evidence"]
}
```

## 测试与验证

```bash
# Agent / tools / memory 单元测试
python -m unittest tests.unit.test_agent_first_architecture

# Python 语法检查
python -m compileall app tests -q
```

## 当前状态

已完成：

- PDF 上传、分片上传、Kafka 异步解析、MinIO 存储
- Docling 解析、Parent-Child chunking
- Elasticsearch `dense_vector` 向量检索
- BM25 + kNN + 应用层 RRF
- 论文画像抽取（LLM + 规则兜底）、索引与论文级检索
- **Agent-first 统一问答入口**（18 个工具、5 种 profile）
- session / user / project 三类 JSON 记忆 + workspace 状态
- **LLM 驱动的学术写作工具**（综述大纲、段落生成、学术润色、自我评审）
- **外部 Skill 包接入系统**（预装 nature-skills + academic-research-skills，共 4 个 Skill）

暂未实现：

- 选定论文生成综述的完整 workflow
- Agent 流式回答
- 前端对 Agent 工具调用过程 / Skill 加载状态的可视化
