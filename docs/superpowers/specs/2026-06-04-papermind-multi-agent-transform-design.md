# PaperMind Multi-Agent 改造设计

**日期：** 2026-06-04
**范围：** 第一期 — 仅 Agent 层
**策略：** 融合架构 — 保留框架，替换领域

---

## 目标

将当前药物发现多 Agent 项目改造为 PaperMind 学术论文多 Agent 项目。保留现有框架（Agent 基类、编排器、MCP Gateway、API/Service 层），删除药物发现领域实现，按 PaperMind 的学术能力重建 Agent 体系。

## Agent 架构

### 新 Agent 体系（6 个 Agent）

| Agent | 对应 PaperMind 能力 | 核心职责 |
|---|---|---|
| Main Agent | intent_router + memory_recall | 意图识别、任务拆解、分发调度 |
| Retrieval Agent | retrieval_handler | 论文搜索、证据检索、RAG 问答 |
| Writing Agent | writing_handler + Skill 系统 | 学术润色、同行评审、草稿生成 |
| Summary Agent | summary_handler | 文献综述/研究综述生成 |
| Profile Agent | profile_handler | 论文画像发现、论文级检索 |
| Chat Agent | chat_handler（统一出口） | 上下文聚合 + 最终回答生成 |

### 编排流程

```
用户 query
    │
    ▼
Main Agent ── 意图识别 + 规划
    │
    ├── retrieval 意图 → Retrieval Agent → 检索证据
    ├── writing 意图   → Writing Agent   → 润色/审稿
    ├── summary 意图   → Summary Agent   → 综述生成
    ├── profile 意图   → Profile Agent   → 论文发现
    └── chat 意图     → 直接回答
    │
    ▼
Chat Agent ── 统一合成最终回答
```

## State 设计

`DrugDiscoveryState` 重命名为 `PaperMindState`，字段对齐学术研究语义：

```python
class PaperMindState(TypedDict, total=False):
    query: str
    session_id: str
    top_k: int | None
    intent: str              # retrieval | writing | summary | profile | chat
    intent_confidence: float
    plan: dict
    selected_agents: list[str]
    agent_results: list[dict]  # [{agent_id, agent_name, task, content, sources}]
    final_answer: str
    final_sources: list[dict]
    used_tools: list[str]
```

## Graph 设计

保留 5 节点结构，语义从药物发现改为学术研究：

```
START → Main Agent（意图识别 + 选择子 Agent）
            │
            ├─ action=dispatch → Execute Sub-Agent（循环）
            │     ├─ Retrieval Agent
            │     ├─ Writing Agent
            │     ├─ Summary Agent
            │     └─ Profile Agent
            │
            └─ action=direct_answer → Chat Agent
            │
            ▼
       Chat Agent → END
```

关键文件重命名：`drug_discovery_graph.py` → `paper_mind_graph.py`

## 各 Agent 详细设计

### Main Agent（编排器）

改造自 `main_agent.py`，核心变化：system prompt 和决策 schema。

意图分类规则：
- 搜索/检索/论文查找 → retrieval
- 润色/改写/审稿/draft → writing
- 综述/summary/梳理文献 → summary
- 论文介绍/画像 → profile
- 简单问答/闲聊 → chat

### Retrieval Agent

对应 PaperMind 的 `retrieval_handler`。核心能力：论文搜索、混合检索、RAG 问答。输出论文列表、证据片段、引用标注。

### Writing Agent

对应 PaperMind 的 `writing_handler` + Skill 系统。核心能力：学术润色、同行评审、草稿生成。支持加载外部 Skill 包作为子流程。

### Summary Agent

对应 PaperMind 的 `summary_handler`。核心能力：基于检索结果的文献综述生成。输入来自上游 Retrieval Agent。

### Profile Agent

对应 PaperMind 的 `profile_handler`。核心能力：论文画像检索、论文发现。

### Chat Agent

由现有 `report_agent.py` 改造，对应 PaperMind 的 `chat_handler`。唯一面向用户的回答出口，确保风格一致、引用规范。

## 文件变更清单

### 删除（药物发现领域）

- `api/agent_api/app/agents/docking_agent.py`
- `api/agent_api/app/agents/admet_pk_agent.py`
- `api/agent_api/app/agents/md_simulation_agent.py`
- `api/agent_api/app/agents/molecule_design_agent.py`
- `api/agent_api/app/agents/synthesis_agent.py`
- `api/agent_api/app/agents/scoring_agent.py`
- `api/agent_api/app/agents/target_disease_agent.py`
- `api/agent_api/app/agents/clinical_safety_agent.py`
- `api/agent_api/app/agents/literature_agent.py`
- `api/agent_api/app/mcp_servers/biomed_tools.py`
- `api/agent_api/app/mcp_servers/tools/` 下所有药物工具
- `api/agent_api/app/graphs/drug_discovery_graph.py`

### 保留并改造（框架层）

- `base_agent.py` — 不变
- `agent_registry.py` — 更新注册列表
- `main_agent.py` — 改造 prompt 为学术意图路由
- `report_agent.py` → Chat Agent
- `prompt_common.py` — 更新通用 prompt
- `graph_state.py` → `PaperMindState`
- `paper_mind_graph.py`（新文件，替代 drug_discovery_graph）
- `mcp_gateway/` — 不变
- `services/` — 不变
- `api/` — 路由和 schema 按需调整

### 新建

- `retrieval_agent.py`
- `writing_agent.py`
- `summary_agent.py`
- `profile_agent.py`

## 不纳入第一期

- PaperMind 记忆系统（Redis SessionStore + ES Episodic + ES Semantic）
- PaperMind PDF 处理流水线（Kafka + Docling）
- PaperMind 前端
- Skill 系统的完整加载机制（Writing Agent 预留接口但不完整实现）
- 论文对比子图（作为第二期 feature）
