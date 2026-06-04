# PaperMind 后端能力迁移设计

**日期：** 2026-06-04
**范围：** 第二期 — 后端基础设施 + 核心服务 + API + Worker
**策略：** 完整接入 PaperMind 的 ES/Redis/Kafka 基础设施和服务代码

---

## 目标

将 PaperMind 的后端能力（PDF 处理流水线、混合检索、多层记忆系统、论文服务、会话 API）完整迁移到当前项目。所有核心服务作为 Agent 可调用的工具，不独立暴露为问答出口。

## 架构约束

所有检索、论文搜索、记忆操作必须通过 Agent 编排调用，不允许绕过 Agent 直接返回答案：

```
用户 Query → Main Agent（意图路由）→ 子 Agent → 调用后端服务 → Chat Agent 合成回答
```

## 迁移模块

### Layer 1 — 配置扩展

合并两套配置为统一的 `app/core/config.py`，新增字段：
- ES: hosts, indices (parent/child/papers/memories)
- Redis: connection URL, TTL
- Kafka: bootstrap servers, topic, group
- Embedding: backend, model, API key
- Chunking: parent/child sizes, overlaps
- Retrieval: kNN/BM25/RRF 参数
- Memory: indices, TTL, limits, consolidation

### Layer 2 — 存储服务

从 PaperMind 复制 `app/services/storage/` 全部文件，只改 import 路径：
- `es.py` — ES 客户端 + 索引管理
- `redis.py` — Redis 客户端封装
- `kafka.py` — Kafka Producer 单例
- `embedding.py` — DashScope Embedding
- `docstore.py` — 文档存储抽象
- `minio.py` — 替换现有 minio_service.py

### Layer 3 — 核心服务

从 PaperMind 复制，全部作为 Agent 的底层调用能力：
- `docling.py` — PDF → Markdown 解析
- `indexing.py` — 父-子块切分 + ES 索引
- `retrieval.py` — BM25 + kNN + RRF 混合检索
- `llm.py` — LLM 工厂函数（补充现有 llm_service）
- `qa.py` — RAG 问答链
- `skills.py` — Skill 加载管理
- `tasks.py` — 任务状态服务
- `papers/search.py` — 论文级搜索
- `papers/profile.py` — 论文画像抽取
- `papers/index.py` — 论文元数据索引
- `memory/` — 7 个文件，三层记忆系统

### Layer 4 — API 层

从 PaperMind 复制路由：
- `sessions.py` — 会话 CRUD（POST/GET/DELETE）
- `upload.py` — PDF 上传 + 分片上传
- `tasks.py` — 任务状态查询
- `_helpers.py` — 上传辅助
- `router.py` — 路由聚合

现有 `chat_api.py` 增加 session_id 支持多轮对话。

### Layer 5 — Worker

- `workers/consumer.py` — Kafka 消费 → Docling 解析 → 索引流水线

## 文件变更

### 新建/复制（~30 个文件）
```
app/core/config.py              (扩展)
app/services/storage/es.py
app/services/storage/redis.py
app/services/storage/kafka.py
app/services/storage/embedding.py
app/services/storage/docstore.py
app/services/storage/minio.py   (替换)
app/services/docling.py
app/services/indexing.py
app/services/retrieval.py
app/services/llm_client.py
app/services/qa.py
app/services/skills.py
app/services/task_svc.py
app/services/papers/search.py
app/services/papers/profile.py
app/services/papers/index.py
app/services/memory/models.py
app/services/memory/config.py
app/services/memory/session_store.py
app/services/memory/episodic.py
app/services/memory/semantic.py
app/services/memory/consolidation.py
app/services/memory/manager.py
app/api/v1/router.py
app/api/v1/_helpers.py
app/api/v1/sessions.py
app/api/v1/upload.py
app/api/v1/task_routes.py
app/workers/consumer.py
```

### 修改（~5 个文件）
```
main.py                    — 注册新路由
chat_api.py                — 增加 session_id
requirements.txt           — 新依赖
docker-compose.yml         — ES/Redis/Kafka 服务
.env.example               — 新配置项
```

## 不纳入本期
- 前端任何改动
- Worker 的实际部署和运行（仅迁移代码）
- ES/Redis/Kafka 的实际运维配置
