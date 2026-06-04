# PaperMind 后端能力迁移实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development

**Goal:** 将 PaperMind 的 ES/Redis/Kafka 基础设施、核心服务、记忆系统、API 和 Worker 迁移到当前项目

**Strategy:** 从 PaperMind 直接复制文件，只改 import 路径

---

### Batch 1: 配置 + 存储服务层

**Files:**
- Modify: `D:/Project/PaperMind_multiAgent/api/agent_api/app/core/config.py` (create unified Settings)
- Create: `D:/Project/PaperMind_multiAgent/api/agent_api/app/services/storage/es.py`
- Create: `D:/Project/PaperMind_multiAgent/api/agent_api/app/services/storage/redis.py`
- Create: `D:/Project/PaperMind_multiAgent/api/agent_api/app/services/storage/kafka.py`
- Create: `D:/Project/PaperMind_multiAgent/api/agent_api/app/services/storage/embedding.py`
- Create: `D:/Project/PaperMind_multiAgent/api/agent_api/app/services/storage/docstore.py`
- Modify: `D:/Project/PaperMind_multiAgent/api/agent_api/app/services/minio_service.py` (replace with PaperMind version)
- Create: `D:/Project/PaperMind_multiAgent/api/agent_api/app/services/storage/__init__.py`

### Batch 2: 核心服务层

- Create: `D:/Project/PaperMind_multiAgent/api/agent_api/app/services/docling.py`
- Create: `D:/Project/PaperMind_multiAgent/api/agent_api/app/services/indexing.py`
- Create: `D:/Project/PaperMind_multiAgent/api/agent_api/app/services/retrieval.py`
- Create: `D:/Project/PaperMind_multiAgent/api/agent_api/app/services/llm_client.py`
- Create: `D:/Project/PaperMind_multiAgent/api/agent_api/app/services/qa.py`
- Create: `D:/Project/PaperMind_multiAgent/api/agent_api/app/services/skills.py`
- Create: `D:/Project/PaperMind_multiAgent/api/agent_api/app/services/task_svc.py`

### Batch 3: 论文服务 + 记忆系统

- Create: `D:/Project/PaperMind_multiAgent/api/agent_api/app/services/papers/search.py`
- Create: `D:/Project/PaperMind_multiAgent/api/agent_api/app/services/p来自/profile.py`
- Create: `D:/Project/PaperMind_multiAgent/api/agent_api/app/services/papers/index.py`
- Create: `D:/Project/PaperMind_multiAgent/api/agent_api/app/services/memory/models.py`
- Create: `D:/Project/PaperMind_multiAgent/api/agent_api/app/services/memory/config.py`
- Create: `D:/Project/PaperMind_multiAgent/api/agent_api/app/services/memory/session_store.py`
- Create: `D:/Project/PaperMind_multiAgent/api/agent_api/app/services/memory/episodic.py`
- Create: `D:/Project/PaperMind_multiAgent/api/agent_api/app/services/memory/semantic.py`
- Create: `D:/Project/PaperMind_multiAgent/api/agent_api/app/services/memory/consolidation.py`
- Create: `D:/Project/PaperMind_multiAgent/api/agent_api/app/services/memory/manager.py`

### Batch 4: API + Worker + 整合

- Create: `D:/Project/PaperMind_multiAgent/api/agent_api/app/api/v1/router.py`
- Create: `D:/Project/PaperMind_multiAgent/api/agent_api/app/api/v1/_helpers.py`
- Create: `D:/Project/PaperMind_multiAgent/api/agent_api/app/api/v1/sessions.py`
- Create: `D:/Project/PaperMind_multiAgent/api/agent_api/app/api/v1/upload.py`
- Create: `D:/Project/PaperMind_multiAgent/api/agent_api/app/api/v1/task_routes.py`
- Create: `D:/Project/PaperMind_multiAgent/api/agent_api/app/workers/consumer.py`
- Modify: `D:/Project/PaperMind_multiAgent/main.py` (register new routes)
- Modify: `D:/Project/PaperMind_multiAgent/api/agent_api/app/api/chat_api.py` (add session_id)
- Modify: `D:/Project/PaperMind_multiAgent/requirements.txt` (add deps)
- Modify: `D:/Project/PaperMind_multiAgent/.env.example` (add config)

### Batch 5: Verification
- Check all imports resolve
- Verify no broken references
