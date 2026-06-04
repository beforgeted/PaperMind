"""AgentRun 状态存储 — MySQL 或内存回退。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from app.core.config import settings
from app.schemas.runtime_schema import AgentRunRecord, WorkflowStepRecord


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except TypeError:
        return str(value)


def _json_dump(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False, default=str)


def _json_load(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return default


class AgentRunService:
    """AgentRun 状态存储，MySQL 优先，内存回退。"""

    def __init__(self):
        self._initialized = False
        # In-memory fallback
        self._runs: Dict[str, AgentRunRecord] = {}
        self._steps: Dict[str, List[WorkflowStepRecord]] = {}

    @property
    def _use_mysql(self) -> bool:
        return bool(settings.mysql_enabled)

    async def _ensure_initialized(self):
        if self._initialized:
            return
        if self._use_mysql:
            from app.services.mysql_service import mysql_service
            await self._init_mysql(mysql_service)
        self._initialized = True

    async def _init_mysql(self, mysql_service):
        await mysql_service.require_engine()
        await mysql_service.execute(
            """CREATE TABLE IF NOT EXISTS agent_runs (
                run_id VARCHAR(80) PRIMARY KEY, query LONGTEXT NOT NULL, status VARCHAR(32) NOT NULL,
                decision JSON NULL, selected_agents JSON NULL, context JSON NULL,
                agent_results JSON NULL, pending_task_id VARCHAR(80) NULL,
                final_report LONGTEXT NULL, error TEXT NULL,
                created_at VARCHAR(40) NOT NULL, updated_at VARCHAR(40) NOT NULL,
                KEY idx_agent_runs_status_updated (status, updated_at),
                KEY idx_agent_runs_pending_task (pending_task_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
        await mysql_service.execute(
            """CREATE TABLE IF NOT EXISTS workflow_steps (
                step_id VARCHAR(80) PRIMARY KEY, run_id VARCHAR(80) NOT NULL,
                step_index INT NOT NULL, node_name VARCHAR(80) NOT NULL,
                title VARCHAR(255) NOT NULL, status VARCHAR(32) NOT NULL,
                agent_id VARCHAR(80) NULL, agent_name VARCHAR(255) NULL,
                task LONGTEXT NULL, result JSON NULL, task_id VARCHAR(80) NULL,
                error TEXT NULL, started_at VARCHAR(40) NULL,
                completed_at VARCHAR(40) NULL, updated_at VARCHAR(40) NOT NULL,
                KEY idx_workflow_steps_run_index (run_id, step_index),
                KEY idx_workflow_steps_agent (run_id, agent_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")

    # ── Public API ──────────────────────────────────────────────

    async def create_run(self, *, query: str, context: Optional[Dict[str, Any]] = None) -> AgentRunRecord:
        await self._ensure_initialized()
        now = utc_now_iso()
        run = AgentRunRecord(
            run_id=f"run_{uuid4().hex}",
            query=query, status="created",
            context=dict(context or {}),
            created_at=now, updated_at=now,
        )
        if self._use_mysql:
            from app.services.mysql_service import mysql_service
            await mysql_service.execute(
                """INSERT INTO agent_runs (run_id, query, status, decision, selected_agents, context,
                   agent_results, pending_task_id, final_report, error, created_at, updated_at)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (run.run_id, run.query, run.status, _json_dump(run.decision),
                 _json_dump(run.selected_agents), _json_dump(run.context),
                 _json_dump(run.agent_results), run.pending_task_id, run.final_report,
                 run.error, run.created_at, run.updated_at))
        else:
            self._runs[run.run_id] = run
            self._steps[run.run_id] = []
        return run

    async def get_run(self, run_id: str) -> Optional[AgentRunRecord]:
        await self._ensure_initialized()
        if self._use_mysql:
            from app.services.mysql_service import mysql_service
            row = await mysql_service.fetchone("SELECT * FROM agent_runs WHERE run_id = %s", (run_id,))
            if not row:
                return None
            steps = await self._load_mysql_steps(mysql_service, run_id)
            return self._row_to_run(row, steps)
        return self._runs.get(run_id)

    async def list_runs(self, *, limit: int = 50) -> List[AgentRunRecord]:
        await self._ensure_initialized()
        if self._use_mysql:
            from app.services.mysql_service import mysql_service
            rows = await mysql_service.fetchall(
                "SELECT * FROM agent_runs ORDER BY created_at DESC LIMIT %s",
                (max(1, limit),),
            )
            runs: List[AgentRunRecord] = []
            for row in rows:
                steps = await self._load_mysql_steps(mysql_service, str(row["run_id"]))
                runs.append(self._row_to_run(row, steps))
            return runs
        # In-memory fallback: sorted by created_at descending
        sorted_runs = sorted(
            self._runs.values(),
            key=lambda r: r.created_at or "",
            reverse=True,
        )
        return sorted_runs[:max(1, limit)]

    async def mark_running(self, run_id: str) -> Optional[AgentRunRecord]:
        run = await self.get_run(run_id)
        if not run: return None
        run.status = "running"
        self._touch(run)
        return await self._save_run(run)

    async def set_decision(self, *, run_id: str, decision: Dict[str, Any],
                           selected_agents: List[str], agents: List[Dict[str, Any]]) -> Optional[AgentRunRecord]:
        run = await self.get_run(run_id)
        if not run: return None
        run.status = "running"
        run.decision = dict(decision)
        run.selected_agents = list(selected_agents)
        if not run.steps:
            run.steps = self._build_steps(run_id=run_id, selected_agents=selected_agents, agents=agents)
        self._touch(run)
        return await self._save_run(run)

    async def complete_step(self, *, run_id: str, node_name: str,
                            result: Optional[Dict[str, Any]] = None) -> Optional[AgentRunRecord]:
        run = await self.get_run(run_id)
        if not run: return None
        step = self._find_step(run, node_name=node_name)
        if step:
            step.status = "completed"
            step.result = dict(result or {})
            step.completed_at = utc_now_iso()
            step.updated_at = step.completed_at
        self._touch(run)
        return await self._save_run(run)

    async def complete_agent_step(self, *, run_id: str, agent_id: str,
                                   result: Dict[str, Any], task_id: Optional[str] = None) -> Optional[AgentRunRecord]:
        run = await self.get_run(run_id)
        if not run: return None
        step = self._find_step(run, agent_id=agent_id)
        if step:
            step.status = "suspended" if task_id else "completed"
            step.task = str(result.get("task") or step.task or "")
            step.result = dict(result)
            step.task_id = task_id
            step.completed_at = None if task_id else utc_now_iso()
            step.updated_at = utc_now_iso()
        run.agent_results = list(self._upsert_agent_result(run.agent_results, result))
        if task_id:
            run.status = "suspended"
            run.pending_task_id = task_id
        self._touch(run)
        return await self._save_run(run)

    async def suspend_run(self, *, run_id: str, task_id: str, step_id: str) -> Optional[AgentRunRecord]:
        run = await self.get_run(run_id)
        if not run: return None
        run.status = "suspended"
        run.pending_task_id = task_id
        for step in run.steps:
            if step.step_id == step_id:
                step.status = "suspended"
                step.task_id = task_id
                step.updated_at = utc_now_iso(); break
        self._touch(run)
        return await self._save_run(run)

    async def resume_from_task_result(self, *, run_id: str, task_id: str,
                                       result: Dict[str, Any]) -> Optional[AgentRunRecord]:
        run = await self.get_run(run_id)
        if not run: return None
        for step in run.steps:
            if step.task_id == task_id:
                step.status = "completed"
                step.completed_at = utc_now_iso()
                step.updated_at = step.completed_at
                step.result = {**step.result, "external_task": {
                    "task_id": task_id, "status": "succeeded", "result": result}}
        run.status = "running"
        run.pending_task_id = None
        self._touch(run)
        return await self._save_run(run)

    async def complete_run(self, *, run_id: str, final_report: str,
                            agent_results: Optional[List[Dict[str, Any]]] = None) -> Optional[AgentRunRecord]:
        run = await self.get_run(run_id)
        if not run: return None
        run.status = "completed"
        run.pending_task_id = None
        run.final_report = final_report
        if agent_results is not None:
            run.agent_results = list(agent_results)
        self._touch(run)
        return await self._save_run(run)

    async def fail_run(self, *, run_id: str, error: str) -> Optional[AgentRunRecord]:
        run = await self.get_run(run_id)
        if not run: return None
        run.status = "failed"
        run.error = error
        self._touch(run)
        return await self._save_run(run)

    async def fail_step(self, *, run_id: str, node_name: str, error: str) -> Optional[AgentRunRecord]:
        run = await self.get_run(run_id)
        if not run: return None
        step = self._find_step(run, node_name=node_name)
        if step:
            step.status = "failed"; step.error = error
            step.completed_at = utc_now_iso(); step.updated_at = step.completed_at
        run.status = "failed"; run.error = error
        self._touch(run)
        return await self._save_run(run)

    async def save_checkpoint(self, *, run_id: str, node_name: str,
                               state: Dict[str, Any], node_update: Dict[str, Any]) -> None:
        await self._ensure_initialized()
        if self._use_mysql:
            try:
                from app.services.mysql_service import mysql_service
                await mysql_service.execute(
                    """INSERT INTO workflow_checkpoints (checkpoint_id, run_id, node_name, state, node_update, created_at)
                       VALUES (%s,%s,%s,%s,%s,%s)""",
                    (f"ckpt_{uuid4().hex}", run_id, node_name, _json_dump(state), _json_dump(node_update), utc_now_iso()))
            except Exception:
                pass  # Checkpoint table may not exist — non-critical

    # ── Internal helpers ────────────────────────────────────────

    async def _save_run(self, run: AgentRunRecord) -> AgentRunRecord:
        if self._use_mysql:
            from app.services.mysql_service import mysql_service
            await mysql_service.execute(
                """UPDATE agent_runs SET query=%s, status=%s, decision=%s, selected_agents=%s,
                   context=%s, agent_results=%s, pending_task_id=%s, final_report=%s,
                   error=%s, updated_at=%s WHERE run_id=%s""",
                (run.query, run.status, _json_dump(run.decision), _json_dump(run.selected_agents),
                 _json_dump(run.context), _json_dump(run.agent_results), run.pending_task_id,
                 run.final_report, run.error, run.updated_at, run.run_id))
            await self._upsert_mysql_steps(mysql_service, run.steps)
        else:
            self._runs[run.run_id] = run
            self._steps[run.run_id] = list(run.steps)
        return run

    async def _load_mysql_steps(self, mysql_service, run_id: str) -> List[WorkflowStepRecord]:
        rows = await mysql_service.fetchall(
            "SELECT * FROM workflow_steps WHERE run_id = %s ORDER BY step_index ASC", (run_id,))
        return [self._row_to_step(row) for row in rows]

    async def _upsert_mysql_steps(self, mysql_service, steps: List[WorkflowStepRecord]) -> None:
        await mysql_service.executemany(
            """INSERT INTO workflow_steps (step_id, run_id, step_index, node_name, title, status,
               agent_id, agent_name, task, result, task_id, error, started_at, completed_at, updated_at)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
               ON DUPLICATE KEY UPDATE step_index=VALUES(step_index), node_name=VALUES(node_name),
               title=VALUES(title), status=VALUES(status), agent_id=VALUES(agent_id),
               agent_name=VALUES(agent_name), task=VALUES(task), result=VALUES(result),
               task_id=VALUES(task_id), error=VALUES(error), started_at=VALUES(started_at),
               completed_at=VALUES(completed_at), updated_at=VALUES(updated_at)""",
            [(s.step_id, s.run_id, s.index, s.node_name, s.title, s.status, s.agent_id,
              s.agent_name, s.task, _json_dump(s.result), s.task_id, s.error,
              s.started_at, s.completed_at, s.updated_at) for s in steps])

    def _row_to_run(self, row: Dict[str, Any], steps: List[WorkflowStepRecord]) -> AgentRunRecord:
        return AgentRunRecord(
            run_id=str(row["run_id"]), query=str(row.get("query") or ""),
            status=str(row.get("status") or "created"),
            decision=dict(_json_load(row.get("decision"), {})),
            selected_agents=list(_json_load(row.get("selected_agents"), [])),
            steps=steps, context=dict(_json_load(row.get("context"), {})),
            agent_results=list(_json_load(row.get("agent_results"), [])),
            pending_task_id=row.get("pending_task_id"),
            final_report=str(row.get("final_report") or ""),
            error=str(row.get("error") or ""),
            created_at=str(row.get("created_at") or ""),
            updated_at=str(row.get("updated_at") or ""))

    def _row_to_step(self, row: Dict[str, Any]) -> WorkflowStepRecord:
        return WorkflowStepRecord(
            step_id=str(row["step_id"]), run_id=str(row["run_id"]),
            index=int(row.get("step_index") or 0),
            node_name=str(row.get("node_name") or ""),
            title=str(row.get("title") or ""),
            status=str(row.get("status") or "pending"),
            agent_id=row.get("agent_id"), agent_name=row.get("agent_name"),
            task=str(row.get("task") or ""),
            result=dict(_json_load(row.get("result"), {})),
            task_id=row.get("task_id"),
            error=str(row.get("error") or ""),
            started_at=row.get("started_at"), completed_at=row.get("completed_at"),
            updated_at=str(row.get("updated_at") or ""))

    def _build_steps(self, *, run_id: str, selected_agents: List[str],
                     agents: List[Dict[str, Any]]) -> List[WorkflowStepRecord]:
        now = utc_now_iso()
        name_by_id = {str(a.get("agent_id")): str(a.get("name") or a.get("agent_id")) for a in agents}
        steps = [WorkflowStepRecord(step_id=f"step_{uuid4().hex}", run_id=run_id, index=1,
                   node_name="main_agent_node", title="Main Agent", status="completed", updated_at=now, completed_at=now)]
        for offset, agent_id in enumerate(selected_agents, 2):
            steps.append(WorkflowStepRecord(step_id=f"step_{uuid4().hex}", run_id=run_id, index=offset,
                          node_name="execute_sub_agent_node", title="子 Agent",
                          agent_id=agent_id, agent_name=name_by_id.get(agent_id, agent_id), updated_at=now))
        summary_idx = len(steps) + 1
        steps.append(WorkflowStepRecord(step_id=f"step_{uuid4().hex}", run_id=run_id, index=summary_idx,
                      node_name="summarize_node" if selected_agents else "direct_answer_node",
                      title="汇总" if selected_agents else "直接回答", updated_at=now))
        return steps

    def _find_step(self, run: AgentRunRecord, *, node_name: Optional[str] = None,
                   agent_id: Optional[str] = None) -> Optional[WorkflowStepRecord]:
        for step in run.steps:
            if agent_id and step.agent_id == agent_id: return step
            if node_name and step.node_name == node_name: return step
        return None

    def _upsert_agent_result(self, current: List[Dict[str, Any]], result: Dict[str, Any]) -> List[Dict[str, Any]]:
        aid = str(result.get("agent_id") or "")
        if not aid: return [*current, dict(result)]
        updated = [dict(result) if str(i.get("agent_id") or "") == aid else i for i in current]
        if not any(str(i.get("agent_id") or "") == aid for i in current):
            updated.append(dict(result))
        return updated

    def _touch(self, run: AgentRunRecord) -> None:
        run.updated_at = utc_now_iso()


agent_run_service = AgentRunService()
