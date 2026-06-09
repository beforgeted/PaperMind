"""验证日志系统：日期目录、agent 业务通道、context 注入、审计隔离。"""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        log_dir = Path(tmp) / "log"
        import app.core.config as config_module
        from app.core import logging as logging_module

        settings = config_module.settings
        originals = {
            "log_dir": settings.log_dir,
            "log_json_to_file": settings.log_json_to_file,
            "log_to_stdout": settings.log_to_stdout,
            "log_date_subdirs": settings.log_date_subdirs,
            "log_agent_business_file": settings.log_agent_business_file,
            "_setup_done": logging_module._setup_done,
        }

        try:
            settings.log_dir = str(log_dir)
            settings.log_json_to_file = True
            settings.log_to_stdout = False
            settings.log_date_subdirs = True
            settings.log_agent_business_file = True
            logging_module._setup_done = False

            from app.core.log_context import log_bind
            from app.core.logging import get_audit_logger, resolve_log_path, setup_logging
            from app.mcp_gateway.audit_logger import audit_logger
            from app.observability.events import MCP_TOOL_CALL, RUN_CREATED

            setup_logging()
            app_logger = __import__("logging").getLogger("test.papermind")

            with log_bind(request_id="req_test123", run_id="run_test456"):
                app_logger.info(
                    "infra noise",
                    extra={"event": "infra_startup", "foo": "bar"},
                )
                app_logger.info(
                    "run created",
                    extra={"event": RUN_CREATED, "query_len": 12},
                )
                audit_logger.log_call(
                    "search_papers",
                    {"query": "transformer", "top_k": 5},
                    latency_ms=42,
                    ok=True,
                )

            today = date.today()
            app_log = resolve_log_path("app.json.log", day=today)
            agent_log = resolve_log_path("agent.json.log", day=today)
            audit_log = resolve_log_path("audit.json.log", day=today)

            assert app_log.exists(), f"missing {app_log}"
            assert agent_log.exists(), f"missing {agent_log}"
            assert audit_log.exists(), f"missing {audit_log}"
            assert today.strftime("%Y/%m/%d") in str(app_log).replace("\\", "/")

            app_lines = [
                json.loads(line)
                for line in app_log.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            agent_lines = [
                json.loads(line)
                for line in agent_log.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            assert len(app_lines) == 2
            assert len(agent_lines) == 1
            assert agent_lines[0].get("event") == RUN_CREATED
            assert app_lines[0].get("request_id") == "req_test123"

            audit_line = json.loads(audit_log.read_text(encoding="utf-8").strip())
            assert audit_line.get("event") == MCP_TOOL_CALL
            assert audit_line.get("run_id") == "run_test456"

            audit_logger_obj = get_audit_logger()
            assert audit_logger_obj.propagate is False

            print("OK: logging system verification passed")
        finally:
            import logging as std_logging

            root = std_logging.getLogger()
            for handler in list(root.handlers):
                handler.close()
                root.removeHandler(handler)
            audit = logging_module.get_audit_logger()
            for handler in list(audit.handlers):
                handler.close()
                audit.removeHandler(handler)
            for key, value in originals.items():
                if key == "_setup_done":
                    logging_module._setup_done = value
                else:
                    setattr(settings, key, value)


if __name__ == "__main__":
    main()
