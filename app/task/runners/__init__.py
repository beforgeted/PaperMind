"""Runner 注册表，所有可用 Runner 的统一入口。"""

from app.task.runners.mcp_tool_runner import McpToolRunner


RUNNER_REGISTRY = {
    McpToolRunner.tool_name: McpToolRunner(),
}


__all__ = ["RUNNER_REGISTRY", "McpToolRunner"]
