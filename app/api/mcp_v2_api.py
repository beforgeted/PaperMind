"""基于 Agent 接口 MCP 服务注册表的 MCP 工具 HTTP 接口。"""

import json
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.mcp_gateway.gateway import mcp_gateway
from app.mcp_gateway.registry import list_tools
from app.schemas.mcp_schema import McpToolCallRequest, McpToolCallResponse
from app.services.mcp_server_service import mcp_server_registry


router = APIRouter(prefix="/api/v1/mcp", tags=["MCP 工具"])


@router.get("/servers")
async def list_mcp_servers() -> dict[str, Any]:
    return {"servers": mcp_server_registry.list_servers()}


@router.get("/tools")
async def list_mcp_tools(mcp_server: str | None = None) -> dict[str, Any]:
    tools = list_tools()
    if mcp_server:
        server = mcp_server_registry.get(mcp_server)
        allowed = set(server.tool_names) if server else set()
        tools = {
            name: meta
            for name, meta in tools.items()
            if name in allowed or meta.get("mcp_server") == mcp_server
        }
    return {"tools": tools}


@router.post("/tools/call", response_model=McpToolCallResponse)
async def call_mcp_tool(request: McpToolCallRequest) -> McpToolCallResponse:
    result = await mcp_gateway.call_tool(request.tool_name, request.arguments)
    return McpToolCallResponse(
        ok=bool(result.get("ok")),
        content=result.get("content"),
        error=str(result.get("error") or ""),
        mcp_server=request.mcp_server,
        tool_name=request.tool_name,
    )


@router.post("/tools/call/stream")
async def stream_mcp_tool(request: McpToolCallRequest) -> StreamingResponse:
    """以 SSE 流式返回任意 MCP 工具执行事件。"""

    async def generate():
        async for event in mcp_gateway.stream_tool(request.tool_name, request.arguments):
            event.setdefault("mcp_server", request.mcp_server)
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
