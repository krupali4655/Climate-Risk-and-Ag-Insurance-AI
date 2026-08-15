"""
Thin sync wrapper around the FastMCP client.

Streamlit scripts run top-to-bottom on every interaction and are not async,
so rather than threading asyncio through app.py, we open a short-lived MCP
connection per call and close it immediately. This is a little less
efficient than holding a persistent connection open, but it's simple,
correct, and matches how Streamlit already reruns the whole script on every
user action - each call below is just doing its own small "rerun-scoped"
round trip to the ag_server.py microservice.
"""

import asyncio
from fastmcp import Client

MCP_SERVER_URL = "http://127.0.0.1:8931/mcp"


async def _list_tools_async():
    async with Client(MCP_SERVER_URL) as client:
        tools = await client.list_tools()
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.inputSchema,
                },
            }
            for t in tools
        ]


async def _call_tool_async(name: str, arguments: dict):
    async with Client(MCP_SERVER_URL) as client:
        result = await client.call_tool(name, arguments)
        return result.content[0].text


def get_mcp_tool_schemas() -> list[dict]:
    """
    Discover tools live from ag_server.py over the MCP protocol and return
    them already formatted for Groq's `tools` parameter.

    This replaces the hand-written `groq_tools` list that used to live in
    app.py: schemas are no longer duplicated by hand, they come straight
    from the server. Add a new @mcp.tool in ag_server.py and it shows up
    here automatically, with no changes needed in app.py.
    """
    try:
        return asyncio.run(_list_tools_async())
    except Exception as e:
        raise RuntimeError(
            "Could not reach the Ag-Insurance MCP server at "
            f"{MCP_SERVER_URL}. Is `python ag_server.py` running in a "
            f"separate terminal? Original error: {e}"
        )


def call_mcp_tool(name: str, arguments: dict) -> str:
    """Call a tool on ag_server.py over the real MCP protocol and return its string result."""
    return asyncio.run(_call_tool_async(name, arguments))