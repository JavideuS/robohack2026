"""
Bedrock + DimOS MCP Agent — connects Claude Sonnet 4.6 (Bedrock, us-west-2)
to the dimos robot's MCP server (localhost:9990 by default), discovers the
22+ robot tools (navigate_with_text, begin_exploration, look_out_for, …),
and runs a streaming converse_stream loop with tool_use handoff.

Drop-in replacement for agent.run_agent_stream — same generator interface so
main.py /query/stream can swap to it without changing protocols.

Usage:
    from agent_mcp import run_mcp_agent_stream
    for token in run_mcp_agent_stream("find a door"):
        print(token, end="", flush=True)
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Generator
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────

MCP_URL = os.environ.get("DIMOS_MCP_URL", "http://localhost:9990/mcp")
AWS_REGION = os.environ.get("AWS_REGION", "us-west-2")
BEDROCK_MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-6"
)
MAX_TOKENS = int(os.environ.get("BEDROCK_MAX_TOKENS", "1024"))
MCP_TIMEOUT = float(os.environ.get("DIMOS_MCP_TIMEOUT", "60"))

SYSTEM_PROMPT = [{
    "text": (
        "You are the brain of a Unitree Go2 quadruped robot running DimOS, "
        "operating either in simulation or in the real world. You can:\n"
        "  - Navigate via the semantic map (navigate_with_text)\n"
        "  - Explore autonomously (begin_exploration / end_exploration)\n"
        "  - Look out for objects (look_out_for)\n"
        "  - Tag locations (tag_location), patrol, follow people, etc.\n"
        "Be concise. Confirm actions out loud (the speak skill is silent text "
        "for now, that's expected). For 'count X' or 'describe scene' style "
        "questions, prefer answering from the semantic map and your own "
        "reasoning rather than hammering the observe tool — observe returns "
        "an asynchronous handle, not the image content."
    )
}]


def _load_dotenv_from_dimos() -> None:
    """If AWS_* env vars aren't set, pull them from dimos's .env so the
    laptop running dimos doesn't need to re-export creds for this server."""
    if os.environ.get("AWS_ACCESS_KEY_ID"):
        return
    dimos_env = Path.home() / "robohack-epfl/dimos/.env"
    if not dimos_env.exists():
        return
    for line in dimos_env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k.startswith("AWS_") and k not in os.environ:
            os.environ[k] = v


_load_dotenv_from_dimos()


# ── MCP HTTP Client ──────────────────────────────────────────

class MCPClient:
    """Tiny HTTP-only MCP client. Speaks the same JSON-RPC envelope dimos's
    server expects (we already verified live: initialize, tools/list,
    tools/call all return 200 in <10 ms locally)."""

    def __init__(self, url: str = MCP_URL, timeout: float = MCP_TIMEOUT) -> None:
        self.url = url
        self.timeout = timeout
        self._next_id = 0
        self._client = httpx.Client(timeout=timeout)

    def _rpc(self, method: str, params: dict | None = None) -> dict:
        self._next_id += 1
        body = {
            "jsonrpc": "2.0",
            "id": self._next_id,
            "method": method,
        }
        if params is not None:
            body["params"] = params
        resp = self._client.post(
            self.url,
            json=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
        )
        resp.raise_for_status()
        # Server may stream SSE for long-running tools; for our short calls it
        # returns plain JSON. Be lenient.
        text = resp.text
        try:
            data = resp.json()
        except Exception:
            # SSE fallback: extract last `data: { ... }` line
            data = None
            for line in text.splitlines():
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                    except Exception:
                        continue
            if data is None:
                raise RuntimeError(f"MCP returned non-JSON: {text[:200]}")
        if "error" in data:
            raise RuntimeError(f"MCP error: {data['error']}")
        return data.get("result", {})

    def initialize(self) -> dict:
        return self._rpc("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "robohack2026-chat", "version": "1.0"},
        })

    def list_tools(self) -> list[dict]:
        return self._rpc("tools/list").get("tools", [])

    def call_tool(self, name: str, arguments: dict) -> str:
        result = self._rpc("tools/call", {"name": name, "arguments": arguments})
        # MCP tool result: {"content": [{"type": "text", "text": "..."}], ...}
        content = result.get("content", [])
        parts: list[str] = []
        for block in content:
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif block.get("type") == "image":
                parts.append("[image returned, not displayed]")
            else:
                parts.append(json.dumps(block))
        if not parts:
            # Some skills return structured result fields (e.g. observe)
            parts.append(json.dumps(result)[:400])
        return "\n".join(parts)

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass


# ── MCP → Bedrock tool-spec conversion ──────────────────────

def _mcp_to_bedrock_tools(mcp_tools: list[dict]) -> list[dict]:
    """Bedrock converse expects:
        {"toolSpec": {"name", "description", "inputSchema": {"json": {...}}}}
    MCP gives:
        {"name", "description", "inputSchema": {...}}
    """
    out: list[dict] = []
    for t in mcp_tools:
        schema = t.get("inputSchema") or {"type": "object", "properties": {}}
        # Bedrock requires properties to exist even if empty
        if "properties" not in schema:
            schema = {**schema, "properties": {}}
        out.append({
            "toolSpec": {
                "name": t["name"],
                "description": t.get("description", "")[:1023],
                "inputSchema": {"json": schema},
            }
        })
    return out


# ── Bedrock streaming agent loop ─────────────────────────────

def _bedrock_stream_with_mcp(
    user_message: str,
    mcp: MCPClient,
    bedrock_tools: list[dict],
) -> Generator[str, None, None]:
    """Multi-turn loop: ask the model → stream tokens → if it requested a
    tool_use, call MCP, append result, loop until the model stops."""
    import boto3

    bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    messages: list[dict] = [{"role": "user", "content": [{"text": user_message}]}]

    max_iterations = 8  # safety guard against runaway tool loops
    for _ in range(max_iterations):
        resp = bedrock.converse_stream(
            modelId=BEDROCK_MODEL_ID,
            system=SYSTEM_PROMPT,
            messages=messages,
            toolConfig={"tools": bedrock_tools} if bedrock_tools else None,
            inferenceConfig={"maxTokens": MAX_TOKENS},
        )

        assistant_content: list[dict] = []
        current_tool: dict[str, Any] = {}
        stop_reason: str | None = None

        for event in resp["stream"]:
            if "contentBlockStart" in event:
                tu = event["contentBlockStart"].get("start", {}).get("toolUse", {})
                if tu:
                    current_tool = {
                        "toolUseId": tu.get("toolUseId"),
                        "name": tu.get("name"),
                        "input_str": "",
                    }

            elif "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                if "text" in delta:
                    yield delta["text"]
                    assistant_content.append({"text": delta["text"]})
                elif "toolUse" in delta:
                    current_tool["input_str"] += delta["toolUse"].get("input", "")

            elif "contentBlockStop" in event:
                if current_tool.get("name"):
                    raw = current_tool.pop("input_str", "") or "{}"
                    try:
                        tool_input = json.loads(raw)
                    except Exception:
                        tool_input = {}
                    assistant_content.append({
                        "toolUse": {
                            "toolUseId": current_tool["toolUseId"],
                            "name": current_tool["name"],
                            "input": tool_input,
                        }
                    })
                    current_tool = {}

            elif "messageStop" in event:
                stop_reason = event["messageStop"].get("stopReason")

        # Merge consecutive text blocks (Bedrock requires non-empty content)
        merged: list[dict] = []
        text_buf = ""
        for blk in assistant_content:
            if "text" in blk:
                text_buf += blk["text"]
            else:
                if text_buf:
                    merged.append({"text": text_buf})
                    text_buf = ""
                merged.append(blk)
        if text_buf:
            merged.append({"text": text_buf})
        if not merged:
            merged.append({"text": ""})
        messages.append({"role": "assistant", "content": merged})

        if stop_reason != "tool_use":
            return

        # Execute MCP tool calls and feed results back
        tool_results: list[dict] = []
        for blk in merged:
            if "toolUse" in blk:
                tu = blk["toolUse"]
                yield f"\n[→ {tu['name']}({json.dumps(tu.get('input', {}))})]\n"
                try:
                    output = mcp.call_tool(tu["name"], tu.get("input", {}))
                except Exception as e:
                    output = f"Error calling {tu['name']}: {e}"
                    logger.exception("MCP tool call failed")
                # Truncate giant outputs so we don't blow the context
                if len(output) > 4000:
                    output = output[:4000] + "\n[...truncated]"
                tool_results.append({
                    "toolResult": {
                        "toolUseId": tu["toolUseId"],
                        "content": [{"text": output}],
                    }
                })
        messages.append({"role": "user", "content": tool_results})

    yield "\n[max tool-loop iterations reached]\n"


# ── Public Interface ─────────────────────────────────────────

def run_mcp_agent_stream(user_message: str) -> Generator[str, None, None]:
    """Public entrypoint — same shape as agent.run_agent_stream so it drops
    into the existing /query/stream endpoint. Yields text tokens.

    Failure modes (each yields a human-readable error then returns):
      - MCP server unreachable  → "MCP not reachable, is dimos running?"
      - AWS not configured      → "Bedrock not configured (set AWS_*)"
      - Bedrock invocation fail → traceback hint
    """
    # 1. Connect to dimos MCP and discover tools
    mcp = MCPClient()
    try:
        try:
            mcp.initialize()
        except Exception as e:
            yield (
                f"⚠ Cannot reach dimos MCP at {MCP_URL}.\n"
                f"Make sure `dimos run unitree-go2-agentic` "
                f"(or `--simulation`) is running.\nError: {e}\n"
            )
            return

        try:
            mcp_tools = mcp.list_tools()
        except Exception as e:
            yield f"⚠ Failed to list dimos tools: {e}\n"
            return

        if not mcp_tools:
            yield (
                "⚠ MCP server returned 0 tools. Wait until dimos finishes "
                "booting (look for `Discovered tools from MCP server. n_tools=22`)\n"
            )
            return

        bedrock_tools = _mcp_to_bedrock_tools(mcp_tools)

        # 2. Run the Bedrock conversation
        try:
            yield from _bedrock_stream_with_mcp(user_message, mcp, bedrock_tools)
        except ImportError:
            yield "⚠ boto3 not installed in this venv. `pip install boto3`.\n"
        except Exception as e:
            logger.exception("Bedrock stream error")
            err = str(e)
            if "ExpiredTokenException" in err or "credentials" in err.lower():
                yield (
                    "⚠ AWS credentials expired or missing.\n"
                    "Refresh AWS_* env vars (or update dimos/.env) and retry.\n"
                    f"Underlying error: {e}\n"
                )
            else:
                yield f"⚠ Bedrock error: {e}\n"
    finally:
        mcp.close()


__all__ = ["run_mcp_agent_stream", "MCPClient"]
