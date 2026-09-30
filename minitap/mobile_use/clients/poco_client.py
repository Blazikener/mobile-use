import asyncio
import json
import math
import struct
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, FiniteFloat


class PocoConfig(BaseModel):
    """Poco-SDK endpoint for an instrumented local Android game."""

    package_name: str = Field(min_length=1, pattern=r"^\S+$")
    host: str = Field(default="127.0.0.1", min_length=1)
    port: int = Field(default=5001, ge=1, le=65535)
    timeout: FiniteFloat = Field(default=2.0, gt=0)


class PocoPayload(BaseModel):
    name: str = ""
    type: str = ""
    text: str | None = None
    visible: bool = True
    pos: tuple[float, float] | None = None
    size: tuple[float, float] | None = None
    anchorPoint: tuple[float, float] = (0.5, 0.5)


class PocoNode(BaseModel):
    name: str = ""
    payload: PocoPayload = Field(default_factory=PocoPayload)
    children: list["PocoNode"] = Field(default_factory=list)


class PocoRpcError(BaseModel):
    code: int
    message: str


class PocoResponse(BaseModel):
    jsonrpc: Literal["2.0"]
    id: str
    result: PocoNode | None = None
    error: PocoRpcError | None = None


MAX_RESPONSE_BYTES = 8 * 1024 * 1024


async def dump_hierarchy(config: PocoConfig) -> PocoNode:
    """Read Poco-SDK's length-prefixed JSON-RPC Dump response."""
    request_id = str(uuid4())
    request = json.dumps(
        {"jsonrpc": "2.0", "id": request_id, "method": "Dump", "params": [True]}
    ).encode("utf-8")
    async with asyncio.timeout(config.timeout):
        reader, writer = await asyncio.open_connection(config.host, config.port)
        try:
            writer.write(struct.pack("<i", len(request)) + request)
            await writer.drain()
            length = struct.unpack("<i", await reader.readexactly(4))[0]
            if not 0 < length <= MAX_RESPONSE_BYTES:
                raise ValueError("Invalid Poco response length")
            response = PocoResponse.model_validate_json(await reader.readexactly(length))
            if response.id != request_id:
                raise ValueError("Poco response does not match the request")
            if response.error is not None:
                raise ValueError(f"Poco Dump failed: {response.error.message}")
            if response.result is None:
                raise ValueError("Poco response has no hierarchy")
            return response.result
        finally:
            writer.close()
            await writer.wait_closed()


def normalize_hierarchy(root: PocoNode, width: int, height: int) -> list[dict]:
    """Convert screen-relative Poco nodes to flat Android selectors and pixel bounds."""
    if width <= 0 or height <= 0:
        raise ValueError("Screen dimensions must be positive")
    elements: list[dict] = []
    nodes = [root]
    while nodes:
        node = nodes.pop()
        payload = node.payload
        if not payload.visible:
            continue
        nodes.extend(reversed(node.children))
        if payload.pos is None or payload.size is None:
            continue
        size_x, size_y = payload.size
        if size_x <= 0 or size_y <= 0:
            continue
        pos_x, pos_y = payload.pos
        anchor_x, anchor_y = payload.anchorPoint
        left = (pos_x - anchor_x * size_x) * width
        top = (pos_y - anchor_y * size_y) * height
        right = left + size_x * width
        bottom = top + size_y * height
        if not all(math.isfinite(value) for value in (left, top, right, bottom)):
            continue
        x1 = round(max(0, min(width, left)))
        y1 = round(max(0, min(height, top)))
        x2 = round(max(0, min(width, right)))
        y2 = round(max(0, min(height, bottom)))
        if x1 >= x2 or y1 >= y2:
            continue
        name = node.name or payload.name
        elements.append(
            {
                "resource-id": name,
                "text": payload.text or "",
                "class": payload.type,
                "bounds": f"[{x1},{y1}][{x2},{y2}]",
            }
        )
    return elements
