import asyncio
import base64
import json
import struct
from contextlib import asynccontextmanager
from io import BytesIO
from unittest.mock import AsyncMock, Mock

import pytest
from adbutils import AdbClient
from PIL import Image
from pydantic import ValidationError

from minitap.mobile_use.clients.poco_client import (
    MAX_RESPONSE_BYTES,
    PocoConfig,
    PocoNode,
    dump_hierarchy,
    normalize_hierarchy,
)
from minitap.mobile_use.clients.ui_automator_client import (
    UIAutomatorClient,
    UIAutomatorScreenData,
)
from minitap.mobile_use.config import get_default_llm_config
from minitap.mobile_use.context import DeviceContext, DevicePlatform, MobileUseContext
from minitap.mobile_use.controllers.controller_factory import create_device_controller
from minitap.mobile_use.controllers.unified_controller import UnifiedMobileController


@pytest.fixture
def tree():
    return {
        "name": "root",
        "payload": {"size": [0, 0], "pos": [0, 0]},
        "children": [
            {
                "name": "Play",
                "payload": {
                    "type": "Button",
                    "text": "Jouer",
                    "pos": [0.25, 0.5],
                    "size": [0.2, 0.2],
                },
            },
            {
                "name": "Play",
                "payload": {"pos": [0.75, 0.5], "size": [0.2, 0.2]},
            },
        ],
    }


@asynccontextmanager
async def rpc_server(tree: dict, mode: str = "success"):
    requests: list[dict] = []
    failures: list[Exception] = []
    handlers: set[asyncio.Task] = set()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        task = asyncio.current_task()
        if task is not None:
            handlers.add(task)
        try:
            length = struct.unpack("<i", await reader.readexactly(4))[0]
            request = json.loads(await reader.readexactly(length))
            requests.append(request)
            if mode == "timeout":
                await reader.read()
                return
            if mode in ("oversized", "negative", "zero"):
                size = {"oversized": MAX_RESPONSE_BYTES + 1, "negative": -1, "zero": 0}[mode]
                writer.write(struct.pack("<i", size))
            elif mode == "truncated":
                writer.write(struct.pack("<i", 50) + b"{")
            else:
                response = {"jsonrpc": "2.0", "id": request["id"], "result": tree}
                if mode == "mismatch":
                    response["id"] = "different-request"
                elif mode == "error":
                    response.pop("result")
                    response["error"] = {"code": -32603, "message": "Scene unavailable"}
                elif mode == "missing":
                    response.pop("result")
                elif mode == "bad-version":
                    response["jsonrpc"] = "1.0"
                body = json.dumps(response, ensure_ascii=False).encode("utf-8")
                if mode == "invalid-json":
                    body = b"not JSON"
                packet = struct.pack("<i", len(body)) + body
                writer.write(packet[:2])
                await writer.drain()
                writer.write(packet[2:9])
                await writer.drain()
                writer.write(packet[9:])
            await writer.drain()
        except Exception as error:
            failures.append(error)
        finally:
            writer.close()
            await writer.wait_closed()
            if task is not None:
                handlers.discard(task)

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    config = PocoConfig(
        package_name="com.example.game", port=server.sockets[0].getsockname()[1], timeout=0.5
    )
    async with server:
        try:
            yield config, requests
        finally:
            if handlers:
                await asyncio.wait_for(asyncio.gather(*handlers), timeout=2)
            assert not failures


@pytest.mark.asyncio
async def test_dump_reads_fragmented_utf8_rpc_response(tree):
    tree["children"][0]["payload"]["text"] = "开始"
    async with rpc_server(tree) as (config, requests):
        root = await dump_hierarchy(config)

    assert root.children[0].payload.text == "开始"
    assert len(requests) == 1
    assert requests[0]["jsonrpc"] == "2.0"
    assert requests[0]["method"] == "Dump"
    assert requests[0]["params"] == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "error"),
    [
        ("oversized", ValueError),
        ("negative", ValueError),
        ("zero", ValueError),
        ("mismatch", ValueError),
        ("error", ValueError),
        ("missing", ValueError),
        ("invalid-json", ValidationError),
        ("bad-version", ValidationError),
        ("truncated", asyncio.IncompleteReadError),
        ("timeout", TimeoutError),
    ],
)
async def test_dump_rejects_invalid_or_unresponsive_endpoints(tree, mode, error):
    async with rpc_server(tree, mode) as (config, requests):
        with pytest.raises(error):
            await dump_hierarchy(config)
    assert len(requests) == 1


@pytest.mark.parametrize(
    ("width", "height", "bounds"),
    [(1000, 500, "[150,200][350,300]"), (500, 1000, "[75,400][175,600]")],
)
def test_normalize_preserves_names_text_order_and_screen_coordinates(tree, width, height, bounds):
    result = normalize_hierarchy(PocoNode.model_validate(tree), width, height)
    assert [element["resource-id"] for element in result] == ["Play", "Play"]
    assert result[0]["text"] == "Jouer"
    assert result[0]["class"] == "Button"
    assert result[0]["bounds"] == bounds


def test_normalize_anchor_clipping_visibility_and_nested_nodes(tree):
    first, second = tree["children"]
    first["payload"].update(pos=[0.9, 0.1], size=[0.4, 0.4], anchorPoint=[0, 1])
    first["children"] = [second]
    tree["children"] = [
        first,
        {"name": "hidden", "payload": {"visible": False}, "children": [second]},
        {"name": "offscreen", "payload": {"pos": [2, 2], "size": [0.1, 0.1]}},
        {"name": "zero-size", "payload": {"pos": [0.5, 0.5], "size": [0, 1]}},
    ]
    result = normalize_hierarchy(PocoNode.model_validate(tree), 1000, 500)
    assert [element["bounds"] for element in result] == [
        "[900,0][1000,50]",
        "[650,200][850,300]",
    ]


@pytest.mark.parametrize("coordinate", [float("nan"), float("inf"), "invalid"])
def test_malformed_coordinates_are_rejected(tree, coordinate):
    tree["children"][0]["payload"]["pos"] = [coordinate, 0.5]
    with pytest.raises(ValidationError):
        PocoNode.model_validate(tree)


def make_context(config: PocoConfig | None):
    adb = Mock(spec=AdbClient)
    adb.device.return_value.shell.return_value = (
        "mCurrentFocus=Window{10 u0 com.example.game/.MainActivity}"
    )
    ui = Mock(spec=UIAutomatorClient)
    ui.get_screenshot.side_effect = lambda: Image.new("RGB", (1000, 500))
    ui.get_screen_data.return_value = UIAutomatorScreenData(
        base64="native-screenshot",
        hierarchy_xml="<hierarchy/>",
        elements=[{"resource-id": "native-control", "bounds": "[0,0][10,10]"}],
        width=1000,
        height=500,
    )
    ctx = MobileUseContext(
        trace_id="poco-test",
        device=DeviceContext(
            host_platform="LINUX",
            mobile_platform=DevicePlatform.ANDROID,
            device_id="test-device",
            device_width=1000,
            device_height=500,
        ),
        llm_config=get_default_llm_config(),
        adb_client=adb,
        ui_adb_client=ui,
        poco_config=config,
    )
    return ctx, adb, ui


@pytest.mark.asyncio
async def test_game_observation_and_indexed_tap_use_fresh_poco_data(tree):
    async with rpc_server(tree) as (config, requests):
        ctx, adb, ui = make_context(config)
        observation_controller = create_device_controller(ctx)
        screen = await observation_controller.get_screen_data()
        assert len(screen.elements) == 2
        with Image.open(BytesIO(base64.b64decode(screen.base64))) as image:
            assert image.size == (screen.width, screen.height) == (1000, 500)

        tree["children"][1]["payload"]["pos"] = [0.6, 0.8]
        action_controller = UnifiedMobileController(ctx)
        result = await action_controller.tap_element(resource_id="Play", index=1)
        assert result.error is None
        adb.device.assert_called_with(serial="test-device")
        adb.device.return_value.shell.assert_called_with("input tap 600 400")
        ui.get_screen_data.assert_not_called()
        assert ui.get_screenshot.call_count == 2
        assert len(requests) == 2
        assert requests[0]["id"] != requests[1]["id"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["error", "timeout", "invalid-json", "empty"])
async def test_failed_game_observation_falls_back_to_native(tree, mode):
    if mode == "empty":
        tree = {"name": "root"}
    async with rpc_server(tree, mode) as (config, requests):
        ctx, _, ui = make_context(config)
        screen = await create_device_controller(ctx).get_screen_data()
        assert screen.base64 == "native-screenshot"
        assert screen.elements == ui.get_screen_data.return_value.elements
        ui.get_screen_data.assert_called_once()
        assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_native_path_when_disabled_or_game_not_foreground(monkeypatch, enabled):
    config = PocoConfig(package_name="com.example.game") if enabled else None
    ctx, adb, ui = make_context(config)
    adb.device.return_value.shell.return_value = (
        "mCurrentFocus=Window{10 u0 com.android.settings/.Settings}"
    )
    dump = AsyncMock()
    monkeypatch.setattr("minitap.mobile_use.controllers.android_controller.dump_hierarchy", dump)
    screen = await create_device_controller(ctx).get_screen_data()
    assert screen.base64 == "native-screenshot"
    dump.assert_not_called()
    ui.get_screenshot.assert_not_called()
    ui.get_screen_data.assert_called_once()


def test_factory_rejects_poco_on_ios():
    ctx, _, _ = make_context(PocoConfig(package_name="com.example.game"))
    ctx.device.mobile_platform = DevicePlatform.IOS
    with pytest.raises(ValueError, match="local Android device"):
        create_device_controller(ctx)
