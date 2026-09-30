# Experimental Android game hierarchies with Poco

Native Android accessibility trees often omit controls rendered by game engines.
Mobile-use can optionally read the element tree exposed by an instrumented
[Poco-SDK](https://github.com/AirtestProject/Poco-SDK) game while retaining its
existing Android screenshots and ADB input.

This is an SDK-only integration for a local Android device. It does not add an
element tree to an arbitrary downloaded game. You need a game build containing
Poco-SDK and a UI framework supported by that SDK. Airtest's image-template
matching is separate from Poco's element-tree service.

## Prepare a game and connection

1. For Unity, follow the [Poco-SDK Unity instructions](https://github.com/AirtestProject/Poco-SDK/tree/master/Unity3D):
   add the SDK scripts for your UI framework, attach `PocoManager` to a persistent
   GameObject, and rebuild the Android game. For Unreal, the SDK provides a
   [PocoSDK plugin](https://github.com/AirtestProject/Poco-SDK/tree/master/Unreal)
   that must be enabled and compiled into the game. Check that SDK's engine and
   platform support; this integration does not establish UE5 compatibility.
2. Install and launch that build on the Android device used by mobile-use.
   Confirm the game exposes a meaningful tree, rather than only a root node.
3. Forward the game's actual Poco TCP port to your machine. The usual default is
   5001, but PocoManager may select another port if it is already occupied.
   Select the same device explicitly for both the forward and mobile-use:

   ```sh
   adb -s YOUR_DEVICE_SERIAL forward tcp:5001 tcp:5001
   ```

The integration connects to the forwarded endpoint directly. It does not require
the Python `pocoui` or `airtest` packages, and does not create or own the forward.
Remove it when finished with `adb -s YOUR_DEVICE_SERIAL forward --remove tcp:5001`.
Use instrumented development/test builds; do not expose the game's debug service
on a public network.

## Configure the SDK

Complete the normal mobile-use setup and configure your LLM provider first.
Replace the device serial and Android package name below:

```python
import asyncio

from minitap.mobile_use.sdk import Agent
from minitap.mobile_use.sdk.builders import Builders
from minitap.mobile_use.sdk.types import DevicePlatform, PocoConfig


async def main():
    config = (
        Builders.AgentConfig
        .for_device(DevicePlatform.ANDROID, "YOUR_DEVICE_SERIAL")
        .with_poco_hierarchy(
            PocoConfig(
                package_name="com.example.game",
                host="127.0.0.1",
                port=5001,
                timeout=2.0,
            )
        )
        .build()
    )
    agent = Agent(config=config)
    try:
        await agent.init()
        await agent.run_task(goal="Tap the Play button in the game")
    finally:
        await agent.clean()


asyncio.run(main())
```

While that package has the focused window, both observations and action-time
element lookup use a fresh Poco `Dump` call. Node names become `resource-id`
selectors, node text remains selectable, and repeated names retain depth-first
order for indexed taps. Bounds use the screenshot's actual dimensions and Poco's
normalized position, size, and anchor point. Invisible subtrees and targets with
empty/off-screen bounds are excluded; partially visible bounds are clipped.

Other apps and system windows use the native hierarchy. A timeout, connection
failure, malformed response, or a tree with no usable bounds logs a warning and
falls back to the native hierarchy. No previous game tree is cached. A configured
endpoint must belong to the selected device and game; the RPC service cannot
verify that binding itself.

## Scope and verification

The initial integration supports screen-space game observation and coordinate,
name, text, and indexed taps through existing Android input. It does not expose
Poco's attribute-setting APIs or promise engine-specific text editing, arbitrary
3D-object targeting, render-offset correction, or iOS/cloud-device support.
Keep the device orientation fixed during an agent run, as existing percentage
tools use the dimensions recorded at initialization.

Offline tests exercise a TCP RPC endpoint, response framing and errors,
normalization, controller observation, refreshed indexed taps, and native
fallback. They do not establish compatibility with an actual Unity or Unreal
build. Before relying on a game, verify its named controls align with screenshot
bounds and that taps change the intended game state, including scene changes,
system dialogs, and disconnect/reconnect.

Protocol references:

- [Poco standard hierarchy dumper](https://github.com/AirtestProject/Poco/blob/master/poco/drivers/std/dumper.py)
- [Poco TCP framing](https://github.com/AirtestProject/Poco/blob/master/poco/utils/simplerpc/transport/tcp/protocol.py)
- [Poco JSON-RPC requests](https://github.com/AirtestProject/Poco/blob/master/poco/utils/simplerpc/simplerpc.py)
- [Node geometry and visibility](https://poco.readthedocs.io/en/latest/_modules/poco/sdk/AbstractNode.html)
