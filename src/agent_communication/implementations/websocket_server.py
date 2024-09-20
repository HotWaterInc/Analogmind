import asyncio
import websockets
import json
from typing import Dict

from src.modules.agent_communication import ai_receive_response
from src.modules.agent_communication.communication_controller import set_server_started

PORT = 8080
websocket_global = None


async def listen(websocket):
    global websocket_global
    websocket_global = websocket
    set_server_started()
    try:
        async for message in websocket:
            ai_receive_response(json.loads(message))
    finally:
        websocket_global = None


async def send_data_string_websockets(websocket, message):
    await websocket.send(message)


def send_data_websockets(json_data: Dict[str, any]):
    global websocket_global
    websocket = websocket_global
    if websocket is None:
        print("WebSocket not connected. Unable to send data.")
        return
    message = json.dumps(json_data)
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(send_data_string_websockets(websocket, message))
    except RuntimeError:  # No running event loop
        asyncio.run(send_data_string_websockets(websocket, message))


async def _start_websockets_server():
    try:
        server = await websockets.serve(
            listen,
            "localhost",
            PORT,
            ping_interval=None,  # Disable ping/pong mechanism
            ping_timeout=None
        )
        print(f"Server started at ws://localhost:{PORT}, waiting for connections")
        await server.wait_closed()
    except Exception as e:
        print(f"Server error: {e}. ")


def start_websockets():
    asyncio.run(_start_websockets_server())
