import asyncio
import websockets
import json
from .communication_interface import receive_data
from .communication_interface import set_server_started, CommunicationInterface
from typing import Dict

PORT = 8080
websocket_global = None


async def listen(websocket):
    global websocket_global
    websocket_global = websocket
    set_server_started()
    async for message in websocket:
        receive_data(json.loads(message))


async def send_data_string_websockets(websocket, message):
    await websocket.send(message)


def send_data_websockets(json_data: Dict[str, any]):
    global websocket_global
    websocket = websocket_global

    print("Sending data to websockets: ", json_data)
    message = json.dumps(json_data)

    # data is sometimes sent by the main thread ( which needs to create a new asyncio event loop ) or by the thread
    # which runs the server ( for example in send_pending_data, the server thread calls this callback which calls
    # send again )

    # We need to cover both cases
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(send_data_string_websockets(websocket, message))
    except RuntimeError:  # No running event loop
        asyncio.run(send_data_string_websockets(websocket, message))


async def start_websockets_server():
    server = await websockets.serve(listen, "localhost", PORT)
    print("Server started at ws://localhost:" + str(PORT) + ", waiting for connections")
    await server.wait_closed()


def start_websockets(set_server_started_cb=(lambda: None)):
    asyncio.run(start_websockets_server())


if __name__ == "__main__":
    start_websockets()
