import asyncio
import websockets
import json
from .communication_interface import receive_data

PORT = 8080
websocket_global = None
set_server_started = None

async def listen(websocket):
    global websocket_global
    websocket_global = websocket
    set_server_started()
    async for message in websocket:
        receive_data(message)


async def send_data_websockets(websocket, message):
    await websocket.send(message)


def send_data(json_data):
    global websocket_global
    websocket = websocket_global

    asyncio.run(send_data_websockets(websocket, json_data))

async def start_websockets_server():
    server = await websockets.serve(listen, "localhost", PORT)
    print("Server started at ws://localhost:" + str(PORT))
    await server.wait_closed()

def start_websockets(set_server_started_cb=(lambda: None)):
    global set_server_started
    set_server_started = set_server_started_cb
    asyncio.run(start_websockets_server())

if __name__ == "__main__":
    start_websockets()
