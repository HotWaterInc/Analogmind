import json
import asyncio
import aioconsole

server_host = 'localhost'
server_port = 8080

async def tcp_client():
    reader, writer = await asyncio.open_connection(server_host, server_port)
    print("Connected to server successfully!")

    while True:
        x_input = await aioconsole.ainput("Enter x coordinate (type 'exit' to quit): ")
        if x_input.lower() == 'exit':
            print("Exiting client.")
            break
        y_input = await aioconsole.ainput("Enter y coordinate: ")

        try:
            x = float(x_input)
            y = float(y_input)
        except ValueError:
            print("Invalid input. Please enter valid floating-point numbers.")
            continue

        data = {"x": x, "y": y}
        await send_data(writer, data)

    print("Closing the connection.")
    writer.close()
    await writer.wait_closed()

async def send_data(writer, data: dict):
    encoded_data = json.dumps(data).encode('utf-8')
    # First, send the length of the encoded data
    writer.write(len(encoded_data).to_bytes(4, byteorder='big'))
    # Then, send the actual encoded data
    writer.write(encoded_data)
    await writer.drain()
    print(f"Data sent successfully: {data}")

if __name__ == "__main__":
    asyncio.run(tcp_client())
