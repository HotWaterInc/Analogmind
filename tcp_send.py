import json
import socket
from typing import Union

server_host = 'localhost'
server_port = 8080

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


def start_client():
    server.connect((server_host, server_port))
    print("Connected to server successfully!")


def encode_data(data: Union[dict, str]):
    if isinstance(data, dict):
        return json.dumps(data).encode()
    elif isinstance(data, str):
        return data.encode()


def send_data(data: Union[dict, str]):
    encoded_data = encode_data(data)
    data_length = len(encoded_data)

    server.sendall(data_length.to_bytes(4, byteorder='big'))
    server.sendall(encoded_data)

    print("Data sent successfully!")


def close_client():
    server.close()
    print("Connection to server closed.")


if __name__ == "__main__":
    start_client()
    try:
        send_data("Hello, server!")
        send_data("Hello, server!2")
        send_data("Hello, server!3")
        send_data("END")  # Indicate the end of transmission if needed
    finally:
        close_client()
