import socket
from modules.data_handlers.external_data_handle import ExternalDataHandler, DataSampleType, Paths
from utils import string_to_json

host = 'localhost'
port = 8080
data_handle = ExternalDataHandler.get_instance()
data_handle.set_data_sample(DataSampleType.Data8x8)

def decode_data(data: bytes):
    return data.decode()

def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen()
        print(f"Server listening on {host}:{port}")
        while True:
            conn, addr = s.accept()
            with conn:
                print(f"Connected by {addr}")
                while True:
                    # Read the first 4 bytes to get the length of the incoming data
                    length_prefix = conn.recv(4)
                    if not length_prefix:
                        break  # Client closed the connection
                    data_length = int.from_bytes(length_prefix, byteorder='big')

                    # Read the actual data
                    data = b''
                    while len(data) < data_length:
                        packet = conn.recv(data_length - len(data))
                        if not packet:
                            break  # Connection closed or error
                        data += packet

                    decoded_data = decode_data(data)
                    print(f"Data received: {decoded_data}")
                    if decoded_data == "END":
                        data_handle.write_data_array_to_file()
                        conn.close()
                        return
                    else:
                        data_handle.append_data(string_to_json(decoded_data))


if __name__ == "__main__":
    start_server()
