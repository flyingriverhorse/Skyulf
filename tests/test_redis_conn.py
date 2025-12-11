import socket
import sys

def check_port(host, port):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        if result == 0:
            print(f"Success: Connected to {host}:{port}")
        else:
            print(f"Error: Could not connect to {host}:{port} (Code: {result})")
        sock.close()
    except Exception as e:
        print(f"Exception connecting to {host}:{port}: {e}")

print("Testing localhost...")
check_port("localhost", 6379)
print("Testing 127.0.0.1...")
check_port("127.0.0.1", 6379)
