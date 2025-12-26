# net/tcp_server.py
from __future__ import annotations
import socket
import select
from typing import Optional, Tuple

class TCPServer:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind((host, port))
        self.srv.listen(1)
        self.srv.setblocking(False)

        self.conn: Optional[socket.socket] = None
        self.addr: Optional[Tuple[str, int]] = None

    def poll(self, timeout_s: float = 0.01):
        rlist = [self.srv]
        if self.conn is not None:
            rlist.append(self.conn)

        readable, _, _ = select.select(rlist, [], [], timeout_s)

        if self.srv in readable:
            new_conn, addr = self.srv.accept()
            if self.conn is not None:
                try: self.conn.close()
                except OSError: pass
            self.conn = new_conn
            self.addr = addr
            self.conn.setblocking(False)

        data = b""
        if self.conn is not None and self.conn in readable:
            try:
                data = self.conn.recv(4096)
            except BlockingIOError:
                data = b""
            except OSError:
                data = b""

            if data == b"":
                # closed
                try: self.conn.close()
                except OSError: pass
                self.conn = None
                self.addr = None

        return data

    def send(self, blob: bytes) -> bool:
        if self.conn is None:
            return False
        try:
            self.conn.sendall(blob)
            return True
        except OSError:
            return False