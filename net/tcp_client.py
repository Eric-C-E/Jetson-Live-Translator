# net/tcp_server.py
from __future__ import annotations
import logging
import socket
import select
from typing import Optional, Tuple

class TCPServer:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.bound_host = host
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.srv.bind((host, port))
        except OSError:
            if host != "0.0.0.0":
                self.srv.bind(("0.0.0.0", port))
                self.bound_host = "0.0.0.0"
                logging.warning("Bind failed for %s:%s; falling back to 0.0.0.0:%s", host, port, port)
            else:
                raise
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
            logging.info("TCP client connected from %s:%s", addr[0], addr[1])

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
                logging.info("TCP client disconnected")

        return data

    def send(self, blob: bytes) -> bool:
        if self.conn is None:
            return False
        try:
            self.conn.sendall(blob)
            return True
        except OSError:
            return False
