"""Tiny length-prefixed pickle IPC over a socket. Stdlib only -> imports in BOTH conda envs.

The closed loop spans two conda envs that cannot co-import (base = torch+dinov2, no pybullet;
langtable = pybullet+language_table, no torch). The planner (base) is the socket SERVER; the sim
worker (langtable) is the CLIENT that connects back. Protocol: 4-byte big-endian length + pickle.
Using a socket (not stdout) keeps the channel clean of pybullet/gym import banners.
"""
import pickle
import socket
import struct


def _recvn(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def send(sock, obj):
    b = pickle.dumps(obj, protocol=4)
    sock.sendall(struct.pack(">I", len(b)) + b)


def recv(sock):
    hdr = _recvn(sock, 4)
    if hdr is None:
        return None
    n = struct.unpack(">I", hdr)[0]
    body = _recvn(sock, n)
    if body is None:
        return None
    return pickle.loads(body)
