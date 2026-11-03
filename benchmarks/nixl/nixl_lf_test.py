#!/usr/bin/env python3
# 2-node NIXL transfer test forced onto the LIBFABRIC (EFA) backend.
# Adapted from NIXL v1.5.0 examples/python/basic_two_peers.py. The gate's
# nixl_test.cpp hard-codes createBackend("UCX"), so it cannot prove the
# LIBFABRIC path; this script creates ONLY the LIBFABRIC backend.
# Usage: target:    nixl_lf_test.py --mode target    --ip <own ip>    --port 5555 [--gpu N]
#        initiator: nixl_lf_test.py --mode initiator --ip <target ip> --port 5555 [--gpu N] [--op READ|WRITE]
# One process drives one GPU. For an N-GPU run, start N target/initiator pairs, each with its
# own --gpu and --port (nixl-lf-run.sh does this from NIXL_GPUS).
# --op READ : the initiator pulls the target's buffer (one-sided RDMA read).
# --op WRITE: the initiator pushes its buffer into the target (RDMA write).
# Both ops verify every transfer: READ on the initiator, WRITE on the target, which
# sends the verdict back so the initiator prints one verify= line per transfer either way.
import argparse
import sys
import time

import torch
from nixl._api import nixl_agent, nixl_agent_config

p = argparse.ArgumentParser()
p.add_argument("--ip", required=True)
p.add_argument("--port", type=int, default=5555)
p.add_argument("--mode", default="initiator", choices=["initiator", "target"])
p.add_argument("--gpu", type=int, default=0, help="CUDA device index this process uses")
p.add_argument("--op", default="READ", choices=["READ", "WRITE"],
               help="initiator side; the target learns it from the initiator")
p.add_argument("--rows", type=int, default=64)          # 64 x 16 MiB = 1 GiB
p.add_argument("--row-mib", type=int, default=16)
p.add_argument("--iters", type=int, default=5)
p.add_argument("--start-at", type=float, default=0.0,
               help="initiator: wall-clock epoch to start the first transfer, so N parallel "
                    "processes overlap (0 = start as soon as connected)")
a = p.parse_args()

torch.cuda.set_device(a.gpu)
dev = f"cuda:{a.gpu}"
tag = f"gpu{a.gpu}"
listen_port = a.port if a.mode == "target" else 0
cfg = nixl_agent_config(True, True, listen_port, backends=["LIBFABRIC"])
agent = nixl_agent(a.mode, cfg)
plugins = sorted(agent.get_plugin_list())
backends = sorted(getattr(agent, "backends", {}).keys())
print(f"[{tag}] NIXL plugins available: {plugins}", flush=True)
print(f"[{tag}] NIXL backends created: {backends}", flush=True)
if backends != ["LIBFABRIC"]:
    print(f"[{tag}] FAIL: backend set is not exactly LIBFABRIC", flush=True)
    sys.exit(2)
print(f"[{tag}] LIBFABRIC backend selected", flush=True)

n = a.row_mib * 1024 * 1024 // 4
BASE = 7.0
# Every transfer moves a fresh value (BASE + iteration), so a stale buffer can't pass.
t = torch.zeros((a.rows, n), dtype=torch.float32, device=dev)
torch.cuda.synchronize()
nbytes = t.numel() * 4
reg = agent.register_memory(t, backends=["LIBFABRIC"])
if not reg:
    print(f"[{tag}] FAIL: memory registration", flush=True)
    sys.exit(3)
rows = [t[i, :] for i in range(a.rows)]
descs = agent.get_xfer_descs(rows)


_pending = {}


def wait_notif(peer, prefix):
    """Block until `peer` sends a notification starting with `prefix`; return it.
    get_new_notifs() drains everything, so keep the messages not asked for yet."""
    q = _pending.setdefault(peer, [])
    while True:
        for i, msg in enumerate(q):
            if msg.startswith(prefix):
                return q.pop(i)
        for k, msgs in agent.get_new_notifs().items():
            _pending.setdefault(k, []).extend(msgs)
        time.sleep(0.001)


if a.mode == "target":
    while not agent.check_remote_metadata("initiator"):
        time.sleep(0.01)
    op = wait_notif("initiator", b"OP=").decode()[3:]
    print(f"[{tag}] target: op={op}", flush=True)
    for it in range(a.iters):
        if op == "READ":
            t.fill_(BASE + it)              # the initiator will read this value
            torch.cuda.synchronize()
            agent.send_notif("initiator", b"READY" + agent.get_serialized_descs(descs))
            wait_notif("initiator", f"r{it}".encode())  # READ done
        else:
            t.zero_()
            torch.cuda.synchronize()
            agent.send_notif("initiator", b"READY" + agent.get_serialized_descs(descs))
            wait_notif("initiator", f"w{it}".encode())  # WRITE landed (notif follows the data)
            ok = bool(torch.all(t == BASE + it).item())
            agent.send_notif("initiator", f"V{it}={'OK' if ok else 'MISMATCH'}".encode())
    wait_notif("initiator", b"ALL_DONE")
    print(f"[{tag}] NIXL_LF_TARGET_DONE", flush=True)
else:
    agent.fetch_remote_metadata("target", a.ip, a.port)
    agent.send_local_metadata(a.ip, a.port)
    while not agent.check_remote_metadata("target"):
        time.sleep(0.01)
    agent.send_notif("target", f"OP={a.op}".encode())
    if a.start_at:
        late = time.time() - a.start_at
        if late > 0:
            print(f"[{tag}] WARN: connected {late:.1f}s after --start-at; transfers may not overlap", flush=True)
        else:
            time.sleep(-late)
    bw, starts, ends = [], [], []
    for it in range(a.iters):
        remote = agent.deserialize_descs(wait_notif("target", b"READY")[5:])
        if a.op == "READ":
            t.zero_()
        else:
            t.fill_(BASE + it)
        torch.cuda.synchronize()
        # NIXL delivers this note to the target after the transfer completes. For WRITE the
        # target verifies on it; for READ the target waits for the explicit r<it> instead.
        note = f"{'x' if a.op == 'READ' else 'w'}{it}".encode()
        h = agent.initialize_xfer(a.op, descs, remote, "target", note, backends=["LIBFABRIC"])
        w0 = time.time()
        t0 = time.perf_counter()
        st = agent.transfer(h)
        if st == "ERR":
            print(f"[{tag}] FAIL: post transfer", flush=True)
            sys.exit(4)
        while True:
            st = agent.check_xfer_state(h)
            if st == "ERR":
                print(f"[{tag}] FAIL: transfer error state", flush=True)
                sys.exit(5)
            if st == "DONE":
                break
        dt = time.perf_counter() - t0
        starts.append(w0)
        ends.append(w0 + dt)
        agent.release_xfer_handle(h)
        if a.op == "READ":
            torch.cuda.synchronize()
            ok = bool(torch.all(t == BASE + it).item())
            agent.send_notif("target", f"r{it}".encode())  # target may refill its buffer now
        else:
            ok = wait_notif("target", f"V{it}=".encode()).endswith(b"OK")
        bw.append(nbytes / dt / 1e9)
        print(f"[{tag}] iter {it}: {nbytes/2**30:.2f} GiB {a.op} in {dt*1e3:.1f} ms = {bw[-1]:.2f} GB/s "
              f"verify={'OK' if ok else 'MISMATCH'}", flush=True)
        if not ok:
            print(f"[{tag}] FAIL: data mismatch", flush=True)
            sys.exit(6)
    agent.send_notif("target", b"ALL_DONE")
    time.sleep(1)
    # t_first/t_last are wall-clock (time.time) so N parallel processes can be combined into an
    # aggregate: sum(bytes) / (max t_last - min t_first).
    print(f"[{tag}] NIXL_LF_RESULT PASS op={a.op} gpu={a.gpu} bytes_per_iter={nbytes} iters={a.iters} "
          f"mean_GBps={sum(bw)/len(bw):.2f} t_first={min(starts):.6f} t_last={max(ends):.6f}", flush=True)
    agent.remove_remote_agent("target")
    agent.invalidate_local_metadata(a.ip, a.port)
agent.deregister_memory(reg)
