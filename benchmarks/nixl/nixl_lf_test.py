#!/usr/bin/env python3
# 2-node NIXL transfer test forced onto the LIBFABRIC (EFA) backend.
# Adapted from NIXL v1.5.0 examples/python/basic_two_peers.py. The gate's
# nixl_test.cpp hard-codes createBackend("UCX"), so it cannot prove the
# LIBFABRIC path; this script creates ONLY the LIBFABRIC backend.
# Usage: target:    nixl_lf_test.py --mode target    --ip <own ip>    --port 5555
#        initiator: nixl_lf_test.py --mode initiator --ip <target ip> --port 5555
import argparse
import sys
import time

import torch
from nixl._api import nixl_agent, nixl_agent_config

p = argparse.ArgumentParser()
p.add_argument("--ip", required=True)
p.add_argument("--port", type=int, default=5555)
p.add_argument("--mode", default="initiator")
p.add_argument("--rows", type=int, default=64)          # 64 x 16 MiB = 1 GiB
p.add_argument("--row-mib", type=int, default=16)
p.add_argument("--iters", type=int, default=5)
a = p.parse_args()

torch.cuda.set_device(0)
listen_port = a.port if a.mode == "target" else 0
cfg = nixl_agent_config(True, True, listen_port, backends=["LIBFABRIC"])
agent = nixl_agent(a.mode, cfg)
plugins = sorted(agent.get_plugin_list())
backends = sorted(getattr(agent, "backends", {}).keys())
print(f"NIXL plugins available: {plugins}", flush=True)
print(f"NIXL backends created: {backends}", flush=True)
if backends != ["LIBFABRIC"]:
    print("FAIL: backend set is not exactly LIBFABRIC", flush=True)
    sys.exit(2)
print("LIBFABRIC backend selected", flush=True)

n = a.row_mib * 1024 * 1024 // 4
val = 7.0
if a.mode == "target":
    t = torch.full((a.rows, n), val, dtype=torch.float32, device="cuda:0")
else:
    t = torch.zeros((a.rows, n), dtype=torch.float32, device="cuda:0")
torch.cuda.synchronize()
nbytes = t.numel() * 4
reg = agent.register_memory(t, backends=["LIBFABRIC"])
if not reg:
    print("FAIL: memory registration", flush=True)
    sys.exit(3)
rows = [t[i, :] for i in range(a.rows)]
descs = agent.get_xfer_descs(rows)

if a.mode == "target":
    while not agent.check_remote_metadata("initiator"):
        time.sleep(0.01)
    agent.send_notif("initiator", agent.get_serialized_descs(descs))
    print("target: descs sent, waiting for initiator", flush=True)
    while True:
        nt = agent.get_new_notifs()
        if "initiator" in nt and b"ALL_DONE" in nt["initiator"]:
            break
        time.sleep(0.01)
    print("NIXL_LF_TARGET_DONE", flush=True)
else:
    agent.fetch_remote_metadata("target", a.ip, a.port)
    agent.send_local_metadata(a.ip, a.port)
    nt = agent.get_new_notifs()
    while "target" not in nt:
        time.sleep(0.01)
        nt = agent.get_new_notifs()
    remote = agent.deserialize_descs(nt["target"][0])
    while not agent.check_remote_metadata("target"):
        time.sleep(0.01)
    bw = []
    for it in range(a.iters):
        t.zero_()
        torch.cuda.synchronize()
        h = agent.initialize_xfer("READ", descs, remote, "target", f"it{it}".encode(), backends=["LIBFABRIC"])
        t0 = time.perf_counter()
        st = agent.transfer(h)
        if st == "ERR":
            print("FAIL: post transfer", flush=True)
            sys.exit(4)
        while True:
            st = agent.check_xfer_state(h)
            if st == "ERR":
                print("FAIL: transfer error state", flush=True)
                sys.exit(5)
            if st == "DONE":
                break
        dt = time.perf_counter() - t0
        agent.release_xfer_handle(h)
        torch.cuda.synchronize()
        ok = bool(torch.all(t == val).item())
        bw.append(nbytes / dt / 1e9)
        print(f"iter {it}: {nbytes/2**30:.2f} GiB READ in {dt*1e3:.1f} ms = {bw[-1]:.2f} GB/s verify={'OK' if ok else 'MISMATCH'}", flush=True)
        if not ok:
            print("FAIL: data mismatch", flush=True)
            sys.exit(6)
    agent.send_notif("target", b"ALL_DONE")
    time.sleep(1)
    print(f"NIXL_LF_RESULT PASS bytes_per_iter={nbytes} iters={a.iters} mean_GBps={sum(bw)/len(bw):.2f}", flush=True)
    agent.remove_remote_agent("target")
    agent.invalidate_local_metadata(a.ip, a.port)
agent.deregister_memory(reg)
