#!/bin/bash
# NIXL LIBFABRIC (EFA) gate driver, one pod per node, mounted from ConfigMap nixl-lf-scripts.
# Usage: nixl-lf-run.sh target|initiator <target_ip>
# Runs nixl_lf_test.py (ONLY the LIBFABRIC backend) once per entry in NIXL_ITERS and snapshots
# the EFA RDMA counters before/after each run, so a pass proves the bytes moved over EFA.
# The repo's nixl_test (test/nixl_test.cpp) hard-codes the UCX backend and cannot prove this.
#
# Knobs (pod env, same values on both pods unless noted):
#   NIXL_ITERS       runs to do, e.g. "1 2 3"                          (default "1 2 3")
#   NIXL_GPUS        GPUs to drive at once, one process each           (default "0")
#                    e.g. "0 1 2 3 4 5 6 7" for all 8 GPUs / all 16 EFA NICs
#   NIXL_OP          READ or WRITE, initiator only                     (default READ)
#   NIXL_TEST_ARGS   extra nixl_lf_test.py args, e.g. "--rows 64 --row-mib 16 --iters 20"
set -u
ROLE=$1; TIP=$2
GPUS=${NIXL_GPUS:-0}
OP=${NIXL_OP:-READ}
EXTRA=${NIXL_TEST_ARGS:-}
# EFA counters via iproute2 `rdma statistic` (sysfs hw_counters is not exposed in these pods).
# Prints per-NIC rdma_read_bytes / rdma_write_bytes / tx_bytes. The tool prints u64 as signed
# 32-bit, so take deltas mod 2^32.
efa_snap() { echo "EFA_COUNTERS $1 $(rdma statistic show 2>/dev/null | awk '{for(i=1;i<NF;i++) if($i=="rdma_read_bytes"||$i=="rdma_write_bytes"||$i=="tx_bytes") printf "%s:%s=%s ", $2, $i, $(i+1)}')"; }
echo "role=$ROLE target_ip=$TIP myip=$(hostname -i) gpus=[$GPUS] op=$OP extra=[$EXTRA]"
echo "EFA devices: $(ls /sys/class/infiniband | tr '\n' ' ')"
echo "NIXL_PLUGIN_DIR=${NIXL_PLUGIN_DIR:-unset}"
LOGD=$(mktemp -d)
rc_all=0
for it in ${NIXL_ITERS:-1 2 3}; do
  echo "=================== NIXL LIBFABRIC iter $it ==================="
  efa_snap "iter$it-before"
  [ "$ROLE" = initiator ] && sleep 10   # let the targets bind first
  # One shared start time, so all GPUs' transfers overlap and NIXL_LF_AGG is real
  # concurrent bandwidth. 30 s covers agent setup + memory registration.
  START_AT=$(( $(date +%s) + 30 ))
  pids=()
  for g in $GPUS; do
    port=$((5500 + it * 16 + g))         # one port per (run, GPU) pair
    if [ "$ROLE" = target ]; then
      timeout 300 python3 /scripts/nixl_lf_test.py --mode target --ip "$TIP" --port $port --gpu "$g" $EXTRA \
        > "$LOGD/$it.$g.log" 2>&1 &
    else
      timeout 300 python3 /scripts/nixl_lf_test.py --mode initiator --ip "$TIP" --port $port --gpu "$g" --op "$OP" --start-at "$START_AT" $EXTRA \
        > "$LOGD/$it.$g.log" 2>&1 &
    fi
    pids+=($!)
  done
  rc=0
  for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
  for g in $GPUS; do cat "$LOGD/$it.$g.log"; done
  if [ "$ROLE" = target ]; then
    echo "NIXL_LF_TARGET iter $it rc=$rc"
  else
    echo "NIXL_LF_INITIATOR iter $it rc=$rc"
    # Aggregate over all GPUs: total bytes / (last finish - first start), wall clock.
    python3 - "$it" "$LOGD/$it".*.log <<'EOF'
import re, sys
it, files = sys.argv[1], sys.argv[2:]
r = [dict(re.findall(r"(\w+)=(\S+)", l)) for f in files for l in open(f) if "NIXL_LF_RESULT PASS" in l]
if r:
    tot = sum(int(x["bytes_per_iter"]) * int(x["iters"]) for x in r)
    span = max(float(x["t_last"]) for x in r) - min(float(x["t_first"]) for x in r)
    s = sum(float(x["mean_GBps"]) for x in r)
    print(f"NIXL_LF_AGG iter {it} gpus={len(r)} op={r[0]['op']} bytes={tot} span_s={span:.3f} "
          f"agg_GBps={tot / span / 1e9:.2f} sum_mean_GBps={s:.2f}")
else:
    print(f"NIXL_LF_AGG iter {it} gpus=0 (no PASS lines)")
EOF
  fi
  efa_snap "iter$it-after"
  [ "$rc" -eq 0 ] || rc_all=1
done
echo "NIXL_LF_RUN_DONE $ROLE rc=$rc_all"
sleep 30
exit $rc_all
