#!/bin/bash
# NIXL LIBFABRIC (EFA) gate driver, one process per node, mounted from ConfigMap nixl-lf-scripts.
# Usage: nixl-lf-run.sh target|initiator <target_ip>
# Runs nixl_lf_test.py (ONLY the LIBFABRIC backend) NIXL_ITERS times and snapshots the EFA
# RDMA counters before/after each run, so a pass proves the bytes moved over EFA.
# The repo's nixl_test (test/nixl_test.cpp) hard-codes the UCX backend and cannot prove this.
set -u
ROLE=$1; TIP=$2
# EFA counters via iproute2 `rdma statistic` (sysfs hw_counters is not exposed in these pods).
# Prints per-NIC rdma_read_bytes / rdma_write_bytes / tx_bytes. The tool prints u64 as signed
# 32-bit, so take deltas mod 2^32.
efa_snap() { echo "EFA_COUNTERS $1 $(rdma statistic show 2>/dev/null | awk '{for(i=1;i<NF;i++) if($i=="rdma_read_bytes"||$i=="rdma_write_bytes"||$i=="tx_bytes") printf "%s:%s=%s ", $2, $i, $(i+1)}')"; }
echo "role=$ROLE target_ip=$TIP myip=$(hostname -i)"
echo "EFA devices: $(ls /sys/class/infiniband | tr '\n' ' ')"
echo "NIXL_PLUGIN_DIR=${NIXL_PLUGIN_DIR:-unset}"
rc_all=0
for it in ${NIXL_ITERS:-1 2 3}; do
  echo "=================== NIXL LIBFABRIC iter $it ==================="
  efa_snap "iter$it-before"
  if [ "$ROLE" = target ]; then
    timeout 300 python3 /scripts/nixl_lf_test.py --mode target --ip "$TIP" --port $((5550 + it)); rc=$?
    echo "NIXL_LF_TARGET iter $it rc=$rc"
  else
    sleep 10   # let the target bind first
    timeout 300 python3 /scripts/nixl_lf_test.py --mode initiator --ip "$TIP" --port $((5550 + it)); rc=$?
    echo "NIXL_LF_INITIATOR iter $it rc=$rc"
  fi
  efa_snap "iter$it-after"
  [ "$rc" -eq 0 ] || rc_all=1
done
echo "NIXL_LF_RUN_DONE $ROLE rc=$rc_all"
sleep 30
exit $rc_all
