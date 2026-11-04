# NIXL + DeepEP V2 on AWS EFA — microbenchmarks

A reproducible container recipe and 2-node validation suite for GPU-collective
communication on AWS EFA. It builds one multi-arch image carrying three stacks —
**NCCL** (collectives over EFA), **DeepEP V2** (expert-parallel dispatch/combine
over NCCL GIN / EFA-GDA), and **NIXL** (point-to-point transfer over the
LIBFABRIC/EFA backend) — and the Kubernetes manifests to benchmark all three on
2 × `p5en.48xlarge` (H200, 8 GPU + 16 EFA NICs per node).

Built for Hopper (`sm_90`) and Blackwell (`sm_100`, `sm_103`) in a single image.

## What's here

| Path | What |
|---|---|
| `awslabs-contrib/architectures/containers/deepv2-nixl-efa/` | the container recipe (one image: Dockerfile, build helper, license gate) |
| `benchmarks/nccl/` | NCCL collectives manifests |
| `benchmarks/deepep/` | DeepEP V2 manifests + GIN probes |
| `benchmarks/nixl/` | NIXL target/initiator manifests |
| `benchmarks/efa/` | EFA version probe |
| `docs/deepep-nixl-pipeline-design.md` | full design doc (decisions, build fixes, pipeline) |
| `env.example.sh` | the environment variables every manifest and doc uses |

All manifests and the recipe use `${VAR}` placeholders (account, region,
cluster, namespace, repo…) instead of hardcoded values. Fill them in by copying
`env.example.sh`, editing it, sourcing it, and substituting with `envsubst`
(shown below).

## Pinned versions

| Component | Version |
|---|---|
| CUDA base | `nvidia/cuda:13.0.2-devel-ubuntu22.04` |
| NIXL | `v1.5.0` |
| NCCL (GIN) | `2.31.2` |
| EFA installer | `1.50.0` |
| torch | `2.11.0` (cu130) |
| gdrcopy | `v2.5.2` |

## Build the container

From the recipe directory:

```bash
cd awslabs-contrib/architectures/containers/deepv2-nixl-efa

DOCKER_BUILDKIT=1 docker build --progress=plain -f Dockerfile \
  --build-arg TORCH_CUDA_ARCH_LIST="9.0;10.0;10.3" \
  --build-arg NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_100,code=sm_100 -gencode=arch=compute_103,code=sm_103" \
  -t <registry>/nixl-deepep-efa:dev .
```

The build is self-checking: the build fails if the NCCL-GIN library is stale or
a backend plugin is missing. A green build is already partly validated when the
build succeeds. Push to your registry:

```bash
docker push <registry>/nixl-deepep-efa:dev
```

### License gate (optional, before publishing)

```bash
export SYFT_LICENSE_CONTENT=all
syft <registry>/nixl-deepep-efa:dev \
  --override-default-catalogers "dpkg-db-cataloger,python-installed-package-cataloger" \
  -o json | python3 awslabs-contrib/architectures/containers/deepv2-nixl-efa/sbom_emit.py > licenses.tsv
python3 awslabs-contrib/architectures/containers/deepv2-nixl-efa/check_licenses.py \
  --gate awslabs-contrib/architectures/containers/deepv2-nixl-efa/validation-gate.yaml licenses.tsv
```

`SIGN-OFF` = clear to publish. `BLOCK` = a license outside the AWS OSS allow-list.

## Reproduce the benchmarks

### Prerequisites

- 2 × `p5en.48xlarge` nodes in an EKS cluster, EFA-enabled.
- **GPU driver flags on the nodes** (required for DeepEP V2 GIN / EFA-GDA):
  `NVreg_EnableStreamMemOPs=1` and `NVreg_RegistryDwords="PeerMappingOverride=1;"`.
  DeepEP V2 lets the GPU drive network transfers itself, with no CPU in the
  loop (GPU-initiated networking). That needs two driver features that are off
  by default:
  - `NVreg_EnableStreamMemOPs=1` lets GPU kernels wait on and write memory
    flags directly. The GPU uses this to signal "data sent" and "data arrived".
  - `PeerMappingOverride=1` lets the EFA NIC map the GPU's memory (MMIO)
    so the NIC and GPU can talk directly.

  Without them, GIN fails with `DevComm setup failed on all available backends`.
  Bake them into the nodegroup launch template userData; a reboot alone does not
  stick, because replaced nodes come back without them. Verify on a node:
  ```bash
  grep -E "EnableStreamMemOPs|RegistryDwords" /proc/driver/nvidia/params
  ```
- The kubeflow **mpi-operator** CRD installed (`kubectl get crd mpijobs.kubeflow.org`).
- The built image pushed to a registry your cluster can pull.

Edit the `env.sh` file with your environment variables first, then apply it
using `envsubst`. The manifests carry `${VAR}` placeholders that get replaced
with your own environment variables:

```bash
cp env.example.sh env.sh      # then edit env.sh with your values
source env.sh
```

### Run the three suites

Each is a 2-node / 16-rank job. Apply, wait for the launcher pod to reach
`Completed`, then read its logs for the success markers.

```bash
# 1. NCCL collectives — all-reduce, all-gather, all-to-all over EFA
envsubst < benchmarks/nccl/nccl-tests-validate.yaml | kubectl apply -f -

# 2. DeepEP V2 — expert-parallel dispatch/combine over NCCL GIN / EFA-GDA
envsubst < benchmarks/deepep/deepep-validate.yaml | kubectl apply -f -

# 3. NIXL — point-to-point GPU transfer over ONLY the LIBFABRIC/EFA backend
kubectl -n ${K8S_NAMESPACE} create configmap nixl-lf-scripts \
  --from-file=benchmarks/nixl/nixl-lf-run.sh --from-file=benchmarks/nixl/nixl_lf_test.py
envsubst < benchmarks/nixl/nixl-lf-target-validate.yaml | kubectl apply -f -      # target first (NODE_1)
kubectl -n ${K8S_NAMESPACE} wait --for=condition=Ready pod/nixl-lf-target --timeout=15m
export NIXL_TARGET_IP=$(kubectl -n ${K8S_NAMESPACE} get pod nixl-lf-target -o jsonpath='{.status.podIP}')
envsubst < benchmarks/nixl/nixl-lf-initiator-validate.yaml | kubectl apply -f -   # then initiator (NODE_0)
```

Every NIXL path in this repo uses the LIBFABRIC backend only: EFA does not support UCX.
The images build NIXL without UCX, so NIXL's own `nixl_test` (which hard-codes
`createBackend("UCX")`) is no longer shipped and its manifests are gone. The images carry
NIXL's LIBFABRIC and backend-free C++ tests in `/opt/nixl-tests` (in the vLLM image, run them
as `nixl-test <name>`), for example `libfabric_topology_test` to print the EFA rail layout.

#### NIXL test knobs (network load)

Export these before the two `envsubst` lines. Empty means the default. `NIXL_GPUS` and
`NIXL_TEST_ARGS` must be the same on both pods.

| Env | Default | What it changes |
|---|---|---|
| `NIXL_GPUS` | `0` | GPUs driven at once, one process each. One p5en GPU uses its 2 nearest EFA NICs; `"0 1 2 3 4 5 6 7"` uses all 16 |
| `NIXL_OP` | `READ` | `READ` (initiator pulls) or `WRITE` (initiator pushes; the target verifies) |
| `NIXL_TEST_ARGS` | empty | extra `nixl_lf_test.py` args: `--rows` (default 64), `--row-mib` (16; 64 x 16 MiB = 1 GiB per transfer), `--iters` (5 transfers per run) |
| `NIXL_ITERS` | `1 2 3` | runs per pod (set in the manifests) |

All initiator processes start their first transfer at one shared time (`--start-at`), so
`NIXL_LF_AGG ... agg_GBps` is the real combined bandwidth. `sum_mean_GBps` adds each GPU's
per-transfer mean. Example, all 8 GPUs, WRITE, 20 transfers each:

```bash
export NIXL_GPUS="0 1 2 3 4 5 6 7" NIXL_OP=WRITE NIXL_TEST_ARGS="--iters 20"
```

Measured on 2 x p5en.48xlarge (image `eeaa6801`), 1 GiB per transfer, 3 runs each, all `verify=OK`:

| GPUs | Op | Transfers/GPU | Per-GPU GB/s | Sum GB/s | Combined (`agg_GBps`) | EFA NICs used |
|---|---|---|---|---|---|---|
| 1 | READ | 5 | 47.2 | 47.2 | 33 | 2 |
| 1 | WRITE | 5 | 46.7 | 46.7 | 33 | 2 |
| 8 | READ | 20 | 47.8 | 382.5 | 300 | 16 |
| 8 | WRITE | 20 | 47.8 | 382.2 | 304 | 16 |

382 GB/s is about 3.06 Tb/s of the node's 3.2 Tb/s (16 x 200 Gb/s). `agg_GBps` is lower because it
includes the per-transfer verify and notification round-trip between transfers.

The EFA byte counters read by `rdma statistic` wrap at 2^32 (4 GiB). The byte total is exact only
while each NIC moves under 4 GiB per run (the defaults move 2.5 GiB per NIC). Above that, check that
every expected NIC (2 per GPU) shows traffic.

### Success markers (what a pass looks like)

| Suite | Pass condition |
|---|---|
| NCCL | every rank logs `NET/OFI Selected provider is efa … (found 16 nics)`; busbw reported; exit 0 |
| DeepEP V2 | `Loaded gin plugin Libfabric_GDAKI`; no `host_register_iomem failed`; no `Failed to open gdr handle` (the workers run `privileged: true` so `/dev/gdrdrv` is in the pod); dispatch + combine complete on all 16 ranks |
| NIXL | `NIXL backends created: ['LIBFABRIC']` and `LIBFABRIC backend selected`; `verify=OK` on every transfer; `NIXL_LF_RESULT PASS`; `NIXL_LF_RUN_DONE ... rc=0` on both pods; EFA `rdma_read_bytes` (in the `EFA_COUNTERS` lines) rises by 5 GiB per iteration (defaults; with `NIXL_OP=WRITE` the initiator's `rdma_write_bytes` rises instead, and with N GPUs it is N times as much) |

Throughput floors for sign-off live in `validation-gate.yaml`; the gate compares
a 3-run mean against each floor and blocks a regression.

See `docs/deepep-nixl-pipeline-design.md` for the full design, the six build fixes,
and the publish/provenance flow.
