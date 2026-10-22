# DeepEP V2 + NIXL + EFA container

A reproducible, cluster-validated container image that puts **two** GPU
expert-parallel transports on **one** EFA (Elastic Fabric Adapter) userspace
stack:

- **DeepEP V2** — MoE dispatch/combine over **NCCL GIN** (GPUDirect Async
  kernel-initiated RDMA; aws-ofi-nccl EFA-GDA plugin, type 5).
- **NIXL v1.5.0** — KV-cache / tensor transfer over its **LIBFABRIC** backend
  (the same EFA libfabric) and its **UCX** backend.

Both ride the same **EFA installer 1.50.0** userspace, so a consumer runs the
image with **zero per-run environment** — EFA/RDMA is used by default.

The [`Dockerfile`](./Dockerfile) is written as a **template**: each technology
has its own clearly labelled section, so you can lift one integration (NIXL,
UCX, nccl-tests) into your own image.

---

## Image contents

| Component | Version | Notes |
|---|---|---|
| CUDA base | `nvidia/cuda:13.0.2-devel-ubuntu22.04` | `-devel` (nvcc present) |
| EFA installer | 1.50.0 | ships the EFA-GDA aws-ofi-nccl GIN plugin + libfabric >=2.6 |
| NCCL | 2.31.2 (GIN-capable) | exports `ncclCommQueryProperties` |
| GDRCopy | v2.5.2 | GPUDirect RDMA |
| DeepEP | `amazon-contributing/DeepEP@main` | V2, NCCL GIN backend |
| NIXL | v1.5.0 | LIBFABRIC + UCX + POSIX plugins |
| UCX | v1.23.x | NIXL UCX backend |
| PyTorch | 2.11.0 (`cu130`) | |
| nccl-tests | `master` | `all_reduce_perf`, `all_gather_perf`, `alltoall_perf` on `PATH` |
| nixl_test | from NIXL `test/nixl_test.cpp` | at `/opt/nixl-tests/nixl_test` |
| GPU arch | sm_90;sm_100;sm_103 (Hopper H100/H200 + Blackwell B200/B300 + Grace-Blackwell GB200/GB300) | override `TORCH_CUDA_ARCH_LIST` / `NVCC_GENCODE` to trim to one arch |

---

## Why it is built the way it is (design decisions)

These are the non-obvious choices. Each was learned the hard way; keep them
unless you know why you are changing one.

### 1. Single-layer squash (`COPY --from=builder / /`)
`torch==2.11.0` pulls a **GIN-less** `nvidia-nccl-cu13 2.28.9` as a dependency.
Re-pinning NCCL 2.31.2 in a later layer updated pip metadata but the exported
image still shipped the stale `libnccl.so.2`: BuildKit's content-addressed
export kept **deduping** the committed `.so` back to the older layer blob (seen
across build iterations, even with `--no-cache`). Flattening the whole builder
tree into **one** layer leaves exactly one copy of every file — the builder's
final, verified state — so there is no lower-layer blob to resolve back to.

### 2. Binary-level self-tests, not pip metadata
The build asserts the committed `libnccl.so.2` **exports the GIN symbol**
(`nm -D ... | grep ncclCommQueryProperties`), that `NCCL_NET_PLUGIN` resolves,
that `libgdrapi.so` is on the loader path, and that `import deep_ep` /
`import nixl` work — all with zero special env, exactly what a consumer sees. A
green build is already partly validated; a stale `.so` fails the build instead
of shipping.

### 3. `NCCL_NET_PLUGIN` is an **absolute** path
The aws-ofi-nccl plugin ships as `libnccl-net-ofi.so`, but NCCL 2.31.2
auto-discovers only the canonical `libnccl-net.so`. Without the absolute path
NCCL **silently** falls back to the TCP Socket transport (no EFA/RDMA). The
image bakes `NCCL_NET_PLUGIN=/opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so`.

### 4. Runtime kernel mode = `unordered`
DeepEP V2 `ordered` kernels demand GIN **strong + VA signals**, which
aws-ofi-nccl on EFA does **not** support; `unordered` needs only weak/counting
signals, which EFA does support. Run with `NCCL_SYM_GIN_KERNELS_ENABLE=0`
(fails otherwise with *"GIN strong signals are required, but the GIN plugin does
not support them"*).

### 5. Node driver prerequisite (host, not image)
GDAKI (the GIN GPUDirect-Async backend) registers a GPU MMIO region that
**requires** two NVIDIA kernel-module flags on the host:
`NVreg_EnableStreamMemOPs=1` and `NVreg_RegistryDwords="PeerMappingOverride=1;"`.
Without them GIN fails `DevComm setup failed on all available backends` on any
node count. Bake these into the **nodegroup launch-template userData** (a
per-instance reboot does not persist — the autoscaler replaces the node and it
comes back un-flagged). Verify on a node:

```bash
grep -E "EnableStreamMemOPs|RegistryDwords" /proc/driver/nvidia/params
# want: EnableStreamMemOPs: 1   and RegistryDwords containing PeerMappingOverride=1
```

On a custom AMI that loads `nvidia` from initramfs, the userData must rebuild
initramfs (`dracut --force` / `update-initramfs -u`), not just drop a
`modprobe.d` file.

### 6. EFA kernel driver >= 3.3.0
GIN needs the EFA completion-counter API (`efadv_create_comp_cntr`), first in
EFA driver 3.3.0. Stock AMIs may ship older. Verify: `modinfo efa | grep ^version`.

---

## Dockerfile sections (template)

| Section | Purpose | Lift it if you want... |
|---|---|---|
| **Base + ARGs** | CUDA 13.0.2 devel; every version is an `ARG` | a different CUDA / GPU arch |
| **awslabs base (verbatim)** | EFA 1.50.0, GIN NCCL 2.31.2, GDRCopy, torch, DeepEP V2 | the DeepEP V2 / EFA-GDA stack — kept byte-identical to [`reference/deepep.Dockerfile`](./reference/deepep.Dockerfile) so it stays in sync with upstream |
| **nccl-tests** | `all_reduce/all_gather/alltoall` perf tests, MPI-enabled | NCCL EFA transport benchmarks |
| **UCX** | UCX 1.23.x with CUDA + GDRCopy + verbs | NIXL's UCX backend |
| **NIXL v1.5.0** | meson build, `enable_plugins=UCX,LIBFABRIC,POSIX`, pointed at the EFA libfabric; builds + preserves `nixl_test` | the NIXL + EFA integration |
| **GIN NCCL re-pin + self-test** | re-asserts NCCL 2.31.2 at the `.so` level | the dedup guard |
| **Squash stage** | `COPY --from=builder / /` | the single-layer fix |
| **Runtime ENV + final gate** | EFA/GIN env + the zero-env contract | the "works out of the box" behaviour |

Integration contracts worth copying verbatim:

- **EFA -> NCCL:** `NCCL_NET_PLUGIN=/opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so` (absolute).
- **EFA -> NIXL:** meson `-Dlibfabric_path=/opt/amazon/efa` so the LIBFABRIC
  backend links the EFA libfabric; `enable_plugins` **errors** loudly if it is
  not found.
- **GPUDirect RDMA:** `/opt/gdrcopy/lib` registered via `/etc/ld.so.conf.d` +
  `ldconfig` (ldconfig does not read `LD_LIBRARY_PATH`).

---

## Recreate the container

### Build

Multi-arch default — Hopper (H100/H200) + Blackwell (B200/B300) + Grace-Blackwell (GB200/GB300):

```bash
cd architectures/containers/deepv2-nixl-efa
DOCKER_BUILDKIT=1 docker build --progress=plain -f Dockerfile \
  --build-arg TORCH_CUDA_ARCH_LIST="9.0;10.0;10.3" \
  --build-arg NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_100,code=sm_100 -gencode=arch=compute_103,code=sm_103" \
  -t <registry>/deepv2-nixl-efa:nixl1.5.0-deepepv2-efa1.50.0 .
```

Compute-capability map: `9.0`=Hopper (sm_90), `10.0`=Blackwell DC (sm_100, incl.
GB200), `10.3`=Blackwell Ultra (sm_103, B300/GB300 steppings). GB200/GB300 are
Grace+Blackwell — the GPU die is the same sm_100/sm_103, Grace is the Arm CPU.
Requires CUDA ≥ 12.8 (built on 13.0.2).

To trim to a single arch (smaller image, faster build), e.g. Hopper-only:

```bash
  --build-arg TORCH_CUDA_ARCH_LIST=9.0 \
  --build-arg NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"
```

The in-build self-tests fail the build if the GIN NCCL `.so` is stale, a
backend plugin is missing, or a test binary did not build.

### Build arguments (traceability)

Every version is a pinned `ARG`; these plus the git SHA of this directory fully
determine the image.

| ARG | Default | Role |
|---|---|---|
| `CUDA_VERSION` | `13.0.2` | base image tag |
| `EFA_INSTALLER_VERSION` | `1.50.0` | EFA userspace + GIN plugin + libfabric |
| `NCCL_VERSION` | `2.31.2` | GIN-capable NCCL |
| `GDRCOPY_VERSION` | `v2.5.2` | GPUDirect RDMA |
| `DEEPEP_REF` | `main` | DeepEP V2 source |
| `NIXL_REF` | `v1.5.0` | NIXL source |
| `UCX_VERSION` | `v1.23.x` | NIXL UCX backend |
| `TORCH_VERSION` | `2.11.0` | PyTorch |
| `NCCL_TESTS_REF` | `master` | nccl-tests |
| `TORCH_CUDA_ARCH_LIST` | `9.0;10.0;10.3` | GPU arch (Hopper+Blackwell+GB) |
| `NVCC_GENCODE` | `...compute_90,sm_90 ...compute_100,sm_100 ...compute_103,sm_103` | codegen |

---

## Validate (2 nodes)

Run on two GPU nodes with EFA, each `nvidia.com/gpu: 8` + `vpc.amazonaws.com/efa:
16`, `SYS_ADMIN`/`IPC_LOCK` caps, and a 200Gi `/dev/shm`. The nodes **must**
carry the driver flags (design decision 5).

### 1. NCCL collectives (EFA transport)

```bash
# all-reduce / all-gather / all-to-all, 16 ranks over 2 nodes
mpirun -np 16 -N 8 --bind-to none \
  -x NCCL_DEBUG=INFO -x NCCL_NET_PLUGIN=ofi -x NCCL_TUNER_PLUGIN=ofi \
  all_reduce_perf  -b 8 -e 8G -f 2 -g 1
mpirun -np 16 -N 8 --bind-to none -x NCCL_NET_PLUGIN=ofi all_gather_perf -b 8 -e 8G -f 2 -g 1
mpirun -np 16 -N 8 --bind-to none -x NCCL_NET_PLUGIN=ofi alltoall_perf   -b 8 -e 8G -f 2 -g 1
```
**Pass:** each rank logs `NET/OFI Selected provider is efa ... (found 16 nics)`,
busbw reported, exit 0.

### 2. DeepEP V2 (NCCL GIN / EFA-GDA)

```bash
NCCL_GIN_TYPE=5  NCCL_SYM_GIN_KERNELS_ENABLE=0  NCCL_NET_PLUGIN=ofi \
TORCHDYNAMO_DISABLE=1  TORCH_COMPILE_DISABLE=1  NVIDIA_GDRCOPY=enabled \
python3 /opt/amazon/deepep/tests/elastic/test_ep.py \
  --test-first-only --num-tokens=8192 --num-topk=8 --num-sms=12
```
**Pass:** `Loaded gin plugin Libfabric_GDAKI`, **no** `host_register_iomem
failed`, **no** `GIN strong signals are required`, dispatch + combine complete
on all 16 ranks. Reference (validated 2026-10-14): dispatch ~265-270 GB/s bus BW
(~82 GB/s wire), combine ~200-250 GB/s bus BW.

> Use `tests/elastic/test_ep.py` (`ElasticBuffer` = GIN backend). The legacy
> `tests/legacy/test_internode.py` exercises the deprecated NVSHMEM/IBGDA path
> and fails on EFA.

### 3. NIXL (`test/nixl_test.cpp`, LIBFABRIC / EFA)

The single tested use case. The binary is baked at `/opt/nixl-tests/nixl_test`
(built from NIXL's `test/nixl_test.cpp`, the single/multi-node nixlAgent API
test). It is a two-process (initiator/target) test — run one process per node
over the LIBFABRIC backend:

```bash
# target (node B)
NIXL_PLUGIN_DIR=/opt/nixl/lib/x86_64-linux-gnu/plugins /opt/nixl-tests/nixl_test   # see --help for role/peer flags

# initiator (node A), pointed at node B
NIXL_PLUGIN_DIR=/opt/nixl/lib/x86_64-linux-gnu/plugins /opt/nixl-tests/nixl_test
```
**Pass:** the LIBFABRIC backend is selected (not UCX-only / not SHM), the
transfer completes across the two nodes, exit 0.

---

## Access the public image

The pipeline publishes the validated image to ECR in the shared account,
region **us-east-2**:

```
159553542841.dkr.ecr.us-east-2.amazonaws.com/smml-containers/deepepv2-nixl-efa:nixl1.5.0-deepepv2-efa1.50.0
```

Pull it (note: the auth token is minted with `--region us-east-1`, but the
registry host and repo are `us-east-2`):

```bash
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin 159553542841.dkr.ecr.us-east-2.amazonaws.com
docker pull 159553542841.dkr.ecr.us-east-2.amazonaws.com/smml-containers/deepepv2-nixl-efa:nixl1.5.0-deepepv2-efa1.50.0
```

The current hand-built dev image (ap-southeast-3, same account) is:

```
159553542841.dkr.ecr.ap-southeast-3.amazonaws.com/paragao/nixl-deepep-efa:nixl1.5.0-deepepv2-efa1.50.0
@sha256:63aace0c36de300934ff20db12fb85cdb4c5b99c828ed27e9f74d90ad24885af
```

Prefer the **digest** for anything long-lived — a tag can be overwritten, a
digest cannot. On an EKS cluster in the same account, nodes pull via the node
IAM role; no pull secret is needed. ECR access is account-wide, not
namespace-scoped.

> A CI pipeline (AWS CodePipeline, us-east-2) builds, scans, cluster-validates
> (cross-region against the cgk cluster in ap-southeast-3), and republishes this
> image with full provenance, then opens this upstream PR. See the pipeline
> design document for the full automation.
