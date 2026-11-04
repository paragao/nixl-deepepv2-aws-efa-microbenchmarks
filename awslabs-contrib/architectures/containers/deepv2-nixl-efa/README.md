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

### Two images, pick one

| Dockerfile | What you get | Use it for |
|---|---|---|
| [`Dockerfile`](./Dockerfile) | DeepEP V2 + NIXL + EFA, torch 2.11.0, Python 3.10 | microbenchmarks (DeepEP, NIXL, NCCL tests) |
| [`Dockerfile.vllm`](./Dockerfile.vllm) | everything above **plus vLLM**, torch 2.14.0, Python 3.12 | serving models with vLLM, incl. disaggregated prefill/decode with `NixlConnector` over EFA |

Both share the same EFA 1.50.0 / GIN NCCL 2.31.2 / NIXL v1.5.0 / DeepEP V2
stack and the same in-build gates. See [vLLM image](#vllm-image-dockerfilevllm) below.

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
| GPU arch | sm_90;sm_100;sm_103;sm_120 (Hopper H100/H200 + Blackwell B200/B300 + Grace-Blackwell GB200/GB300 + RTX PRO 6000 Blackwell on g7e) | override `TORCH_CUDA_ARCH_LIST` / `NVCC_GENCODE` to trim to one arch |

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

### 7. SSH is relaxed for in-cluster `mpirun` (both images)
The awslabs base installs `openssh-server` / `openssh-client` so `mpirun` can
launch ranks on peer pods, and it relaxes host checks so that launch needs no
prompts or pre-seeded keys:

| Setting | File | Effect |
|---|---|---|
| `StrictHostKeyChecking no` | `/etc/ssh/ssh_config` | the client accepts any peer host key |
| `UserKnownHostsFile /dev/null` | `/etc/ssh/ssh_config` | peer host keys are never saved or compared |
| `StrictModes no` | `/etc/ssh/sshd_config` | `sshd` does not check key-file ownership/permissions |

This trades away man-in-the-middle protection for zero-setup multi-node runs.
It is acceptable only on a private, trusted network (a single cluster's pod
network, no public exposure). Do not expose port 22 of these pods outside the
cluster, and do not bake private keys into the image. The image ships no keys
and does not start `sshd` on its own: the MPI Operator `MPIJob` manifests in
[`benchmarks/`](../../../../benchmarks/) generate the keys, mount them as a
Kubernetes `Secret` and start `sshd` on the workers. Pods that run only
NCCL/NIXL/vLLM without `mpirun` never start `sshd`.

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

Multi-arch default — Hopper (H100/H200) + Blackwell (B200/B300) + Grace-Blackwell (GB200/GB300)
+ RTX PRO 6000 Blackwell (g7e.48xlarge):

```bash
cd architectures/containers/deepv2-nixl-efa
DOCKER_BUILDKIT=1 docker build --progress=plain -f Dockerfile \
  --build-arg TORCH_CUDA_ARCH_LIST="9.0;10.0;10.3;12.0" \
  --build-arg NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_100,code=sm_100 -gencode=arch=compute_103,code=sm_103 -gencode=arch=compute_120,code=sm_120" \
  -t <registry>/deepv2-nixl-efa:nixl1.5.0-deepepv2-efa1.50.0 .
```

Compute-capability map: `9.0`=Hopper (sm_90), `10.0`=Blackwell DC (sm_100, incl.
GB200), `10.3`=Blackwell Ultra (sm_103, B300/GB300 steppings), `12.0`=Blackwell RTX
(sm_120, RTX PRO 6000 on g7e). GB200/GB300 are Grace+Blackwell — the GPU die is the
same sm_100/sm_103, Grace is the Arm CPU. sm_120 is a separate die, not sm_100-compatible.
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
| `TORCH_CUDA_ARCH_LIST` | `9.0;10.0;10.3;12.0` | GPU arch (Hopper+Blackwell+GB+RTX/g7e) |
| `NVCC_GENCODE` | `...compute_90,sm_90 ...compute_100,sm_100 ...compute_103,sm_103 ...compute_120,sm_120` | codegen |

### Validated build (`Dockerfile`)

Built from this branch at commit `020e077` (rootless BuildKit, `--no-cache`, 30 min, no build-args,
so the Dockerfile defaults above were used) and pushed as:

```
${AWS_ACCOUNT_ID}.dkr.ecr.${ECR_REGION}.amazonaws.com/${ECR_REPO}:nixl1.5.0-deepepv2-efa1.50.0-sm120
@sha256:091ce165ab2413c43b3096e0a9651dd151d7853be6a5141dae4f3795760acf1d
```

About 10.2 GiB compressed, one squashed layer, linux/amd64. Every in-build gate passed: GIN NCCL
2.31.2 (torch layer, re-pin layer and squashed image), DeepEP V2 installed, nccl-tests built,
LIBFABRIC plugin present, `nixl_test` preserved, hygiene, and the squashed-image import gate.

GPU code per binary (`cuobjdump --list-elf`, verified in a fresh pod pulled by digest):

| Binary | sm_90 | sm_100 | sm_103 | sm_120 |
|---|---|---|---|---|
| DeepEP V2 `deep_ep/_C*.so` | yes | yes | yes | yes |
| nccl-tests `all_reduce_perf`, `all_gather_perf`, `alltoall_perf` | yes | yes | yes | yes |
| NIXL EP `nixl_ep_cu13/nixl_ep_cpp_torch211*.so` | yes | yes | yes | yes |

NIXL's core libraries and its UCX / LIBFABRIC / POSIX plugins carry no device code.
`import deep_ep, nixl` works. sm_120 is checked at the binary level only: it has not yet run on a
g7e (RTX PRO 6000) node.

---

## vLLM image (`Dockerfile.vllm`)

[`Dockerfile.vllm`](./Dockerfile.vllm) is the same recipe with vLLM added on top.
It keeps every gate of the base image (GIN NCCL `.so` check, squash, hygiene,
import checks) and adds its own.

### What changes vs `Dockerfile`

| Item | `Dockerfile` | `Dockerfile.vllm` | Why |
|---|---|---|---|
| Python | 3.10 (jammy) | 3.12 (deadsnakes PPA) | vLLM wheels need Python >= 3.11. `/usr/bin/python3` stays 3.10 for apt; `python3` on `PATH` is 3.12. The PPA is added by hand with a vendored key ([`deadsnakes.asc`](./deadsnakes.asc)), not `add-apt-repository`, so no `software-properties-common` apt Python packages ship |
| PyTorch | 2.11.0 | 2.14.0 | the exact torch the pinned vLLM wheel requires |
| vLLM | – | v0.33.0 from PyPI (same commit as v0.33.0rc3) | contains the FA4 SM90 split-KV fix |
| Lock file | – | [`vllm-requirements.txt`](./vllm-requirements.txt) | vLLM's full dependency closure, installed with `pip --no-deps --require-hashes` |
| Pinning | refs and tags | every input pinned to exact bytes | see [Pinned inputs](#pinned-inputs-dockerfilevllm) |

vLLM and torch pin an older, GIN-less `nvidia-nccl-cu13`. A plain `pip install vllm`
would silently replace the GIN NCCL 2.31.2 that DeepEP V2 is built against. So
the image installs the pinned lock with `--no-deps` (torch, NCCL and vLLM are left
out of it) and then asserts torch and NCCL did not move.

### Why v0.33.0 and not v0.32.0

vLLM v0.32.0 ships FA4 code from just before
[flash-attention#198](https://github.com/vllm-project/flash-attention/pull/198).
On SM90 (H100/H200) its split-KV decode kernel makes one split do all the work.
v0.32.1rc0 and later (including v0.33.0) pin flash-attention `b273a02621`, which has the fix.
Measured on Gemma-4-31B-IT, 8x H200, ISL 14000 / OSL 1000:

| TPOT p50, 1 user | v0.25.1 | v0.32.0 | nightly c50aca14 | v0.33.0 (this image) |
|---|---|---|---|---|
| TP8, FA4 | 5.35 ms | 14.39 ms | 5.37 ms | 5.38 ms |
| TP4 DP2, FA4 | 6.18 ms | 14.85 ms | 6.18 ms | – |

The build fails if the fix is missing (`VLLM_REQUIRE_FA4_SM90_FIX=1`).

### Build

```bash
cd architectures/containers/deepv2-nixl-efa
DOCKER_BUILDKIT=1 docker build --progress=plain -f Dockerfile.vllm \
  --build-arg TORCH_CUDA_ARCH_LIST="9.0;10.0;10.3;12.0" \
  --build-arg NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_100,code=sm_100 -gencode=arch=compute_103,code=sm_103 -gencode=arch=compute_120,code=sm_120" \
  -t <registry>/deepv2-nixl-efa:vllm0.33.0 .
```

Build context needs `Dockerfile.vllm`, `setup_deepep_gin.sh`, `vllm-requirements.txt`,
`build-requirements.txt`, `flashinfer-kernels.txt` and `deadsnakes.asc`.

### vLLM build arguments

| ARG | Default | Role |
|---|---|---|
| `PYTHON_VERSION` | `3.12` | interpreter vLLM runs on |
| `TORCH_VERSION` | `2.14.0` | must equal the vLLM wheel's `torch==` pin |
| `VLLM_WHEEL_URL` | v0.33.0 x86_64 wheel on PyPI (`files.pythonhosted.org`) | the vLLM wheel |
| `VLLM_WHEEL_SHA256` | `4b6bdec4…` | wheel integrity check |
| `VLLM_VERSION` | `0.33.0` | asserted after install |
| `VLLM_REQUIRE_FA4_SM90_FIX` | `1` | fail the build if the FA4 SM90 fix is absent |

To pin a different vLLM: change `VLLM_WHEEL_URL`, `VLLM_WHEEL_SHA256`,
`VLLM_VERSION`, `TORCH_VERSION` and `TORCH_WHEEL_SHA256` together, and regenerate
`vllm-requirements.txt` (and `build-requirements.txt`) with the commands in their
headers; [`trim_lock_hashes.py`](./trim_lock_hashes.py) adds the target-only hashes.

### Pinned inputs (`Dockerfile.vllm`)

A rebuild of the same commit fetches the same bytes, or fails.

| Input | Pinned by | ARG / file |
|---|---|---|
| base image | digest | `CUDA_IMAGE_DIGEST` |
| DeepEP V2 | commit SHA | `DEEPEP_REF` |
| UCX | commit SHA on `v1.23.x` | `UCX_VERSION` |
| nccl-tests | commit SHA | `NCCL_TESTS_REF` |
| GDRCopy, NIXL | tag, asserted to equal a commit SHA | `GDRCOPY_COMMIT`, `NIXL_COMMIT` |
| EFA installer | sha256 | `EFA_INSTALLER_SHA256` |
| deadsnakes PPA key | vendored file + sha256 | `deadsnakes.asc`, `DEADSNAKES_KEY_SHA256` |
| get-pip.py, pip | pypa/get-pip commit + sha256, `pip==` | `GET_PIP_COMMIT`, `GET_PIP_SHA256`, `PIP_VERSION` |
| torch, vLLM wheels | sha256 | `TORCH_WHEEL_SHA256`, `VLLM_WHEEL_SHA256` |
| GIN NCCL, `nixl` meta-wheel | `==` + sha256 | `NCCL_WHEEL_SHA256`, `NIXL_META_SHA256` |
| build tools | `--require-hashes` | `build-requirements.txt` |
| vLLM dependencies | `--require-hashes` | `vllm-requirements.txt` |
| FlashInfer kernels | `--require-hashes`, sm90a/sm100a/sm103a/sm120f only | `flashinfer-kernels.txt` |

Two build changes come with the pins. NIXL's Python bindings build with
`--no-build-isolation` (as NIXL's own `contrib/build-wheel.sh` does), so they use
the hashed build tools and the image's torch instead of the `torch==2.11.*` that
NIXL's `pyproject.toml` asks the isolated env for. FlashInfer kernels install from
`flashinfer-kernels.txt` instead of `flashinfer download-kernels`, which ran an
unhashed `pip install --upgrade` with dependencies and added ~222 MB of kernels for
GPUs this image does not target (sm80/sm89). apt packages themselves are
not version-pinned: the deadsnakes PPA keeps only its latest build.

### License gate (`Dockerfile.vllm`)

[`check_licenses.py`](./check_licenses.py) reads licenses as SPDX expressions
(`A OR B` passes if either side does; `A WITH <exception>` needs the exception under
`allow_exceptions`). For the vLLM closure, [`validation-gate.yaml`](./validation-gate.yaml)
adds aliases for free-text license names, upstream-verified pins for wheels with no
license metadata, and two scoped exceptions:

| Exception | Packages | Licenses | State |
|---|---|---|---|
| `permissive-extra` | detect-installer, email-validator, pillow, regex, supervisor | 0BSD, Unlicense, MIT-CMU, CNRI-Python, Repoze | approved |
| `weak-copyleft-unmodified` | certifi, tqdm, orjson, pycountry | MPL-2.0, LGPL-2.1-only | **pending approval** |

While an exception has `pending_approval: true` the gate prints
`LICENSE GATE: PENDING-APPROVAL` and exits 2. Run its tests with
`python3 -m unittest test_check_licenses.py`.

### Run vLLM

The image has no `ENTRYPOINT` for vLLM; start it explicitly:

```bash
vllm serve <model> --tensor-parallel-size 8 \
  --attention-backend FLASH_ATTN --attention-config '{"flash_attn_version": 4}'
```

Disaggregated prefill/decode across nodes uses NIXL's LIBFABRIC backend over EFA:

```bash
# prefill
vllm serve <model> --tensor-parallel-size 4 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_buffer_device":"cuda"}'
# decode
vllm serve <model> --tensor-parallel-size 4 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_buffer_device":"cuda"}'
```

Request all 16 EFA devices per pod. In the logs expect
`Backend LIBFABRIC was instantiated` once per TP rank and
`Created 16 rails using provider=efa`. Harmless: `nixl-cu13 is not installed`
(NIXL's expert-parallel module, unused by vLLM's KV transfer).

### Validated build

Built from this branch at commit `12a6ae6` (vLLM v0.33.0; rootless BuildKit, `--no-cache`,
default build-args) and pushed as:

```
${AWS_ACCOUNT_ID}.dkr.ecr.${ECR_REGION}.amazonaws.com/${ECR_REPO}:vllm0.33.0-sm120
@sha256:eeaa6801eeea1b24e9f0c42ada6be02e499c8707d092a82d94e1519893aa2662
```

About 10.8 GiB compressed, one squashed layer, linux/amd64, GPU code for sm_90 / sm_100 /
sm_103 / sm_120. The previous validated build (vLLM nightly c50aca14, commit `485c55d`) was
`vllm-nightly-c50aca14-sm120@sha256:8a5867348f9066ac81879658ff95fdbc8ab83c5d5e195d322f2b590f0f523892`.

Every pipeline stage passed (build, arch audit, SBOM + license gate, vLLM smoke, 2-node
validation gate). Every in-build gate passed:

| Gate | Result |
|---|---|
| pinned inputs | every `sha256sum -c`, commit-SHA assert and `--require-hashes` install OK |
| torch-layer and squashed `libnccl.so.2` | GIN 2.31.2, exports `ncclCommQueryProperties` |
| vLLM block | `torch 2.14.0+cu130 \| nvidia-nccl-cu13 2.31.2 \| vllm 0.33.0` |
| FA4 SM90 fix | `OK: FA4 SM90 split-KV fix present` |
| FlashInfer kernels | `flashinfer-cubin 0.7.2.post1`, jit-cache providers sm90a / sm100a / sm103a / sm120f |
| GPU archs | DeepEP V2 `_C`, NIXL EP and nccl-tests carry SASS for sm_90 / sm_100 / sm_103 / sm_120 |
| DeepEP V2 / NIXL | `import deep_ep` OK, LIBFABRIC plugin present, `nixl_test` present |
| hygiene | autocommand, my-test-package, devscripts, patchelf and all apt Python packages absent |

Prebuilt libraries already ship sm_120 SASS: torch 2.14 (`sm_75 ... sm_100, sm_120`), the GIN
NCCL 2.31.2 wheel, nvshmem, and vLLM's own `_C` / `_moe_C` kernels. vLLM's bundled FA2/FA3
(`vllm_flash_attn`) are sm_80 / sm_90a only; on sm_120 use `--attention-backend FLASHINFER`.

GPU check on p5en.48xlarge (8x H200): torch sees 8 GPUs, vLLM uses NCCL 2.31.2, NIXL lists
LIBFABRIC/POSIX/UCX, 32 EFA endpoints, and the server log has no "FlashInfer's precompiled
kernels are missing" line. sm_120 is checked at the binary level only: it has not yet run on a
g7e (RTX PRO 6000) node.

License gate on this image: 0 blocked, `PENDING-APPROVAL` (the same 4 MPL-2.0 / LGPL-2.1 wheels await
sign-off, see [License gate](#license-gate-dockerfilevllm)). vs the nightly build: 1 new package
(`flashinfer-jit-cache-sm120f`, Apache-2.0), 37 version bumps, no licence changes.

vLLM smoke (TP8 FA4, unique seeds, exact request counts, 0 prefix-cache hits), TPOT ms / tok/s:

| Users | v0.33.0 (`eeaa6801`) | nightly build (`8a586734`) | stock vLLM nightly c50aca14 |
|---|---|---|---|
| 1 | 5.38 / 2,634 | 5.37 / 2,640 | 5.37 / 2,643 |
| 32 | 15.02 / 28,468 | 15.02 / 28,355 | 14.93 / 28,448 |

Validation gate ([`validation-gate.yaml`](./validation-gate.yaml), 2 p5en nodes, 16 ranks, mean of 3 runs):

| Metric | Baseline | Floor | Mean | Result |
|---|---|---|---|---|
| NCCL all_reduce busbw @8G | 476.5 | 462.2 | 477.7 | PASS |
| NCCL all_gather busbw @8G | 352.9 | 342.3 | 349.0 | PASS |
| NCCL all_to_all busbw @8G | 83.5 | 81.0 | 83.5 | PASS |
| DeepEP V2 dispatch bus BW | 267.0 | 259.0 | 264.9 | PASS |
| DeepEP V2 combine bus BW | 187.0 | 177.6 | 193.0 | PASS |
| NIXL LIBFABRIC GPU-to-GPU, 1 GiB | – | 3/3 pass | 47.0 GB/s | PASS |

NIXL used only the LIBFABRIC backend; EFA `rdma_read_bytes` rose by exactly the bytes moved.
DeepEP needed `privileged: true` on the workers, because `NVIDIA_GDRCOPY=enabled` did not
expose `/dev/gdrdrv` in the pod on this cluster.

### Measured performance

Gemma-4-31B-IT, ISL 14000 / OSL 1000, client `vllm bench serve` (`--dataset-name random --ignore-eos`).
Mean over 3 seeds of each run's p50. Every run uses a new seed, so prefix-cache hits are 0%.

TP8, `--attention-backend FLASH_ATTN --attention-config '{"flash_attn_version": 4}'`, fp8 weights, bf16 KV
(TTFT ms / TPOT ms / total tok/s):

| Users | Nightly build (`8a586734`) | Unpinned build (`5b009bee`) | Stock `vllm/vllm-openai` nightly c50aca14 | vLLM v0.25.1 | vLLM v0.32.0 |
|---|---|---|---|---|---|
| 1 | 322 / 5.37 / 2,640 | 319 / 5.38 / 2,636 | 320 / 5.37 / 2,643 | 331 / 5.35 / 2,644 | 329 / 14.39 / 1,021 |
| 32 | 1,459 / 15.45 / 28,355 | 1,466 / 15.37 / 28,465 | 1,534 / 15.33 / 28,447 | 1,534 / 15.76 / 27,641 | 1,038 / 23.15 / 19,737 |

The image matches stock vLLM nightly within 0.8% TPOT / 0.4% tok/s (run-to-run noise), and the
pinned build matches the earlier unpinned one within 1%. It also carries NIXL over EFA, which
stock vLLM images cannot do (they ship no libfabric). 0 failed requests in 660.

Disaggregated 1P1D across two p5en nodes, NIXL LIBFABRIC over EFA (`NixlConnector`, FA4, fp8
weights; same seeds 4xxx/5xxx/6xxx in every column). On both sm_120 builds: LIBFABRIC on 8/8 ranks per
pod, 0 UCX, 16 EFA rails, 8 transfers x 393 MB per request, 0 failed / 0 expired transfers:

| Users | TP8 + TP8, v0.33.0 (`eeaa6801`) | TP8 + TP8, nightly build (`8a586734`) | TP8 + TP8, vLLM 0.32.0 | TP4 + TP4, vLLM 0.32.0 |
|---|---|---|---|---|
| 8 | 427 / 6.16 / 17,046 | 429 / 6.16 / 17,040 | 499 / 14.56 / 7,447 | 618 / 15.26 / 7,035 |
| 32 | 453 / 9.30 / 43,911 | 460 / 9.31 / 43,868 | 504 / 16.58 / 25,946 | 637 / 17.75 / 23,205 |
| 128 | 17,256 / 13.38 / 56,683 | 17,142 / 13.38 / 56,809 | 10,635 / 21.04 / 51,216 | 32,817 / 19.14 / 33,273 |

The nightly's FA4 fix cuts decode TPOT by 58% / 44% / 36%, so 1P1D tok/s rises 2.3x / 1.7x / 1.1x at 8 / 32 / 128 users.
At 128 users the faster decode drains requests sooner and the single TP8 prefill becomes the
bottleneck (TTFT 17.1 s); add prefill workers (2P1D) for that load. v0.33.0 matches the nightly
build within 0.2% tok/s at every load (0 failed requests in 1,812).

More workers, same nightly build (`8a586734`), same seeds, one TP8 worker per p5en node. The
proxy round-robins prefills and decodes (50/50 split measured on every role). LIBFABRIC on 8/8
ranks in every pod, 0 UCX, 16 EFA rails, 0 failed / 0 expired transfers, 0 failed requests:

| Users | 1P1D (16 GPUs) | 2P1D (24 GPUs) | 2P2D (32 GPUs) |
|---|---|---|---|
| 8 | 429 / 6.16 / 17,040 | 436 / 6.15 / 16,446 | 425 / 5.64 / 17,492 |
| 32 | 460 / 9.31 / 43,868 | 452 / 9.30 / 46,408 | 452 / 6.96 / 56,182 |
| 128 | 17,142 / 13.38 / 56,809 | 597 / 23.11 / 70,784 | 3,298 / 12.88 / 102,417 |
| tok/s per GPU, 128 users | 3,551 | 2,949 | 3,201 |

- 2P1D fixes the 1P1D prefill bottleneck: TTFT at 128 users drops 29x (17.1 s -> 0.6 s), tok/s +25%.
  Its single decode worker then saturates (TPOT 13.4 -> 23.1 ms).
- 2P2D fixes that: TPOT at 128 users is back to 12.9 ms and tok/s is 102,417 (+45% over 2P1D, 1.8x
  1P1D). TTFT rises to 3.3 s, so the two prefill workers are the limit again at 128 users.
- At 8 users extra workers add nothing; at 32 users only the second decode worker helps (+21% tok/s).

The 2P2D 32-user row is a clean rerun (480/480 requests). The first 2P2D run lost 1 of 480 requests
to an HTTP read error in the benchmark proxy (`toy_proxy_server.py`), with no worker or NIXL error;
its numbers (440 / 6.97 / 57,744) matched the rerun.

Gemma-4-31B is dense, so DeepEP V2 (MoE expert-parallel) is present but not exercised by these tests.

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

## Access the published image

The pipeline publishes the validated image to ECR. Set the environment
variables below (see the repo root `env.example.sh`) and substitute them:

```
${AWS_ACCOUNT_ID}.dkr.ecr.${ECR_REGION}.amazonaws.com/${PUBLISH_REPO}:nixl1.5.0-deepepv2-efa1.50.0
```

Pull it (note: the auth token is minted with `--region ${ECR_TOKEN_REGION}`,
but the registry host and repo are in `${ECR_REGION}` — some accounts require a
specific token-mint region):

```bash
aws ecr get-login-password --region ${ECR_TOKEN_REGION} \
  | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${ECR_REGION}.amazonaws.com
docker pull ${AWS_ACCOUNT_ID}.dkr.ecr.${ECR_REGION}.amazonaws.com/${PUBLISH_REPO}:nixl1.5.0-deepepv2-efa1.50.0
```

A dev image may live in a separate per-user repo in the same account:

```
${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:nixl1.5.0-deepepv2-efa1.50.0
@${IMAGE_DIGEST}
```

Prefer the **digest** for anything long-lived — a tag can be overwritten, a
digest cannot. On an EKS cluster in the same account, nodes pull via the node
IAM role; no pull secret is needed. ECR access is account-wide, not
namespace-scoped.

> A CI pipeline (AWS CodePipeline, `${ECR_REGION}`) builds, scans,
> cluster-validates (cross-region against the EKS cluster in `${AWS_REGION}`),
> and republishes this image with full provenance, then opens this upstream PR.
> See the pipeline design document for the full automation.
