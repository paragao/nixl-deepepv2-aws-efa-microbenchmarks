# DeepEP V2 + NIXL + EFA container

Container images that run two GPU transports on one EFA (Elastic Fabric Adapter)
userspace stack:

- **DeepEP V2**: MoE dispatch/combine over **NCCL GIN** (GPUDirect Async,
  kernel-initiated RDMA; aws-ofi-nccl EFA-GDA plugin, type 5).
- **NIXL v1.5.2**: KV-cache and tensor transfer over its **LIBFABRIC** backend,
  on the same EFA libfabric.

EFA does not support UCX. Every NIXL path in these images is LIBFABRIC only: there
is no UCX build, no UCX plugin built by this recipe, and no NIXL test that needs UCX.

Both images use **EFA installer 1.51.0**. A consumer runs them with no per-run
environment; EFA/RDMA is the default.

### Two images, pick one

| Dockerfile | Base | What you get | Use it for |
|---|---|---|---|
| [`Dockerfile`](./Dockerfile) | `nvidia/cuda:13.0.2-devel-ubuntu22.04`, built from source | DeepEP V2 + NIXL + nccl-tests + OpenMPI, torch 2.11.0, Python 3.10 | microbenchmarks (NCCL tests, DeepEP, NIXL) |
| [`Dockerfile.vllm`](./Dockerfile.vllm) | `vllm/vllm-openai:v0.33.0-x86_64-efa` (by digest) | vLLM 0.33.0 + DeepEP V2 + NIXL LIBFABRIC tests, torch 2.14.0, Python 3.12 | serving with vLLM, incl. disaggregated prefill/decode with `NixlConnector` over EFA |

Both carry NCCL 2.31.2 (GIN), EFA installer 1.51.0, NIXL v1.5.2 (LIBFABRIC) and
DeepEP V2 at `amazon-contributing/DeepEP@d75f22a5`. They differ in how they get
there. See [vLLM image](#vllm-image-dockerfilevllm) below.

---

## Image contents (`Dockerfile`)

| Component | Version | Notes |
|---|---|---|
| CUDA base | `nvidia/cuda:13.0.2-devel-ubuntu22.04` | `-devel` (nvcc present) |
| EFA installer | 1.51.0 | sha256-checked; ships the EFA-GDA aws-ofi-nccl GIN plugin and libfabric |
| NCCL | 2.31.2 (GIN) | pinned once, in the torch layer; exports `ncclCommQueryProperties` |
| GDRCopy | v2.5.2 | GPUDirect RDMA |
| DeepEP V2 | `amazon-contributing/DeepEP@d75f22a5` | NCCL GIN backend |
| NIXL | v1.5.2 (commit `e72c52c`) | `-Denable_plugins=LIBFABRIC` only |
| PyTorch | 2.11.0 (`cu130`) | Python 3.10 |
| nccl-tests | `master` | `all_reduce_perf`, `all_gather_perf`, `alltoall_perf` on `PATH` |
| NIXL tests | from NIXL v1.5.2 | `/opt/nixl-tests`, see below |
| GPU arch | sm_90, sm_100, sm_103, sm_120 | H100/H200, B200/B300, GB200/GB300, RTX PRO 6000 (g7e) |

NIXL tests shipped in `/opt/nixl-tests`:

| Binary | What it checks |
|---|---|
| `libfabric_topology_test` | EFA rail / GPU topology seen by the LIBFABRIC backend. Run it as `libfabric_topology_test p5en.48xl`; with no argument it also runs built-in scenarios for other instance types, which fail on p5en |
| `rail_active_refcount_test` | LIBFABRIC rail reference counting |
| `desc_example` | descriptor lists (no backend) |
| `p2p_test` | point-to-point helpers |
| `serdes_test` | serializer (no backend) |
| `telemetry_reader` | reads NIXL telemetry |

Not built or not shipped, because each hard-codes the UCX backend: `nixl_ep` (NIXL EP),
`nixl_test`, `agent_example`, `nixl_example`.

---

## Why it is built the way it is (design decisions)

These are the non-obvious choices. Keep them unless you know why you are changing one.

### 1. NCCL 2.31.2 goes in with torch, in one layer (no squash)
`torch==2.11.0` pulls a GIN-less `nvidia-nccl-cu13 2.28.9`. Older builds re-pinned NCCL
2.31.2 in a later layer; the 2.28.9 `libnccl.so.2` stayed in the lower layer and the
pushed image kept shipping it. Those builds flattened everything into one layer
(`COPY --from=builder / /`) to get rid of it.

Now the torch `RUN` overwrites 2.28.9 with 2.31.2 in the same layer, so the stale `.so`
is never committed to any layer. The squash is no longer needed and was removed: a test
build without it shipped the byte-identical `libnccl.so.2` (sha256 `78103983...`) at
9.98 GiB in 30 layers, against 9.92 GiB in one squashed layer. The final `nm` gate still
fails the build if a stale `.so` ever comes back.

The vLLM image does not squash. Its base already ships NCCL 2.31.2 and nothing in
`Dockerfile.vllm` installs another NCCL. The build asserts the version did not move.

### 2. Binary-level gates, not pip metadata
The build asserts that the committed `libnccl.so.2` exports the GIN symbol
(`nm -D ... | grep ncclCommQueryProperties`), that the aws-ofi-nccl plugins resolve,
that `libgdrapi.so` is on the loader path, that the LIBFABRIC plugin is present and no
UCX plugin was built, that every NIXL test resolves its libraries and `serdes_test`
runs, and that `import deep_ep` and `import nixl` work. All checks use no special env,
which is what a consumer sees.

### 3. `NCCL_NET_PLUGIN` must resolve to the aws-ofi-nccl plugin
The plugin ships as `libnccl-net-ofi.so`. If NCCL cannot find it, NCCL falls back to
the TCP Socket transport without an error. The base image sets the absolute path
`NCCL_NET_PLUGIN=/opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so`. The vLLM base sets
`NCCL_NET_PLUGIN=ofi` and registers `/opt/amazon/ofi-nccl/lib` with `ld.so.conf.d`;
`Dockerfile.vllm` gates on both plugins being in the loader cache.

### 4. Runtime kernel mode = `unordered`
DeepEP V2 `ordered` kernels need GIN strong + VA signals, which aws-ofi-nccl on EFA does
not support. `unordered` needs only weak/counting signals. Run with
`NCCL_SYM_GIN_KERNELS_ENABLE=0` (otherwise: *"GIN strong signals are required, but the
GIN plugin does not support them"*).

### 5. Node driver prerequisite (host, not image)
GDAKI (the GIN GPUDirect-Async backend) registers a GPU MMIO region that needs two
NVIDIA kernel-module flags on the host: `NVreg_EnableStreamMemOPs=1` and
`NVreg_RegistryDwords="PeerMappingOverride=1;"`. Without them GIN fails
`DevComm setup failed on all available backends` on any node count. Put them in the
nodegroup launch-template userData. A per-instance reboot does not persist: the
autoscaler replaces the node and it comes back without the flags. Verify on a node:

```bash
grep -E "EnableStreamMemOPs|RegistryDwords" /proc/driver/nvidia/params
# want: EnableStreamMemOPs: 1   and RegistryDwords containing PeerMappingOverride=1
```

On a custom AMI that loads `nvidia` from initramfs, the userData must rebuild initramfs
(`dracut --force` / `update-initramfs -u`), not only drop a `modprobe.d` file.

### 6. EFA kernel driver >= 3.3.0
GIN needs the EFA completion-counter API (`efadv_create_comp_cntr`), first in EFA
driver 3.3.0. Stock AMIs may ship older. Verify: `modinfo efa | grep ^version`.

### 7. SSH is relaxed for in-cluster `mpirun` (base image)
The base image installs `openssh-server` / `openssh-client` so `mpirun` can launch ranks
on peer pods, and relaxes host checks so launch needs no prompts or pre-seeded keys:

| Setting | File | Effect |
|---|---|---|
| `StrictHostKeyChecking no` | `/etc/ssh/ssh_config` | the client accepts any peer host key |
| `UserKnownHostsFile /dev/null` | `/etc/ssh/ssh_config` | peer host keys are never saved or compared |
| `StrictModes no` | `/etc/ssh/sshd_config` | `sshd` does not check key-file ownership/permissions |

This trades away man-in-the-middle protection for zero-setup multi-node runs. Use it only
on a private, trusted network (one cluster's pod network, no public exposure). Do not
expose port 22 of these pods outside the cluster, and do not bake private keys into the
image. The image ships no keys and does not start `sshd` itself: the MPI Operator
`MPIJob` manifests in [`benchmarks/`](../../../../benchmarks/) generate the keys, mount
them as a Kubernetes `Secret` and start `sshd` on the workers.

The vLLM image has no OpenMPI and no nccl-tests, so it is not launched with `mpirun`.

---

## How `Dockerfile` is laid out

| Part | What it does |
|---|---|
| awslabs base | Follows [`reference/deepep.Dockerfile`](./reference/deepep.Dockerfile) (vendored unchanged). Changes from it: EFA installer 1.51.0 with a sha256 check, DeepEP pinned to `d75f22a5`, NCCL 2.31.2 pinned once in the torch layer, `debhelper`/`devscripts` dropped |
| nccl-tests | MPI-enabled perf tests built against the GIN NCCL |
| NIXL v1.5.2 | meson build with `-Denable_plugins=LIBFABRIC`, `-Dlibfabric_path=/opt/amazon/efa`; copies the LIBFABRIC and backend-free tests to `/opt/nixl-tests` |
| Runtime ENV + gates | EFA/GIN env, loader config, hygiene, final binary + import gate |

Integration contracts worth copying:

- **EFA -> NCCL:** `NCCL_NET_PLUGIN=/opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so`.
- **EFA -> NIXL:** meson `-Dlibfabric_path=/opt/amazon/efa -Denable_plugins=LIBFABRIC`.
  A requested plugin that cannot be built is a meson error, so a missing libfabric fails
  the build.
- **NIXL plugin dir:** `NIXL_PLUGIN_DIR` is one directory, not a colon list (NIXL
  `opendir()`s the value as is).
- **GPUDirect RDMA:** `/opt/gdrcopy/lib` registered via `/etc/ld.so.conf.d` + `ldconfig`
  (ldconfig does not read `LD_LIBRARY_PATH`).

---

## Recreate the container

### Build

Multi-arch default (Hopper, Blackwell, Grace-Blackwell, RTX PRO 6000 on g7e):

```bash
cd architectures/containers/deepv2-nixl-efa
DOCKER_BUILDKIT=1 docker build --progress=plain -f Dockerfile \
  -t <registry>/deepv2-nixl-efa:nixl1.5.2-deepepv2-efa1.51.0 .
```

Compute-capability map: `9.0` = Hopper (sm_90), `10.0` = Blackwell DC (sm_100, incl.
GB200), `10.3` = Blackwell Ultra (sm_103, B300/GB300), `12.0` = Blackwell RTX (sm_120,
RTX PRO 6000 on g7e). GB200/GB300 use the same GPU die as B200/B300; Grace is the Arm
CPU. sm_120 is a separate die, not sm_100-compatible. Requires CUDA >= 12.8 (built on
13.0.2).

To trim to one arch (smaller image, faster build), e.g. Hopper only:

```bash
  --build-arg TORCH_CUDA_ARCH_LIST=9.0 \
  --build-arg NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"
```

The NIXL arch list (`-Dnixl_cuda_arch_list`) is fixed at `90,100,103,120` in this file.

### Pinned versions (`Dockerfile`)

| ARG | Default | Pinned by |
|---|---|---|
| `CUDA_VERSION` | `13.0.2` | base image tag |
| `EFA_INSTALLER_VERSION` / `EFA_INSTALLER_SHA256` | `1.51.0` / `83993044...` | sha256 of the tarball |
| `NCCL_VERSION` | `2.31.2` | `==` |
| `GDRCOPY_VERSION` | `v2.5.2` | tag |
| `DEEPEP_REF` | `d75f22a5e07c95eb0bae94ad6445a5009c06c886` | commit SHA |
| `NIXL_REF` / `NIXL_COMMIT` | `v1.5.2` / `e72c52c7...` | tag, asserted to equal the commit |
| `TORCH_VERSION` | `2.11.0` (`cu130`) | `==` |
| `PYTHON_VERSION` | `3.10` | Ubuntu 22.04 package |
| `NCCL_TESTS_REF` | `master` | not pinned |
| `TORCH_CUDA_ARCH_LIST` | `9.0;10.0;10.3;12.0` | |
| `NVCC_GENCODE` | sm_90, sm_100, sm_103, sm_120 | |

### Validated build (`Dockerfile`)

Built from commit `e7d45ef` (rootless BuildKit, `--no-cache`, default build-args, 18 min)
and pushed as

```
${AWS_ACCOUNT_ID}.dkr.ecr.${ECR_REGION}.amazonaws.com/${ECR_REPO}:nixl1.5.2-deepepv2-efa1.51.0
@sha256:a8688aadbc2fba04ddf6a7b1033837860b8870eb777284b08a332f7b19ac5916
```

About 10.0 GiB compressed, 30 layers (no squash), linux/amd64. Every in-build gate passed,
including `OK: nccl-tests <b> loads the GIN libnccl.so.2` for all three binaries.
`cuobjdump` in a fresh pod pulled by digest shows SASS for sm_90 / sm_100 / sm_103 /
sm_120 in DeepEP V2 `_C*.so` and the nccl-tests binaries. Every installed `nixl*` package
is 1.5.2. License gate: SIGN-OFF, 0 blocked, 0 pending.

Validation gate, 2 x p5en.48xlarge, 16 ranks, mean of 3 runs (2026-11-06). Every NCCL rank
logs `NCCL version 2.31.2+cuda13.3` and `NET/OFI Selected provider is efa ... (found 16 nics)`:

| Metric | Floor | 3 runs | Mean | Result |
|---|---|---|---|---|
| NCCL all_reduce busbw @8G (GB/s) | 465.1 | 479.54 / 479.81 / 479.00 | 479.5 | PASS |
| NCCL all_gather busbw @8G | 338.8 | 349.40 / 349.12 / 349.43 | 349.3 | PASS |
| NCCL all_to_all busbw @8G | 85.2 | 87.84 / 87.81 / 87.79 | 87.8 | PASS |
| DeepEP V2 dispatch bus BW | 259.0 | 266.1 / 265.9 / 265.8 | 266.0 | PASS |
| DeepEP V2 combine bus BW | 177.6 | 193.2 / 193.0 / 192.4 | 192.9 | PASS |
| NIXL 1 GPU READ (`nixl_lf`) | 45.8 | 46.80 / 46.75 / 46.88 | 46.81 | PASS |
| NIXL 8 GPU WRITE, 20 x 1 GiB per GPU (sum) | 371 | 382.66 / 383.45 / 383.85 | 383.3 | PASS |
| NIXL 8 GPU READ, 20 x 1 GiB per GPU (sum) | (382.5 ref) | 382.90 / 382.80 / 383.07 | 382.9 | PASS |

Every NIXL run used LIBFABRIC only and verified every transfer (160/160 per 8-GPU run,
GPU 0 included) with 0 `post failed` lines. NIXL tests and GDRCopy `sanity` / `copybw` /
`copylat` pass in a GPU pod; the NIXL plugin list is `['LIBFABRIC']`.

**NCCL version in earlier results.** Every earlier NCCL run in this README (images
`091ce165`, `eeaa6801`, and the squashed slim build `7fba91f1`) ran on **NCCL 2.28.3**, not
2.31.2. Their squash stage started from a fresh `nvidia/cuda` image, which ships apt NCCL
2.28.3, and `COPY --from=builder / /` cannot delete it; nccl-tests found it first. DeepEP and
NIXL always used 2.31.2. The NCCL gate floors were set on 2.28.3; 2.31.2 clears them
(all_to_all +5%), and these runs are now the NCCL baseline (floors 465.1 / 338.8 / 85.2,
-3%). The new build gate fails if nccl-tests do not load the 2.31.2
`libnccl.so.2`.

The squashed slim build `7fba91f1` (commit `c8a6180`, 9.9 GiB, one layer) passed the same
gates, with NCCL on 2.28.3 (478.0 / 349.2 / 83.6).

History (not this Dockerfile): the previous version, at commit `020e077` (EFA 1.50.0,
DeepEP `main`, NIXL with UCX + LIBFABRIC + POSIX, NIXL EP and `nixl_test` shipped), was
pushed as

```
${AWS_ACCOUNT_ID}.dkr.ecr.${ECR_REGION}.amazonaws.com/${ECR_REPO}:nixl1.5.0-deepepv2-efa1.50.0-sm120
@sha256:091ce165ab2413c43b3096e0a9651dd151d7853be6a5141dae4f3795760acf1d
```

About 10.2 GiB compressed, one squashed layer, linux/amd64. All in-build gates passed.
`cuobjdump --list-elf` in a fresh pod pulled by digest showed SASS for sm_90 / sm_100 /
sm_103 / sm_120 in DeepEP V2 `_C*.so`, the nccl-tests binaries and NIXL EP. sm_120 was
checked at the binary level only; it has not run on a g7e node.

---

## vLLM image (`Dockerfile.vllm`)

[`Dockerfile.vllm`](./Dockerfile.vllm) starts from vLLM's own EFA image and adds only
what that image lacks.

Base: `vllm/vllm-openai:v0.33.0-x86_64-efa@sha256:5fc20aa179d3f7f83aa4a8b66473edf80582d865c62b630e49a207c76b9a0886`

### What the base already has

| Component | Version |
|---|---|
| vLLM | 0.33.0 |
| PyTorch | 2.14.0+cu130 |
| Python | 3.12 |
| NCCL | 2.31.2 (GIN, `nvidia-nccl-cu13` wheel) |
| EFA installer | 1.51.0: libfabric 2.7.0, aws-ofi-nccl 1.22.0, `NCCL_NET_PLUGIN=ofi` |
| GDRCopy | 2.5.1 (NVIDIA deb) |
| NIXL | 1.5.0 wheel (`nixl-cu13`); replaced by 1.5.2 in `Dockerfile.vllm` (see below) |
| Mooncake | 0.3.13.post1 (EFA) |
| LMCache | 0.5.6 |
| FlashInfer | 0.7.2.post1 |
| DeepEP V2 | 2.1.0+874779c, sm_90a / sm_100a / sm_103a only |

It has no OpenMPI and no nccl-tests. The NIXL wheel also ships a UCX plugin; nothing in
this image selects it.

### What `Dockerfile.vllm` adds

| Item | Detail |
|---|---|
| NIXL 1.5.2 wheels | `nixl-cu13` and `nixl` 1.5.2 from PyPI (sha256-pinned, `--no-deps`) in place of the base's 1.5.0. See [EFA 1.51.0 and NIXL 1.5.2](#efa-1510-and-nixl-152) |
| DeepEP V2 at `d75f22a5` | Rebuilt for sm_90 / sm_100 / sm_103 / sm_120. Carries 9 EFA fixes newer than the base's `874779c` (GIN flush before exit, `gin.flush` timeout, proxy ring wait, hybrid dispatch races) |
| NIXL LIBFABRIC tests | The same six tests as the base image, built from NIXL `e72c52c` (v1.5.2) with `-Denable_plugins=LIBFABRIC` in a discarded stage. Installed in `/opt/nixl-tests` with their libraries in `/opt/nixl` |
| `nixl-test` wrapper | `nixl-test <name> [args]` runs `/opt/nixl-tests/<name>` against `/opt/nixl`. `/opt/nixl` stays off the global loader path, so vLLM keeps using the wheel's libnixl |
| Runtime ENV | `FI_PROVIDER=efa`, `NCCL_GIN_TYPE=5`, `NVIDIA_GDRCOPY=enabled` |
| Build gates | vLLM / torch / NCCL versions unchanged; NCCL GIN symbol; aws-ofi-nccl plugins in the loader cache; `libgdrapi.so.2`; the wheel's LIBFABRIC plugin resolves to the EFA libfabric; NIXL wheels are 1.5.2; `NIXL_ETCD_ENDPOINTS` is unset or non-empty; every NIXL test resolves; `deep_ep.ElasticBuffer` exists; DeepEP `_C*.so` has SASS for every arch |

Files no longer used by either Dockerfile: [`vllm-requirements.txt`](./vllm-requirements.txt),
[`flashinfer-kernels.txt`](./flashinfer-kernels.txt) and [`deadsnakes.asc`](./deadsnakes.asc).
They are kept in the directory for reference. [`trim_lock_hashes.py`](./trim_lock_hashes.py)
is still used to regenerate `build-requirements.txt`.

### Why v0.33.0 and not v0.32.0

vLLM v0.32.0 ships FA4 code from just before
[flash-attention#198](https://github.com/vllm-project/flash-attention/pull/198). On SM90
(H100/H200) its split-KV decode kernel makes one split do all the work. v0.32.1rc0 and
later (including v0.33.0) pin flash-attention `b273a02621`, which has the fix. Measured on
Gemma-4-31B-IT, 8x H200, ISL 14000 / OSL 1000 (earlier builds, see history below):

| TPOT p50, 1 user | v0.25.1 | v0.32.0 | nightly c50aca14 | v0.33.0 (`eeaa6801`) |
|---|---|---|---|---|
| TP8, FA4 | 5.35 ms | 14.39 ms | 5.37 ms | 5.38 ms |
| TP4 DP2, FA4 | 6.18 ms | 14.85 ms | 6.18 ms | – |

### Build

```bash
cd architectures/containers/deepv2-nixl-efa
DOCKER_BUILDKIT=1 docker build --progress=plain -f Dockerfile.vllm \
  -t <registry>/deepv2-nixl-efa:vllm0.33.0-efa .
```

Build context: `Dockerfile.vllm`, `setup_deepep_gin.sh`, `build-requirements.txt`.

### Pinned inputs (`Dockerfile.vllm`)

| ARG / file | Value | Pinned by |
|---|---|---|
| `BASE_IMAGE` | `vllm/vllm-openai:v0.33.0-x86_64-efa` | digest `sha256:5fc20aa1...` |
| `DEEPEP_REF` | `d75f22a5e07c95eb0bae94ad6445a5009c06c886` | commit SHA |
| `NIXL_REF` / `NIXL_COMMIT` | `v1.5.2` / `e72c52c7...` | tag, asserted to equal the commit |
| `NIXL_VERSION` + wheel sha256s | `1.5.2`; `nixl-cu13` `591eda3f...`, `nixl` `e4ef5e16...` | `--require-hashes` |
| `build-requirements.txt` | meson, ninja, pybind11 and friends | `--require-hashes`, NIXL test stage only |
| `TORCH_CUDA_ARCH_LIST` | `9.0;10.0;10.3;12.0` | also sets the NIXL arch list |
| `VLLM_VERSION`, `TORCH_VERSION`, `NCCL_VERSION` | `0.33.0`, `2.14.0`, `2.31.2` | asserted, not installed |

apt packages (`libhwloc-dev`, `pkg-config`, in the discarded stage only) are not
version-pinned. To move to a newer vLLM, change `BASE_IMAGE` and the three asserted
versions together.

### License gate (`Dockerfile.vllm`)

[`check_licenses.py`](./check_licenses.py) reads licenses as SPDX expressions
(`A OR B` passes if either side does; `A WITH <exception>` needs the exception under
`allow_exceptions`). For the vLLM closure, [`validation-gate.yaml`](./validation-gate.yaml)
adds aliases for free-text license names, upstream-verified pins for wheels with no
license metadata, and two scoped exceptions:

| Exception | Packages | Licenses | State |
|---|---|---|---|
| `permissive-extra` | detect-installer, email-validator, pillow, regex, supervisor | 0BSD, Unlicense, MIT-CMU, CNRI-Python, Repoze | approved |
| `weak-copyleft-unmodified` | certifi, tqdm, orjson, pycountry | MPL-2.0, LGPL-2.1-only | approved 2026-11-06 |

The vllm-openai base also needed: a hygiene step that removes setuptools `autocommand`,
the `pkg_resources` test fixture and the system-Python debs `software-properties-common`
pulls in (python3-apt, python3-gi, launchpadlib, lazr, wadllib, blinker, pyparsing), and
pins for `accelerate` (Apache-2.0), `gguf` (MIT) and `mooncake-transfer-engine*`
(Apache-2.0, from the wheel classifier and upstream repo; the wheel has no LICENSE file).
With those, the gate gives 0 BLOCK and SIGN-OFF. An exception with
`pending_approval: true` makes the gate print `LICENSE GATE: PENDING-APPROVAL` and exit 2.
Run its tests with `python3 -m unittest test_check_licenses.py`.

### Run vLLM

Start vLLM explicitly:

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

Request all 16 EFA devices per pod. In the logs expect `Backend LIBFABRIC was
instantiated` once per TP rank and `Created 16 rails using provider=efa`.

Run a NIXL test: `nixl-test libfabric_topology_test` prints the EFA rail layout.

### Validated build (`Dockerfile.vllm`)

Built from commit `c8a6180` (rootless BuildKit, `--no-cache`, 13 min) and pushed as

```
${AWS_ACCOUNT_ID}.dkr.ecr.${ECR_REGION}.amazonaws.com/${ECR_REPO}:vllm0.33.0-efa-slim-nixl152
@sha256:94da4fa48b300546520f4572a3efe92d3314a29899538a9eebd1d3507f629761
```

About 9.9 GiB compressed (the previous source-built `eeaa6801` was 10.8 GiB),
linux/amd64. Every in-build gate passed. DeepEP V2 `_C*.so` has SASS for sm_90 / sm_100 /
sm_103 / sm_120. License gate: 0 blocked; the 4 `weak-copyleft-unmodified` wheels were
approved on 2026-11-06.

Results (2 x p5en.48xlarge, mean of 3 runs, 2026-11-04 / 05):

| Metric | Floor / reference | Mean | Result |
|---|---|---|---|
| DeepEP V2 dispatch bus BW (torchrun, 2 x 8 ranks) | 259.0 | 265.9 | PASS |
| DeepEP V2 combine bus BW | 177.6 | 193.1 | PASS |
| NIXL 1 GPU READ (`nixl_lf`) | 45.8 | 46.91 | PASS |
| NIXL 8 GPU WRITE (sum, GB/s) | 371 | 383.0 | PASS |
| NIXL 8 GPU READ (sum, GB/s) | 382.5 ref | 383.0 | PASS |
| 1P1D TP8 + TP8, 8 / 32 / 128 users (tok/s) | 17,040 / 43,868 / 56,809 | 17,015 / 43,840 / 56,715 | PASS (-0.15% / -0.06% / -0.17%) |
| Smoke TP8 FA4, 1 user TPOT ms / tok/s | 5.38 / 2,634 (`eeaa6801`) | 5.38 / 2,636 | PASS |
| Smoke TP8 FA4, 32 users TPOT ms / tok/s | 15.02 / 28,468 | 15.04 / 28,431 | PASS |

1P1D: 0 failed requests, 0 failed KV transfers (14,976), 0 `post failed` lines,
LIBFABRIC on 8/8 ranks per pod, 16 EFA rails. The smoke numbers are from the build one
commit earlier (`9cf70d6`); the only change since is the removed
`FI_EFA_USE_UNSOLICITED_WRITE_RECV` ENV. There is no NCCL suite: this image has no
OpenMPI and no nccl-tests.

### EFA 1.51.0 and NIXL 1.5.2

With EFA 1.51.0 (libfabric 2.7.0), NIXL 1.5.0 WRITE fails. 8-GPU WRITE lost GPU 0 on
every run (`drainPostQueue: post failed ret=-22 (Invalid argument) on rail 0`, or an EFA
provider abort: `Packet already processed`), and vLLM 1P1D lost its decode engine after
a few hundred KV transfers. EFA 1.50.0 (libfabric 2.6.0) did not show it. Two NIXL bugs,
both fixed in v1.5.2:

| Bug | Fix |
|---|---|
| The progress thread posts and polls rail 0 without the endpoint lock, racing the caller's notification sends; libfabric 2.7 corrupts endpoint state | [ai-dynamo/nixl#2308](https://github.com/ai-dynamo/nixl/pull/2308) |
| The PyPI wheel, built against libfabric 1.21 headers, never turns off EFA unsolicited write recv, which can lose remote write completions | [ai-dynamo/nixl#2494](https://github.com/ai-dynamo/nixl/pull/2494) |

NIXL 1.5.2 logs `Successfully disabled unsolicited write recv for rail N` per endpoint.
Do not also set `FI_EFA_USE_UNSOLICITED_WRITE_RECV=0` for the whole image: it makes
aws-ofi-nccl's GIN `createContext` hang, so DeepEP V2 never starts. Do not set
`NIXL_ETCD_ENDPOINTS=""` either: the NIXL wheel rejects an empty value and no agent
starts. Leave it unset.

History (not this Dockerfile). The previous `Dockerfile.vllm`, at commit `12a6ae6`, built
vLLM v0.33.0 from source on `nvidia/cuda:13.0.2` (deadsnakes Python 3.12, hashed
`vllm-requirements.txt` lock, UCX + LIBFABRIC NIXL, one squashed layer). It was pushed as

```
${AWS_ACCOUNT_ID}.dkr.ecr.${ECR_REGION}.amazonaws.com/${ECR_REPO}:vllm0.33.0-sm120
@sha256:eeaa6801eeea1b24e9f0c42ada6be02e499c8707d092a82d94e1519893aa2662
```

About 10.8 GiB compressed, linux/amd64. Every pipeline stage passed (build, arch audit,
SBOM + license gate, vLLM smoke, 2-node validation gate). The build before it (vLLM
nightly c50aca14, commit `485c55d`) was
`vllm-nightly-c50aca14-sm120@sha256:8a5867348f9066ac81879658ff95fdbc8ab83c5d5e195d322f2b590f0f523892`.

Results on `eeaa6801`:

- License gate: 0 blocked, `PENDING-APPROVAL` (the 4 MPL-2.0 / LGPL-2.1 wheels above).
- vLLM smoke (TP8 FA4, unique seeds, 0 prefix-cache hits), TPOT ms / tok/s: 1 user
  5.38 / 2,634; 32 users 15.02 / 28,468.
- Validation gate (2 p5en nodes, 16 ranks, mean of 3 runs):

| Metric | Baseline | Floor | Mean | Result |
|---|---|---|---|---|
| NCCL all_reduce busbw @8G (NCCL 2.28.3, see above; old floors) | 476.5 | 462.2 | 477.7 | PASS |
| NCCL all_gather busbw @8G | 352.9 | 342.3 | 349.0 | PASS |
| NCCL all_to_all busbw @8G | 83.5 | 81.0 | 83.5 | PASS |
| DeepEP V2 dispatch bus BW | 267.0 | 259.0 | 264.9 | PASS |
| DeepEP V2 combine bus BW | 187.0 | 177.6 | 193.0 | PASS |
| NIXL LIBFABRIC GPU-to-GPU, 1 GiB | – | 3/3 pass | 47.0 GB/s | PASS |

DeepEP needed `privileged: true` on the workers, because `NVIDIA_GDRCOPY=enabled` did
not expose `/dev/gdrdrv` in the pod on this cluster.

### Measured performance (history)

These numbers come from the earlier source-built images (`8a586734`, `5b009bee`), not
from the current `Dockerfile.vllm`.

Gemma-4-31B-IT, ISL 14000 / OSL 1000, client `vllm bench serve` (`--dataset-name random
--ignore-eos`). Mean over 3 seeds of each run's p50. Every run uses a new seed, so
prefix-cache hits are 0%.

TP8, `--attention-backend FLASH_ATTN --attention-config '{"flash_attn_version": 4}'`, fp8
weights, bf16 KV (TTFT ms / TPOT ms / total tok/s):

| Users | Nightly build (`8a586734`) | Unpinned build (`5b009bee`) | Stock `vllm/vllm-openai` nightly c50aca14 | vLLM v0.25.1 | vLLM v0.32.0 |
|---|---|---|---|---|---|
| 1 | 322 / 5.37 / 2,640 | 319 / 5.38 / 2,636 | 320 / 5.37 / 2,643 | 331 / 5.35 / 2,644 | 329 / 14.39 / 1,021 |
| 32 | 1,459 / 15.45 / 28,355 | 1,466 / 15.37 / 28,465 | 1,534 / 15.33 / 28,447 | 1,534 / 15.76 / 27,641 | 1,038 / 23.15 / 19,737 |

The build matched stock vLLM nightly within 0.8% TPOT / 0.4% tok/s (run-to-run noise).
0 failed requests in 660.

Disaggregated 1P1D across two p5en nodes, NIXL LIBFABRIC over EFA (`NixlConnector`, FA4,
fp8 weights; same seeds in every column). LIBFABRIC on 8/8 ranks per pod, 0 UCX, 16 EFA
rails, 8 transfers x 393 MB per request, 0 failed / 0 expired transfers:

| Users | TP8 + TP8, nightly build (`8a586734`) | TP8 + TP8, vLLM 0.32.0 | TP4 + TP4, vLLM 0.32.0 |
|---|---|---|---|
| 8 | 429 / 6.16 / 17,040 | 499 / 14.56 / 7,447 | 618 / 15.26 / 7,035 |
| 32 | 460 / 9.31 / 43,868 | 504 / 16.58 / 25,946 | 637 / 17.75 / 23,205 |
| 128 | 17,142 / 13.38 / 56,809 | 10,635 / 21.04 / 51,216 | 32,817 / 19.14 / 33,273 |

The FA4 fix cuts decode TPOT by 58% / 44% / 36%, so 1P1D tok/s rises 2.3x / 1.7x / 1.1x
at 8 / 32 / 128 users. At 128 users the single TP8 prefill becomes the bottleneck (TTFT
17.1 s); add prefill workers for that load.

More workers, same build (`8a586734`), same seeds, one TP8 worker per p5en node. The
proxy round-robins prefills and decodes. LIBFABRIC on 8/8 ranks in every pod, 0 UCX, 16
EFA rails, 0 failed / 0 expired transfers, 0 failed requests:

| Users | 1P1D (16 GPUs) | 2P1D (24 GPUs) | 2P2D (32 GPUs) |
|---|---|---|---|
| 8 | 429 / 6.16 / 17,040 | 436 / 6.15 / 16,446 | 425 / 5.64 / 17,492 |
| 32 | 460 / 9.31 / 43,868 | 452 / 9.30 / 46,408 | 452 / 6.96 / 56,182 |
| 128 | 17,142 / 13.38 / 56,809 | 597 / 23.11 / 70,784 | 3,298 / 12.88 / 102,417 |
| tok/s per GPU, 128 users | 3,551 | 2,949 | 3,201 |

- 2P1D fixes the 1P1D prefill bottleneck: TTFT at 128 users drops 29x (17.1 s -> 0.6 s),
  tok/s +25%. Its single decode worker then saturates (TPOT 13.4 -> 23.1 ms).
- 2P2D fixes that: TPOT at 128 users is back to 12.9 ms and tok/s is 102,417 (+45% over
  2P1D, 1.8x 1P1D). TTFT rises to 3.3 s, so the two prefill workers are the limit again.
- At 8 users extra workers add nothing; at 32 users only the second decode worker helps
  (+21% tok/s).

The 2P2D 32-user row is a clean rerun (480/480 requests). The first 2P2D run lost 1 of
480 requests to an HTTP read error in the benchmark proxy (`toy_proxy_server.py`), with
no worker or NIXL error; its numbers (440 / 6.97 / 57,744) matched the rerun.

Gemma-4-31B is dense, so DeepEP V2 is present but not exercised by these tests.

---

## Validate (2 nodes)

The gate is [`validation-gate.yaml`](./validation-gate.yaml): three suites, 3 runs each,
on 2 p5en.48xlarge nodes. The manifests are in the repo root
[`benchmarks/`](../../../../benchmarks/); the root [`README.md`](../../../../README.md)
has the apply commands. Nodes need `nvidia.com/gpu: 8`, `vpc.amazonaws.com/efa: 16`,
`SYS_ADMIN`/`IPC_LOCK`, a large `/dev/shm`, and the driver flags (design decision 5).

### 1. NCCL collectives (EFA transport)

`benchmarks/nccl/nccl-tests-validate.yaml`, 16 ranks over 2 nodes:

```bash
mpirun -np 16 -N 8 --bind-to none \
  -x NCCL_DEBUG=INFO -x NCCL_NET_PLUGIN=ofi -x NCCL_TUNER_PLUGIN=ofi \
  all_reduce_perf  -b 8 -e 8G -f 2 -g 1
mpirun -np 16 -N 8 --bind-to none -x NCCL_NET_PLUGIN=ofi all_gather_perf -b 8 -e 8G -f 2 -g 1
mpirun -np 16 -N 8 --bind-to none -x NCCL_NET_PLUGIN=ofi alltoall_perf   -b 8 -e 8G -f 2 -g 1
```

**Pass:** each rank logs `NET/OFI Selected provider is efa ... (found 16 nics)`, busbw at
or above the floor, exit 0. nccl-tests and `mpirun` are in the base image only.

### 2. DeepEP V2 (NCCL GIN / EFA-GDA)

`benchmarks/deepep/deepep-validate.yaml`:

```bash
NCCL_GIN_TYPE=5  NCCL_SYM_GIN_KERNELS_ENABLE=0  NCCL_NET_PLUGIN=ofi \
TORCHDYNAMO_DISABLE=1  TORCH_COMPILE_DISABLE=1  NVIDIA_GDRCOPY=enabled \
python3 /opt/amazon/deepep/tests/elastic/test_ep.py \
  --test-first-only --num-tokens=8192 --num-topk=8 --num-sms=12
```

**Pass:** `Loaded gin plugin Libfabric_GDAKI`, no `host_register_iomem failed`, no `GIN
strong signals are required`, dispatch + combine complete on all 16 ranks, bus BW at or
above the floor (dispatch 259.0, combine 177.6 GB/s).

Use `tests/elastic/test_ep.py` (`ElasticBuffer` = GIN backend). The legacy
`tests/legacy/test_internode.py` exercises the deprecated NVSHMEM/IBGDA path and fails on
EFA. The `/opt/amazon/deepep` source tree is in the base image; the vLLM image removes
its DeepEP build tree after install.

### 3. NIXL over LIBFABRIC (`nixl_lf`)

`benchmarks/nixl/nixl-lf-target-validate.yaml` and `nixl-lf-initiator-validate.yaml` run
`benchmarks/nixl/nixl_lf_test.py` (via `nixl-lf-run.sh`, from a ConfigMap). It creates
only the LIBFABRIC backend and reads 1 GiB GPU-to-GPU from the target, 5 transfers per
run, 3 runs. This replaces NIXL's `nixl_test`, which hard-codes UCX; the four
`nixl-test-*.yaml` manifests are deleted.

**Pass:** `NIXL backends created: ['LIBFABRIC']` and `LIBFABRIC backend selected`,
`verify=OK` on 5/5 transfers per run, `NIXL_LF_RESULT PASS` and rc=0 on both pods, EFA
`rdma_read_bytes` rising by 5 GiB per run.

| Metric | Baseline | Floor | Context (not gated) |
|---|---|---|---|
| 1 GPU READ, mean GB/s | 47.2 | 45.8 (-3%) | 8 GPUs: 382.5 GB/s READ, 382.2 GB/s WRITE (sum) |

Baseline from image `eeaa6801`, 2 x p5en, 2026-11-03. The 8-GPU cases use
`NIXL_GPUS="0 1 2 3 4 5 6 7" NIXL_TEST_ARGS="--iters 20"`.

---

## Access the published image

The pipeline publishes the validated image to ECR. Set the environment variables below
(see the repo root `env.example.sh`) and substitute them:

```
${AWS_ACCOUNT_ID}.dkr.ecr.${ECR_REGION}.amazonaws.com/${PUBLISH_REPO}:nixl1.5.2-deepepv2-efa1.51.0
```

The validated dev images are listed under [Validated build](#validated-build-dockerfile)
and [Validated build (`Dockerfile.vllm`)](#validated-build-dockerfilevllm). Earlier
images carry `efa1.50.0` tags.

Pull it (the auth token is minted with `--region ${ECR_TOKEN_REGION}`, but the registry
host and repo are in `${ECR_REGION}`; some accounts require a specific token-mint region):

```bash
aws ecr get-login-password --region ${ECR_TOKEN_REGION} \
  | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${ECR_REGION}.amazonaws.com
docker pull ${AWS_ACCOUNT_ID}.dkr.ecr.${ECR_REGION}.amazonaws.com/${PUBLISH_REPO}:nixl1.5.2-deepepv2-efa1.51.0
```

A dev image may live in a separate per-user repo in the same account:

```
${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:nixl1.5.2-deepepv2-efa1.51.0
@${IMAGE_DIGEST}
```

Prefer the digest for anything long-lived: a tag can be overwritten, a digest cannot. On
an EKS cluster in the same account, nodes pull via the node IAM role; no pull secret is
needed. ECR access is account-wide, not namespace-scoped.

A CI pipeline (AWS CodePipeline, `${ECR_REGION}`) builds, scans, cluster-validates
(cross-region against the EKS cluster in `${AWS_REGION}`) and republishes this image with
provenance, then opens the upstream PR. See
[`docs/deepep-nixl-pipeline-design.md`](../../../../docs/deepep-nixl-pipeline-design.md).
