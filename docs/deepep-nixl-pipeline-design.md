# DeepEP V2 + EFA + NIXL Container Build Pipeline — Design Document

**Status:** Design / for review
**Owner:** paragao
**Last updated:** 2026-11-04 (branch `refactor/slim-dockerfiles`: LIBFABRIC-only NIXL, EFA 1.51.0, vLLM image on `vllm/vllm-openai:v0.33.0-x86_64-efa`)
**Audience:** anyone building GPU-collective container images (DeepEP V2, NIXL) on AWS and wanting a reproducible, scanned, cluster-validated build that ends in an upstream contribution.

> **Note on repo location:** this doc uses `awslabs/awsome-distributed-ai` as a
> worked example, but the pattern does **not** require that repo. The Dockerfile
> may live in **any git repo that is accessible** (public or org-internal) as
> long as it follows the pattern in §7.4: the repo points at where the image is
> published, and the registry points back at the repo.

---

## 1. Purpose

Automate, make traceable, and make reproducible the build of the
`nixl-deepep-efa` container image, which integrates three technologies on one
EFA (Elastic Fabric Adapter) userspace stack:

- **DeepEP V2** — Mixture-of-Experts dispatch/combine over **NCCL GIN** (GPUDirect
  Async kernel-initiated RDMA, aws-ofi-nccl plugin **type 5**, EFA-GDA path).
- **NIXL v1.5.2** — KV-cache / tensor transfer over its **LIBFABRIC** backend
  (same EFA libfabric). EFA does not support UCX, so LIBFABRIC is the only NIXL
  backend built, shipped or tested.
- **EFA installer 1.51.0** (sha256-checked) — `>=1.50` ships the EFA-GDA
  aws-ofi-nccl GIN plugin and the libfabric both transports link against. 1.51.0
  is the version vLLM's own EFA image uses.

Two images come out of the recipe: the base `Dockerfile` (built from
`nvidia/cuda:13.0.2-devel`, torch 2.11.0, Python 3.10) and `Dockerfile.vllm`
(built on `vllm/vllm-openai:v0.33.0-x86_64-efa`, pinned by digest). See §4.

The pipeline must deliver seven things, in order:

1. **Build** the image from a traceable Dockerfile with explicitly pinned versions.
2. **Scan** the image for CVEs and fail on HIGH/CRITICAL.
3. **SBOM + license** — generate an `SBOM.txt` for every image (a hard
   prerequisite for publishing) and gate it: every license in the SBOM must be
   on the **AWS OSS allow-list**; a disallowed or unknown license BLOCKS publish.
4. **Validate** the build on the shared GPU cluster with real 2-node collective tests.
5. **Publish** the validated image to ECR with full provenance (digest, SBOM,
   labels), **self-documentation** (what the image does + the command to run it),
   and a **back-pointer to the source git repo**.
6. **Register the cross-pointer** — the git repo records where the image is
   published, and the ECR repo description points back at the git repo.
7. **Open a PR** to the source git repo's upstream (e.g.
   `awslabs/awsome-distributed-ai`, or wherever the recipe lives) contributing
   the recipe.

---

## 2. Design decisions (and why)

| # | Decision | Rationale | Alternatives rejected |
|---|---|---|---|
| D1 | **Pipeline engine = AWS CodePipeline** in account `${AWS_ACCOUNT_ID}`, region **${ECR_REGION}** (ECR repo home) | Shared account standard; keeps build + ECR + validation IAM in one account/region; no GitHub-hosted runner egress to the cluster | GitHub Actions (needs a self-hosted runner or kubeconfig secret to reach the cluster — more moving parts outside the account) |
| D1a | **ECR auth call uses `--region ${ECR_TOKEN_REGION}`, but the repo and image live in `${ECR_REGION}`** | `aws ecr get-login-password` is only minting an auth token; the login/push registry host (`...dkr.ecr.${ECR_REGION}...`) and the repo region are `${ECR_REGION}`. The ${ECR_TOKEN_REGION} auth region is an account-specific requirement for the token mint. | Minting the token in ${ECR_REGION} — not used here per the account's auth setup |
| D2 | **Build engine = AWS CodeBuild** with a privileged Linux GPU-less compute | CodePipeline's native build provider; supports Docker/BuildKit; can assume a role to push to ECR | In-cluster rootless BuildKit (current method) — kept as a documented fallback (§8.4), but CodeBuild gives CI isolation and logs |
| D3 | **NCCL 2.31.2 is pinned in the torch layer; no squash** | Build iterations v4–v12 shipped the stale GIN-less 2.28.9 `libnccl.so.2` from a lower layer, and a `COPY --from=builder / /` squash was the fix. Since torch and NCCL 2.31.2 now go in in one `RUN`, 2.28.9 is never committed to a layer, so the squash was removed (2026-11-06: a no-squash build ships the byte-identical `.so`, 9.98 vs 9.92 GiB). The final `nm` gate remains. The vLLM image does not squash: its base already ships NCCL 2.31.2 and nothing on top installs another | Dockerfile-level re-pin, dedicated final layer, `--no-cache` alone — all defeated by multi-layer export dedup |
| D4 | **Validation runs on the shared EKS cluster** (`${EKS_CLUSTER}`, p5en.48xlarge H200 nodes), triggered by CodeBuild applying k8s manifests | The collective tests need real EFA + GPUDirect hardware; CodeBuild has no GPUs. CodeBuild assumes a role, runs `kubectl apply`, waits, reads logs, gates | Running tests in CodeBuild directly (no GPUs); a separate manual step (loses the gate) |
| D5 | **Node driver prerequisite baked into the nodegroup launch template** (`NVreg_EnableStreamMemOPs=1`, `NVreg_RegistryDwords="PeerMappingOverride=1;"`) | GDAKI (GIN GPUDirect-Async backend) registers a GPU MMIO region (`host_register_iomem`) that **requires** these NVIDIA kernel-module flags. Without them GIN fails `DevComm setup failed on all available backends` on **any** node count | Per-instance reboot (nodes get replaced by the autoscaler and come back un-flagged — a loop) |
| D6 | **Tests = NCCL (all-reduce, all-gather, all-to-all) + DeepEP V2 + NIXL**, all **2-node / 16-rank** | NCCL proves the EFA transport end to end; DeepEP V2 proves the GIN/EFA-GDA path; NIXL proves the LIBFABRIC/EFA path. 2 nodes is the minimum that exercises **internode** RDMA | Single-node (never touches the wire); 1 collective only (misses transport-specific failures) |
| D7 | **Runtime kernel-mode = `unordered`** via `NCCL_SYM_GIN_KERNELS_ENABLE=0` | DeepEP V2 `ordered` kernels demand GIN **strong signals + VA signals**, which aws-ofi-nccl 1.21.1 on EFA does **not** support; `unordered` needs only weak/counting signals, which EFA supports | `ordered` (fails `GIN strong signals are required, but the GIN plugin does not support them`) |
| D8 | **Traceability = pinned ARGs + OCI labels + SBOM + git provenance** | Anyone must be able to answer "what versions produced this digest" from the image alone | Floating tags / `latest` (irreproducible) |
| D9 | **Scanning = ECR enhanced scanning (Amazon Inspector) + Trivy in CodeBuild** | Two independent scanners; Trivy gates pre-push, Inspector gives continuous post-push coverage | One scanner only |
| D10 | **Final output = PR to the recipe's upstream repo** (e.g. `awslabs/awsome-distributed-ai`) via `gh`, from a fork | Contributes the recipe upstream; the Dockerfile's base half follows the awslabs reference (with a short, listed set of changes, §4), so the delta (NIXL layering, validation manifests) is the contribution | Keeping it private (defeats the stated goal) |
| D11 | **SBOM.txt per image + AWS OSS license gate** | An SBOM is a **hard prerequisite** for publishing the image; publishing without one is not allowed. Beyond the SBOM existing, every license it lists must be on the **AWS OSS allow-list** — a disallowed/unknown license is a legal blocker, so it gates publish, not just a scan report | Publishing with no SBOM (not permitted); SBOM-without-license-check (ships a GPL/AGPL/unknown dependency unnoticed) |
| D11a | **License gate carries scoped exceptions + verified pins, never a blanket pass** | A strict MIT/Apache/BSD list on a whole GPU container false-blocks the base OS (GPL userland + toolchain) and the NVIDIA proprietary-but-redistributable stack, and leaves source-built deps as `NOASSERTION`. The gate resolves these with two **scoped, named** exceptions (`base-os`, `nvidia-proprietary`), verified `known_licenses` pins, and `license_aliases` normalization — each narrow and auditable (§7.2) | A single broad allow-list (false-blocks every Linux image on `apt`/`gcc`); ignoring an entire layer (would wave through a genuinely bad app-layer license) |
| D12 | **Dockerfile may live in ANY accessible git repo; a bidirectional repo↔registry pointer is mandatory** | We should not be tied to one repo. The only hard rule: the Dockerfile sits in a git repo anyone-with-access can clone, the repo README names **where the image is published**, and the ECR repo description points **back at the git repo + path**. Either half alone leaves an image or a recipe orphaned | Hardcoding `awsome-distributed-ai` as the only valid home (artificial restriction); publishing an image with no link to its source (irreproducible provenance) |
| D13 | **Every published image is self-documenting: what it does + how to run it** | A consumer must be able to `docker pull` and run the image out of the box without reading the pipeline. The image carries a one-line purpose and the exact run command, both in OCI labels **and** mirrored in the ECR repo description / README | Purpose/run-command only in a wiki (drifts from the image); nothing (consumer has to reverse-engineer the entrypoint) |

---

## 3. Architecture

```
                         ┌──────────────────────────────────────────────┐
   git push (Dockerfile, │              AWS CodePipeline                 │
   setup script,         │              acct ${AWS_ACCOUNT_ID} / ${ECR_REGION}    │
   manifests)            │                                               │
        │                │  Source ─► Build ─► Scan ─► Validate ─► Publish ─► Contribute
        ▼                │    │        │        │        │          │          │       │
  ┌───────────┐          │    │        │        │        │          │          │       │
  │  Source   │──────────┼────┘        │        │        │          │          │       │
  │ (CodeStar │          │  CodeBuild  │        │        │          │          │       │
  │  conn to  │          │  BuildKit   │        │        │          │          │       │
  │  GitHub)  │          │  ──► ECR    │        │        │          │          │       │
  └───────────┘          │  (untested  │        │        │          │          │       │
                         │   tag)      │        │        │          │          │       │
                         │             ▼        │        │          │          │       │
                         │        Trivy scan ───┤        │          │          │       │
                         │        gate H/CRIT   │        │          │          │       │
                         │                      ▼        │          │          │       │
                         │             CodeBuild assumes cluster role│          │       │
                         │             kubectl apply ───────────────►│          │       │
                         └──────────────────────────────┼───────────┼──────────┼───────┘
                                                         ▼           ▼          ▼
                                            ┌────────────────────────────────────────┐
                                            │   EKS ${EKS_CLUSTER} (p5en H200) │
                                            │   NCCL ar/ag/a2a + DeepEP V2 + NIXL      │
                                            │   2 nodes / 16 ranks, EFA + GIN          │
                                            └────────────────────────────────────────┘
                                                         │ pass
                                                         ▼
                                       ECR retag :validated + digest + SBOM
                                                         │
                                                         ▼
                                       gh pr create ─► awslabs/awsome-distributed-ai
```

**Flow in words:**
1. **Source** — a CodeStar Connection watches the git repo holding the Dockerfile,
   `setup_deepep_gin.sh`, and the k8s test manifests. The repo can be any
   accessible git repo (D12), not necessarily awsome-distributed-ai.
2. **Build** — CodeBuild runs BuildKit, produces the image, pushes it to ECR under a **candidate** tag (`:candidate-<gitsha>`).
3. **Scan + SBOM + license (D11)** — Trivy scans the candidate; HIGH/CRITICAL
   fails the stage. `syft` emits `SBOM.txt`; a license step cross-checks every
   SBOM license against the AWS OSS allow-list and fails on any disallowed/unknown
   license. ECR enhanced scanning runs in parallel for the record.
4. **Validate** — CodeBuild assumes a cluster-scoped IAM role, `kubectl apply`s the MPIJobs, waits for completion, parses logs, gates on all tests passing (and the §6.5 throughput floors).
5. **Publish** — on pass, retag the **same digest** to `:validated` and
   `:nixl1.5.2-deepepv2-efa1.51.0`, attach the SBOM, and write the
   self-documentation labels (purpose + run command, D13) and the git-source
   back-pointer (D12).
6. **Cross-pointer (D12)** — set the ECR repo description to point at the git
   repo + path; the git repo README names the published image URI.
7. **Contribute** — a final CodeBuild step runs `gh pr create` against the
   recipe's upstream (e.g. `awslabs/awsome-distributed-ai`).

---

## 4. The Dockerfile as a template

The base `Dockerfile` is written so others can **lift one integration section** for
their own image. It is a **single-stage** build `FROM nvidia/cuda:13.0.2-devel-ubuntu22.04`. Sections, top to bottom:

| Section | What it does | Reuse note |
|---|---|---|
| **Base + ARGs** | CUDA 13.0.2 devel, all versions as `ARG`s | Change `TORCH_CUDA_ARCH_LIST` / `NVCC_GENCODE` for a different GPU arch |
| **① awslabs base** | EFA installer 1.51.0 (sha256-checked), GDRCopy v2.5.2, torch 2.11.0+cu130 with GIN NCCL 2.31.2 pinned in the same layer, DeepEP V2 at `d75f22a5` | Follows `reference/deepep.Dockerfile` (vendored unchanged). Changes from it: EFA 1.51.0 + sha256, DeepEP pinned to a commit, NCCL pinned once in the torch layer, `debhelper`/`devscripts` dropped |
| **② nccl-tests** | MPI-enabled `all_reduce/all_gather/alltoall` perf tests against the GIN NCCL | NCCL EFA transport benchmarks |
| **③ NIXL v1.5.2** | meson/ninja build, `-Denable_plugins=LIBFABRIC`, `-Dlibfabric_path=/opt/amazon/efa`; no UCX build, no `nixl_ep`; copies the LIBFABRIC and backend-free tests to `/opt/nixl-tests` | The NIXL+EFA integration template; `enable_plugins` errors if LIBFABRIC can't be built, and the step fails if a UCX plugin appears |
| **⑤ Runtime ENV + final gate** | `FI_PROVIDER=efa`, `NCCL_GIN_TYPE=5`, absolute `NCCL_NET_PLUGIN`, `ldconfig` for gdrcopy; final `nm` + plugin + NIXL test + import gate | The "works with zero per-run env" contract |

The separate "GIN NCCL re-pin" layer and the UCX section of earlier versions are gone.

Shipped NIXL tests (`/opt/nixl-tests`): `libfabric_topology_test`,
`rail_active_refcount_test`, `desc_example`, `p2p_test`, `serdes_test`,
`telemetry_reader`, plus the `*-topo.xml` files `libfabric_topology_test` reads.
Not shipped, because each hard-codes the UCX backend: `nixl_test`, `agent_example`,
`nixl_example`, and NIXL EP (`nixl_ep`).

**`Dockerfile.vllm`** starts `FROM vllm/vllm-openai:v0.33.0-x86_64-efa@sha256:5fc20aa1…`.
That base already has vLLM 0.33.0, torch 2.14.0+cu130, Python 3.12, NCCL 2.31.2 (GIN),
EFA 1.51.0 (libfabric 2.7.0, aws-ofi-nccl 1.22.0, `NCCL_NET_PLUGIN=ofi`), GDRCopy 2.5.1,
the NIXL 1.5.0 wheel (`nixl-cu13`, LIBFABRIC plugin linked to the EFA libfabric),
Mooncake EFA 0.3.13.post1, LMCache 0.5.6, FlashInfer 0.7.2.post1 and DeepEP 2.1.0+874779c
(sm_90a/100a/103a). It has no OpenMPI and no nccl-tests. On top, `Dockerfile.vllm` adds:

| Addition | Why |
|---|---|
| DeepEP V2 rebuilt at `d75f22a5` for sm_90/100/103/120 | 9 EFA fixes newer than `874779c` (GIN flush before exit, `gin.flush` timeout, proxy ring wait, hybrid dispatch races) and sm_120 for g7e |
| NIXL LIBFABRIC tests in `/opt/nixl-tests`, run via `nixl-test <name>` | Same six tests, built in a discarded stage. `/opt/nixl` stays off the global loader path, so vLLM keeps the wheel's libnixl |
| `ENV FI_PROVIDER=efa NCCL_GIN_TYPE=5` | DeepEP V2 EFA-GDA runtime selection |
| Build gates | vLLM/torch/NCCL unchanged, GIN symbol, aws-ofi-nccl plugins in the loader cache, GDRCopy, wheel LIBFABRIC plugin → EFA libfabric, NIXL tests resolve and `serdes_test` runs, DeepEP SASS per arch |

`vllm-requirements.txt`, `flashinfer-kernels.txt` and `deadsnakes.asc` are no longer
used by either Dockerfile. They stay in the directory.

### 4.1 Integration contracts worth copying

- **EFA → NCCL:** `NCCL_NET_PLUGIN=/opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so`
  (absolute). NCCL 2.31.2 auto-discovers only `libnccl-net.so`; without the
  absolute path it **silently** falls back to the TCP Socket transport (no RDMA).
- **EFA → NIXL:** meson `-Dlibfabric_path=/opt/amazon/efa` so the LIBFABRIC
  backend links the EFA libfabric (`>=2.6`), not a system one. Build fails if the
  `libplugin_LIBFABRIC.so` is not produced.
- **GIN NCCL integrity:** the committed `libnccl.so.2` is asserted at the binary
  level (`nm -D … | grep ncclCommQueryProperties`), never by pip metadata — pip
  metadata lied across v4–v8.
- **GPUDirect RDMA:** `/opt/gdrcopy/lib` registered via `/etc/ld.so.conf.d` +
  `ldconfig` so `libgdrapi.so` resolves for every process, not only those that
  inherit `LD_LIBRARY_PATH` (ldconfig does not read `LD_LIBRARY_PATH`).

### 4.2 Build defects found and fixed (the six fixes)

The first working image took several build iterations; each fix pushed the
failure later. They are listed here because every one is a trap a derived recipe
will hit again. Rows 3, 5 and 6 were found through `nixl_test`, which is no longer
built or shipped (it hard-codes UCX). Their symptom column is history; the fix in
each row is still in the Dockerfile and still needed by the shipped LIBFABRIC tests.

| # | Symptom | Root cause | Fix (in the Dockerfile) |
|---|---|---|---|
| 1 | `ld: cannot find -lnccl` while building nccl-tests | The GIN NCCL is installed as `libnccl.so.2` only; the linker needs the dev symlink `libnccl.so` | `ln -sf libnccl.so.2 "$NCCL_LIB/libnccl.so"` before `make` |
| 2 | NIXL meson error on the CUDA arch list | `-Dnixl_cuda_arch_list` wants a **comma**-separated list; a semicolon list is mis-parsed | pass `-Dnixl_cuda_arch_list="90,100,103,120"` (commas), not `"9.0;10.0;10.3;12.0"` |
| 3 | (obsolete symptom) `nixl_test` binary never built | NIXL skips the whole `test/` subdir when meson `buildtype=release` (the default) | `meson setup … --buildtype=debugoptimized` so the test subdir builds. Still needed: the shipped tests in `/opt/nixl-tests` come from the same subdir |
| 4 | libfabric utils fail to compile under `-O2` | `-O2` inlining makes GCC raise a **false-positive** `-Wmaybe-uninitialized` inside the EFA SDK header `fi_domain.h` (`fi_mr_reg`); NIXL's default `werror=true` makes it fatal | `meson setup … -Dwerror=false` |
| 5 | (obsolete symptom) `nixl_test` segfaults at startup: `createBackend('UCX')` fails | Baked `NIXL_PLUGIN_DIR` was a **colon-joined** candidate list; NIXL `opendir()`s the value literally (does not split on `:`), so the open fails | set `NIXL_PLUGIN_DIR` to the **single** real dir `${NIXL_PREFIX}/lib/x86_64-linux-gnu/plugins`. Still applies to `createBackend('LIBFABRIC')` |
| 6 | (obsolete symptom) `nixl_test` can't start: `libserdes.so: cannot open shared object file` | NIXL installs its libs into the multiarch subdir `${NIXL_PREFIX}/lib/x86_64-linux-gnu`, which was **not** on `LD_LIBRARY_PATH` | prepend that subdir to `LD_LIBRARY_PATH` (base image) or set it per test via the `nixl-test` wrapper (vLLM image) |

Two process rules learned alongside the fixes, both now standing practice:
- **Verify the pushed image by immutable `@digest` with `imagePullPolicy: Always`.**
  A verify pod pinned to the mutable `:multiarch-baseline` tag reused a node's
  **stale cached** layer (`IfNotPresent`) and tested the *prior* build — a false
  pass. Fixes #5/#6 were first "confirmed" against the wrong image this way.
- **Assert runtime state in a FRESH, unmutated pod.** A pod that earlier `kubectl
  exec` commands manually patched produces a false pass that masks a broken
  pushed image.

---

## 5. Build arguments (traceability matrix)

Every version is a pinned `ARG`. These, plus the git SHA of the build context,
determine the image.

Base `Dockerfile`:

| ARG | Value | Role |
|---|---|---|
| `CUDA_VERSION` | `13.0.2` | base image `nvidia/cuda:13.0.2-devel-ubuntu22.04` |
| `EFA_INSTALLER_VERSION` / `EFA_INSTALLER_SHA256` | `1.51.0` / `83993044…` | EFA userspace + aws-ofi-nccl GIN plugin + libfabric; tarball sha256-checked |
| `NCCL_VERSION` | `2.31.2` | GIN-capable NCCL (exports `ncclCommQueryProperties`), pinned once in the torch layer |
| `GDRCOPY_VERSION` | `v2.5.2` | GPUDirect RDMA copy |
| `DEEPEP_REF` | `d75f22a5e07c95eb0bae94ad6445a5009c06c886` (`amazon-contributing/DeepEP`) | DeepEP V2 source, by commit |
| `NIXL_REF` / `NIXL_COMMIT` | `v1.5.2` / `e72c52c7…` (2026-10-28) | NIXL source; tag asserted to equal the commit. 1.5.2 fixes two libfabric 2.7 (EFA 1.51) WRITE bugs in 1.5.0 (ai-dynamo/nixl#2308, #2494) |
| `TORCH_VERSION` | `2.11.0` (`cu130`) | PyTorch |
| `PYTHON_VERSION` | `3.10` | interpreter |
| `NCCL_TESTS_REF` | `master` | nccl-tests (not pinned) |
| `TORCH_CUDA_ARCH_LIST` | `9.0;10.0;10.3;12.0` | Hopper + Blackwell + Grace-Blackwell + RTX PRO 6000 (g7e) |
| `NVCC_GENCODE` | `...sm_90 ...sm_100 ...sm_103 ...sm_120` | multi-arch codegen |

`UCX_VERSION` and `NCCL_REPIN_BUST` no longer exist.

`Dockerfile.vllm`:

| ARG | Value | Role |
|---|---|---|
| `BASE_IMAGE` | `vllm/vllm-openai@sha256:5fc20aa179d3f7f83aa4a8b66473edf80582d865c62b630e49a207c76b9a0886` (`v0.33.0-x86_64-efa`) | base, by digest |
| `DEEPEP_REF` | `d75f22a5…` | DeepEP V2 rebuild |
| `NIXL_REF` / `NIXL_COMMIT` | `v1.5.2` / `e72c52c7…` | NIXL LIBFABRIC tests; the base's 1.5.0 wheels are replaced by `nixl-cu13`/`nixl` 1.5.2 (sha256-pinned) |
| `TORCH_CUDA_ARCH_LIST` | `9.0;10.0;10.3;12.0` | DeepEP + NIXL tests |
| `VLLM_VERSION` / `TORCH_VERSION` / `NCCL_VERSION` | `0.33.0` / `2.14.0` / `2.31.2` | asserted unchanged from the base |
| `build-requirements.txt` | `--require-hashes` | meson/ninja/pybind11 for the NIXL test stage |

**Image labels** (base `Dockerfile`, queryable from the image alone):
```
org.opencontainers.image.description = DeepEP V2 (NCCL GIN/EFA-GDA) + NIXL v1.5.2 (LIBFABRIC only) on CUDA 13.0.2, EFA installer 1.51.0, multi-arch sm_90/sm_100/sm_103/sm_120 (...)
nixl.ref        = v1.5.0
deepep.ref      = amazon-contributing/DeepEP@d75f22a5e07c95eb0bae94ad6445a5009c06c886
efa.installer   = 1.51.0
```
`Dockerfile.vllm` adds `vllm.version`, `torch.version` and `base.image` (the digest above).
Add in CI: `org.opencontainers.image.revision=<gitsha>`,
`org.opencontainers.image.source=<repo url>`,
`org.opencontainers.image.created=<timestamp>`.

**Published image (${AWS_REGION}):**
```
${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:nixl1.5.2-deepepv2-efa1.51.0
@${IMAGE_DIGEST}
```
Validated images from this branch (commit `c8a6180`, 2026-11-05; results in the
container README): base `nixl1.5.2-deepepv2-efa1.51.0@sha256:a8688aad…` (commit `e7d45ef`, no squash) and vLLM
`vllm0.33.0-efa-slim-nixl152@sha256:94da4fa4…`, both ~9.9 GiB. Earlier validated images,
from the previous Dockerfiles (EFA 1.50.0, NIXL with UCX): base `091ce165` (commit
`020e077`, ~10.2 GiB) and vLLM `eeaa6801` (commit `12a6ae6`, ~10.8 GiB).
The pipeline (D1) publishes to **${ECR_REGION}** ECR, repo
`${PUBLISH_REPO}` (to be created — the pipeline's publish
stage creates it on first run if absent). Note the auth split (D1a): the login
token is minted with `--region ${ECR_TOKEN_REGION}`, but the registry host and repo are
`${ECR_REGION}`.

---

## 6. Validation stage — the tests

All tests run as **2-node / 16-rank** jobs (8 GPU + 16 EFA NICs per
p5en.48xlarge node), pinned to driver-flagged nodes. Each is gated: a non-zero
exit or a missing success marker fails the pipeline.

### 6.1 NCCL collectives (EFA transport)
`nccl-tests` is built into the base image (builder stage, against the GIN NCCL +
EFA/OMPI stack) and placed on `PATH`. The vLLM image has neither nccl-tests nor
OpenMPI. Run all-reduce, all-gather, all-to-all:
```
mpirun -np 16 -N 8 --bind-to none \
  -x NCCL_DEBUG=INFO -x NCCL_NET_PLUGIN=ofi -x NCCL_TUNER_PLUGIN=ofi \
  all_reduce_perf  -b 8 -e 8G -f 2 -g 1
mpirun ... all_gather_perf   -b 8 -e 8G -f 2 -g 1
mpirun ... alltoall_perf     -b 8 -e 8G -f 2 -g 1
```
**Gate:** every rank logs `NET/OFI Selected provider is efa … (found 16 nics)`,
busbw reported, exit 0. Proves the EFA RDMA path end to end.

### 6.2 DeepEP V2 (NCCL GIN / EFA-GDA)
The MPIJob we validated — `tests/elastic/test_ep.py` (`ElasticBuffer` = GIN
backend), with the awslabs recipe env:
```
NCCL_GIN_TYPE=5  NCCL_SYM_GIN_KERNELS_ENABLE=0  NCCL_NET_PLUGIN=ofi
TORCHDYNAMO_DISABLE=1  TORCH_COMPILE_DISABLE=1  NVIDIA_GDRCOPY=enabled
python3 /opt/amazon/deepep/tests/elastic/test_ep.py \
  --test-first-only --num-tokens=8192 --num-topk=8 --num-sms=12
```
**Gate:** `Loaded gin plugin Libfabric_GDAKI`, no `host_register_iomem failed`,
no `GIN strong signals are required`, dispatch + combine complete on all 16 ranks.
**Reference numbers (validated 2026-10-14):** dispatch ~265–270 GB/s bus BW
(~82 GB/s wire), combine ~200–250 GB/s bus BW. Nodes **must** carry the GDAKI
driver flags (D5).

### 6.3 NIXL over LIBFABRIC / EFA (`nixl_lf`)
The NIXL suite is `nixl_lf`: `benchmarks/nixl/nixl-lf-target-validate.yaml` and
`benchmarks/nixl/nixl-lf-initiator-validate.yaml`, which run
`benchmarks/nixl/nixl_lf_test.py` through `nixl-lf-run.sh` (both from a ConfigMap).
It creates only the LIBFABRIC backend and reads 1 GiB GPU-to-GPU from the target
pod, 5 transfers per run, 3 runs (`NIXL_ITERS="1 2 3"`), 1 GPU by default:
```
kubectl -n ${K8S_NAMESPACE} create configmap nixl-lf-scripts \
  --from-file=benchmarks/nixl/nixl-lf-run.sh --from-file=benchmarks/nixl/nixl_lf_test.py
envsubst < benchmarks/nixl/nixl-lf-target-validate.yaml | kubectl apply -f -      # NODE_1
export NIXL_TARGET_IP=$(kubectl -n ${K8S_NAMESPACE} get pod nixl-lf-target -o jsonpath='{.status.podIP}')
envsubst < benchmarks/nixl/nixl-lf-initiator-validate.yaml | kubectl apply -f -   # NODE_0
```
**Gate:** `NIXL backends created: ['LIBFABRIC']` and `LIBFABRIC backend selected`,
`verify=OK` on 5/5 transfers per run, `NIXL_LF_RESULT PASS` and rc=0 on both pods,
EFA `rdma_read_bytes` rising by 5 GiB per run, and mean GB/s at or above the floor
(§6.5).

This replaces NIXL's own `nixl_test` (`test/nixl_test.cpp`), which hard-codes
`createBackend("UCX")` and so never tested LIBFABRIC over EFA. `nixl_test` is no
longer built, and the four `benchmarks/nixl/nixl-test-*.yaml` manifests are deleted.
The NIXL C++ tests that do ship (`/opt/nixl-tests`, §4) are not part of the gate;
`serdes_test` runs as a build gate.

### 6.4 Test harness notes
- Rendezvous: headless Service with `publishNotReadyAddresses: true` + per-pod
  DNS, OR the kubeflow **mpi-operator** MPIJob (SSH rendezvous). The validated
  path uses the MPIJob.
- Pin workers to driver-flagged nodes (any 2 p5en in the v2 nodegroup qualify).
- Worker pods need: `nvidia.com/gpu: 8`, `vpc.amazonaws.com/efa: 16`,
  `SYS_ADMIN` + `IPC_LOCK` caps, 200Gi `/dev/shm`, `NVIDIA_GDRCOPY=enabled`.

### 6.5 Throughput floors (the sign-off gate)
Functional pass (§6.1–6.3) is necessary but not sufficient — the build must also
meet a **performance floor** or publishing is blocked. The floors live in
`validation-gate.yaml` next to the Dockerfile. Each metric's 3-run mean must be
at or above its floor:

| Suite / metric | Baseline | Floor | Tolerance |
|---|---|---|---|
| NCCL all_reduce busbw @8G | 479.5 | 465.1 | −3% |
| NCCL all_gather busbw @8G | 349.3 | 338.8 | −3% |
| NCCL all_to_all busbw @8G | 87.8 | 85.2 | −3% |
| DeepEP dispatch bus BW | 267.0 | 259.0 | −3% |
| DeepEP combine bus BW | 187.0 | 177.6 | **−5%** |
| nixl_lf (1 GPU READ, 1 GiB) mean GB/s | 47.2 | 45.8 | −3%, plus PASS on all 3 runs |

NCCL baselines reset 2026-11-07 on NCCL 2.31.2 (image `a8688aad`). The old ones (476.5 /
352.9 / 83.5) were measured on apt NCCL 2.28.3, which the former squash stage brought back.

The nixl_lf baseline is from image `eeaa6801`, 2 x p5en, 2026-11-03 (47.23 / 47.26 /
47.22). Context, not gated: 8 GPUs give 382.5 GB/s READ and 382.2 GB/s WRITE (sum).

**Why combine is −5%:** DeepEP combine's measured run-to-run spread is ~6% (a
rendezvous-role-tied per-node asymmetry — worker-0 ranks run faster than
worker-1), wider than the −3% band used for the tighter metrics, so a clean
image would occasionally dip under −3% on a single 3-run mean. The looser floor
catches a real regression without false-blocking on intrinsic variance.
**Gate rule:** sign off only if every metric clears its floor AND nixl_lf
passes 3/3 at or above its floor; otherwise BLOCK publish.

## 7. SBOM, licenses, and image self-documentation

### 7.1 SBOM.txt — a publish prerequisite (D11)
Every image **must** carry an SBOM before it can be published; a missing SBOM
blocks the publish stage. The pipeline generates it from the pushed candidate
digest and keeps it as a pipeline artifact and alongside the image:

```bash
# syft >= 1.55. Scope to OS (dpkg) + pip packages only -- the go-module/binary
# catalogers otherwise add ~300 Go deps vendored inside baked tools (kubectl,
# the EFA installer), which are not shipped app dependencies. License-content
# enrichment makes deb copyright files resolve to real SPDX ids.
export SYFT_LICENSE_CONTENT=all
CATS="dpkg-db-cataloger,python-installed-package-cataloger"
# machine-readable SBOM (kept as the provenance artifact)
syft "$IMG" --override-default-catalogers "$CATS" -o spdx-json > sbom.spdx.json
# human-readable SBOM.txt (the required per-image file)
syft "$IMG" --override-default-catalogers "$CATS" -o syft-table > SBOM.txt
```

`SBOM.txt` lists every OS package, Python wheel, and vendored library (CUDA,
EFA, GIN NCCL, DeepEP, NIXL, torch) with its **version and license**. It is:
- attached to the ECR image (OCI referrer / `aws ecr put-image` referrer, or at
  minimum published next to the image in the repo README),
- committed into the git recipe repo under the recipe path, and
- a hard gate: **no SBOM.txt → no publish.**

### 7.2 License gate — AWS OSS allow-list (D11)
Having an SBOM is not enough; every license it lists must be **allowed by AWS
OSS policy**. The pipeline extracts the license field for each component and
checks it against the AWS OSS allow-list:

```bash
# emit (name, version, type, license) rows from syft NATIVE json (spdx-json
# mangles multi-license packages; native json carries per-license spdxExpression).
# `type` lets the gate tell a GPL *dpkg* base-OS package (excused) from a GPL
# *pip wheel* (blocked). The emitter maps syft's type to deb/pypi/... .
export SYFT_LICENSE_CONTENT=all
CATS="dpkg-db-cataloger,python-installed-package-cataloger"
syft "$IMG" --override-default-catalogers "$CATS" -o json \
  | python3 sbom_emit.py > licenses.tsv
# BLOCK publish if any in-scope license is not allow-listed and not excepted.
# check_licenses.py reads the policy (allow-list, exceptions, known_licenses,
# license_aliases, scan_scope) straight from validation-gate.yaml:
python3 check_licenses.py --gate validation-gate.yaml licenses.tsv
```

Rules:
- **Allowed** — the AWS OSS policy allow-list for this pipeline. `validation-gate.yaml`
  `license_policy.allow` holds: `MIT`, `Apache-2.0`, `BSD-2-Clause`,
  `BSD-3-Clause`, `ISC`, `Python-2.0` (PSF).
- **Blocked** — every other license: all copyleft (GPL, LGPL, AGPL, MPL), all
  source-available / non-OSS (SSPL, BUSL, Commons Clause), and any
  `NOASSERTION` / unknown (`unknown_blocks: true`).
- A license outside the allow-list, or an unknown one, **fails the stage** and
  blocks publish. A component that must stay despite a flagged license requires
  an explicit, recorded policy **exception** or a verified **pin** — never a
  silent pass.

**Three resolver helpers the gate applies before the allow/block decision**
(all in `validation-gate.yaml`, enforced by `check_licenses.py`):

| Helper | Key | What it does |
|---|---|---|
| scan scope | `scan_scope.include_types: [deb, pypi]` | Only OS (dpkg) + pip packages are gated. The `go-module`/`binary` catalogers find Go vendored inside baked tools (kubectl, the EFA installer) — not shipped deps — so they are out of scope. Matches the `--override-default-catalogers` flag on the scan. |
| known-license pins | `known_licenses` | Source-built components (`nixl`, `deep_ep`, `nccl`, `torch`, `triton`, `gdrcopy`) carry no scanner-readable metadata (installed by `git clone && make`), so their SBOM license is `NOASSERTION`. Each is pinned to its upstream license, verified against the repo's LICENSE file. A pin overrides the missing value, then the pinned license is still checked against the allow-list. Common PyPI wheels syft reads as `NOASSERTION` are pinned the same way. |
| license aliases | `license_aliases` | Free-text strings pip wheels emit (`"Apache License 2.0"`, `"ASL 2"`, `"LGPLv3"`, `"NVIDIA Proprietary Software"`) are normalized to SPDX ids before the check, so a permissive wheel passes and a copyleft one still blocks. |

**Two standing exceptions (D11a).** A strict MIT/Apache/BSD list applied to a
whole GPU container false-blocks two groups that are *not* our application code,
so the gate carries two **scoped, named** exceptions (defined in
`validation-gate.yaml` under `license_policy.exceptions`, enforced by
`check_licenses.py`):

| Exception | Covers | Why it's OK | Scope limit |
|---|---|---|---|
| `base-os` | Ubuntu userland + build toolchain from the `nvidia/cuda:*-devel-ubuntu22.04` base (`apt`, `gcc`, `g++`, `cpp`, `binutils`, `coreutils`, `bash`, `tcl`, `libisl`…) under GPL-2.0/3.0/LGPL | The copyleft applies to the **OS layer**, present in every Linux image; it does not reach our separately-licensed app layer | Only `deb` (dpkg) packages on the named list, and only for the listed copyleft IDs. A **pip wheel** claiming GPL still blocks. |
| `nvidia-proprietary` | CUDA, cuDNN, NVIDIA-distributed NCCL, cuBLAS/cuFFT/cuRAND/cuSOLVER, driver libs — the **NVIDIA proprietary EULA**, not OSS at all | Redistributable in a container per the NVIDIA Deep Learning Container license; required for any GPU image | Only the named `cuda*`/`libcu*`/`nvidia-*` packages, only under the EULA / `NOASSERTION` NVIDIA wheels emit |

An exception excuses **only the named packages (by glob) under the named license
IDs** — it is not a blanket "ignore this layer". A base-OS package showing an
*unexpected* license, or any non-exception component under a copyleft/unknown
license, still blocks. New copyleft package names on a future build must widen
the exception **on purpose**, with the reviewer recorded in the gate's
`review_note` — never by a catch-all.

> This matters because the stack vendors a lot: a transitive wheel pulling in a
> GPL/AGPL dependency would otherwise ship unnoticed into a publishable AWS
> image. The exceptions keep the gate from false-blocking the base OS and the
> (non-OSS but redistributable) NVIDIA stack, while still catching a genuinely
> bad license in our own application layer.

### 7.3 Image self-documentation — run out of the box (D13)
A consumer should `docker pull` and run the image without reading this doc. Two
places carry the "what + how", and they must agree:

**(a) OCI labels baked at build time:**
```dockerfile
LABEL org.opencontainers.image.description="DeepEP V2 (NCCL GIN/EFA-GDA) + NIXL v1.5.2 (LIBFABRIC only) on CUDA 13.0.2, EFA 1.51.0, multi-arch sm_90/sm_100/sm_103/sm_120"
LABEL org.opencontainers.image.source="https://github.com/<org>/<repo>/tree/<path>"   # back-pointer to the git recipe (D12)
LABEL run.purpose="2-node GPU-collective benchmarking: NCCL (ar/ag/a2a), DeepEP V2 MoE dispatch/combine, NIXL transfer, over EFA"
LABEL run.command="mpirun -np 16 -N 8 --bind-to none all_reduce_perf -b 8 -e 8G -f 2 -g 1   # see repo README for DeepEP / NIXL invocations"
```

**(b) The ECR repo description / README** mirrors the same purpose and the exact
run commands (the §6 invocations), so the "how to run it" is visible from the
registry UI without pulling the image.

Verify from the image alone:
```bash
docker inspect --format '{{json .Config.Labels}}' "$IMG" | jq .
```

### 7.4 Repo ↔ registry cross-pointer (D12)
The Dockerfile may live in **any accessible git repo** — the pattern, not the
specific repo, is what matters. The one hard requirement is a **two-way link**:

| Direction | Where | What it says |
|---|---|---|
| git repo → registry | recipe repo README | the published image URI(s), e.g. `${AWS_ACCOUNT_ID}.dkr.ecr.${ECR_REGION}.amazonaws.com/${PUBLISH_REPO}:nixl1.5.2-deepepv2-efa1.51.0`, and that `:validated` tracks the latest signed-off digest |
| registry → git repo | ECR repo description + the `org.opencontainers.image.source` label | the git repo URL + path the image was built from |

Set the ECR repo description back-pointer at publish time:
```bash
aws ecr put-repository-catalog-data \
  --repository-name ${PUBLISH_REPO} \
  --catalog-data 'about=Built from https://github.com/<org>/<repo>/tree/<path>. Purpose: 2-node GPU-collective benchmarking (NCCL/DeepEP V2/NIXL over EFA). Run: see repo README.' \
  --region ${ECR_REGION} --profile ${AWS_PROFILE}
```
Neither half is optional: an image with no source link is an orphaned binary; a
recipe with no published-location note leaves a consumer unable to find the image.

---

## 8. Reproduction — step by step

### 8.0 Prerequisites
- AWS access to `${AWS_ACCOUNT_ID}` (profile `${AWS_PROFILE}`), regions ${ECR_REGION}
  (pipeline/ECR; auth token minted via `--region ${ECR_TOKEN_REGION}`) and ${AWS_REGION}
  (cluster, if validating against `${EKS_CLUSTER}`).
- `kubectl` context for `${EKS_CLUSTER}`; the kubeflow mpi-operator CRD
  installed (`kubectl get crd mpijobs.kubeflow.org`).
- An **accessible git repo** holding the build context (D12 — need not be
  awsome-distributed-ai); `gh` authenticated for the upstream PR.
- **Driver flags on the GPU nodes (D5)** — see §8.3.

### 8.1 Build locally (dev loop)
```bash
cd docker/nixl-deepep-efa
DOCKER_BUILDKIT=1 docker build --progress=plain -f Dockerfile \
  --build-arg TORCH_CUDA_ARCH_LIST="9.0;10.0;10.3" \
  --build-arg NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_100,code=sm_100 -gencode=arch=compute_103,code=sm_103" \
  -t <registry>/nixl-deepep-efa:dev .
```
The in-build self-tests (the torch-layer `nm` check in ① and the final gate in ⑤)
fail the build if the GIN NCCL `.so` is stale, the LIBFABRIC plugin is missing, a
UCX plugin appears, or a NIXL test does not resolve — a green build is already partly validated.

### 8.2 Build in CI (CodeBuild + BuildKit → ECR)
`buildspec.yml` (sketch):
```yaml
version: 0.2
phases:
  pre_build:
    commands:
      # Auth token minted in ${ECR_TOKEN_REGION} (account requirement); registry host + repo are ${ECR_REGION}
      - aws ecr get-login-password --region ${ECR_TOKEN_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${ECR_REGION}.amazonaws.com
      - export GITSHA=$(git rev-parse --short HEAD)
      - export IMG=${AWS_ACCOUNT_ID}.dkr.ecr.${ECR_REGION}.amazonaws.com/${PUBLISH_REPO}:candidate-$GITSHA
  build:
    commands:
      - |
        docker build --progress=plain -f docker/nixl-deepep-efa/Dockerfile \
          --build-arg TORCH_CUDA_ARCH_LIST="9.0;10.0;10.3" \
          --build-arg NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_100,code=sm_100 -gencode=arch=compute_103,code=sm_103" \
          --label org.opencontainers.image.revision=$GITSHA \
          --label org.opencontainers.image.source=$CODEBUILD_SOURCE_REPO_URL \
          --label org.opencontainers.image.created=$(date -u +%Y-%m-%dT%H:%M:%SZ) \
          -t $IMG docker/nixl-deepep-efa
      - docker push $IMG
  post_build:
    commands:
      # SBOM + scan + license gate — see §7
      - trivy image --exit-code 1 --severity HIGH,CRITICAL $IMG
      - syft $IMG -o spdx-json > sbom.spdx.json
      - syft $IMG -o syft-table > SBOM.txt
      - python3 check_licenses.py --allowlist aws-oss-allowlist.txt <(syft $IMG -o spdx-json | jq -r '.packages[] | [.name,.versionInfo,(.licenseConcluded // .licenseDeclared // "NOASSERTION")] | @tsv')
artifacts:
  files: [sbom.spdx.json, SBOM.txt]
```
> **Note on `--no-cache` / dedup:** the original in-cluster rootless BuildKit hit
> the layer-dedup trap (D3). Pinning NCCL in the torch layer is the fix. If a CI
> runner ever reintroduces a stale `libnccl.so.2`, the in-build `nm` assertion
> fails the build — it cannot ship silently.

### 8.3 Set the node driver flags (one-time, D5)
Bake into the **nodegroup launch template userData** so every (current and
replacement) node comes up flagged — a per-instance reboot does **not** persist
because the autoscaler replaces nodes:
```bash
# modprobe.d drop baked via a nodeadm pre-kubelet hook in the LT userData:
options nvidia NVreg_EnableStreamMemOPs=1 NVreg_RegistryDwords="PeerMappingOverride=1;"
```
Create a new LT version, point the nodegroup at it, verify on a fresh node:
```bash
grep -E "EnableStreamMemOPs|RegistryDwords" /proc/driver/nvidia/params
# want: EnableStreamMemOPs: 1   and RegistryDwords containing PeerMappingOverride=1
```
(On a custom AMI that loads `nvidia` from initramfs, the userData must rebuild
initramfs — `dracut --force` / `update-initramfs -u` — not just drop the file.)

### 8.4 Validate on the cluster
Apply the test manifests (§6) and gate on their results:
```bash
envsubst < benchmarks/nccl/nccl-tests-validate.yaml | kubectl apply -f -   # ar / ag / a2a
envsubst < benchmarks/deepep/deepep-validate.yaml   | kubectl apply -f -   # GIN / EFA-GDA
# NIXL: the nixl-lf target + initiator pods, in order (see §6.3)       # LIBFABRIC / EFA
# wait for each launcher / pod Completed, parse logs for the success markers in §6
```
> **In-cluster BuildKit fallback:** if CodeBuild cannot be used, the original
> `build-pod.yaml` (rootless `moby/buildkit:v0.17.3-rootless`, context from a
> ConfigMap, ECR push via an `ecr-push` secret) still works. Keep it documented
> but prefer CodeBuild for CI isolation.

### 8.5 Publish with provenance
Publish only when **all** gates are green: functional tests, throughput floors
(§6.5), `SBOM.txt` present, and the AWS OSS license check (§7.1–7.2). Retag the
**same digest** (no rebuild), attach the SBOM, write the self-documentation
labels, and set the registry → repo back-pointer:
```bash
MANIFEST=$(aws ecr batch-get-image --repository-name ${PUBLISH_REPO} \
  --image-ids imageTag=candidate-$GITSHA --region ${ECR_REGION} \
  --profile ${AWS_PROFILE} --query 'images[0].imageManifest' --output text)
aws ecr put-image --repository-name ${PUBLISH_REPO} \
  --image-tag nixl1.5.2-deepepv2-efa1.51.0 --image-manifest "$MANIFEST" \
  --region ${ECR_REGION} --profile ${AWS_PROFILE}
# registry → git repo back-pointer + purpose/run (D12, D13)
aws ecr put-repository-catalog-data \
  --repository-name ${PUBLISH_REPO} \
  --catalog-data 'about=Built from https://github.com/<org>/<repo>/tree/<path>. Purpose: 2-node GPU-collective benchmarking (NCCL/DeepEP V2/NIXL over EFA). Run: see repo README.' \
  --region ${ECR_REGION} --profile ${AWS_PROFILE}
```
The `org.opencontainers.image.source`, `run.purpose` and `run.command` labels
(§7.3) are baked at build time, so the published image is self-describing.

### 8.6 Contribute upstream (final stage)
The recipe lives in an accessible git repo (D12); contributing upstream is the
default when that repo has one. Worked example against
`awslabs/awsome-distributed-ai`:
```bash
# fork + branch + PR to the recipe's upstream (awslabs/awsome-distributed-ai here)
gh repo fork awslabs/awsome-distributed-ai --clone
# the contribution lives under
#   architectures/containers/deepv2-nixl-efa/
#   (Dockerfile, reference/deepep.Dockerfile, setup_deepep_gin.sh,
#    validation-gate.yaml, SBOM.txt, README.md)
git checkout -b deepep-v2-nixl-efa-template
git add . && git commit -m "Add DeepEP V2 + NIXL + EFA build template and 2-node validation"
git push -u origin deepep-v2-nixl-efa-template
gh pr create --repo awslabs/awsome-distributed-ai \
  --title "DeepEP V2 + NIXL v1.5.2 on EFA: reproducible build + 2-node validation" \
  --body-file PR_BODY.md
```
If the recipe lives in an org-internal repo instead, this stage is simply the
repo's own review/merge flow — the pattern (§7.4 cross-pointer, §7.1–7.3
SBOM/license/self-doc) is identical regardless of where the repo is hosted.

---

## 9. IAM & security

- **CodeBuild service role:** `ecr:GetAuthorizationToken`, push/pull on the
  `${PUBLISH_REPO}` repo, `ecr:PutImage` +
  `ecr:PutRepositoryCatalogData` (the publish retag + registry back-pointer),
  read the build-context secret, and `sts:AssumeRole`
  into the **cluster-access role** for the validate stage. No standing cluster
  admin on the build role.
- **Cluster-access role:** mapped in the EKS `aws-auth` / access entries to a
  namespace-scoped RBAC role that can only `create/get/delete` Jobs/MPIJobs/Pods
  in the test namespace — not cluster admin.
- **Scanning gate:** Trivy `--exit-code 1 --severity HIGH,CRITICAL` pre-push;
  ECR enhanced scanning (Inspector) post-push for continuous CVE coverage.
- **SBOM + license gate (D11):** `SBOM.txt` is generated per image and is a hard
  publish prerequisite; every SBOM license is checked against the AWS OSS
  allow-list and a disallowed/unknown license blocks publish. This is a *legal*
  gate, distinct from the CVE scan.
- **Secrets:** ECR push uses the role, not a static dockercfg. The GitHub token
  for the final PR stage lives in Secrets Manager, injected only into that stage.
- **No credential material in build args or logs.** The image carries no secrets.
- **Provenance:** digest-pinned publish, SBOM artifact, OCI revision/source
  labels, and the bidirectional repo↔registry pointer (D12).

---

## 10. Resolved decisions

1. **Validation region = cross-region to ${AWS_REGION}.** The pipeline runs in
   ${ECR_REGION} but validation runs against the existing cluster
   (`${EKS_CLUSTER}`, ${AWS_REGION}). CodeBuild's validate stage uses a
   cross-region `kubectl` context; no ${ECR_REGION} p5en nodegroup is stood up.
2. **CodeBuild → cluster reach:** the build role's access entry into the
   cluster is **pending** — the repo owner will confirm the IAM/access-entry
   mapping after the build stage exists.
3. **Recipe repo is not fixed to awsome-distributed-ai (D12).** The Dockerfile
   may live in any accessible git repo, provided the bidirectional repo↔registry
   pointer (§7.4) holds. The worked example contributes
   `architectures/containers/deepv2-nixl-efa/` to awslabs — the Dockerfile (with
   explicit, liftable integration sections), the vendored awslabs base under
   `reference/`, `setup_deepep_gin.sh`, `validation-gate.yaml`, `SBOM.txt`, and a
   README covering design decisions, recreation, public-image access, and the
   run command.
4. **nccl-tests + NIXL test are in the image.** nccl-tests
   (`all_reduce_perf`/`all_gather_perf`/`alltoall_perf`) is built in the builder
   stage against the GIN NCCL + EFA/OMPI stack and placed on `PATH`. The NIXL
   tests are the LIBFABRIC and backend-free C++ tests in `/opt/nixl-tests`
   (`libfabric_topology_test`, `rail_active_refcount_test`, `desc_example`,
   `p2p_test`, `serdes_test`, `telemetry_reader`); the final gate checks each one
   resolves and runs `serdes_test`. `nixl_test` is no longer built (it hard-codes
   UCX). The vLLM image has no nccl-tests and no OpenMPI; it carries the same NIXL
   tests, run via `nixl-test <name>`. The NIXL validation suite is `nixl_lf` (§6.3).

---

## 11. Appendix — provenance of this recipe

This image and its validation were developed empirically; the non-obvious facts
baked into the design:

- GIN NCCL `.so` dedup trap → NCCL pinned in the torch layer (D3).
- GDAKI needs `NVreg_EnableStreamMemOPs=1` + `PeerMappingOverride=1` on the host (D5).
- Autoscaler replaces rebooted nodes → bake flags into the launch template, not per-instance (D5).
- DeepEP V2 switched internode from NVSHMEM to **NCCL GIN**; the legacy
  `tests/legacy/test_internode.py` exercises the deprecated NVSHMEM/IBGDA path
  and fails on EFA — use `tests/elastic/test_ep.py` (`ElasticBuffer`).
- `ordered` kernels need strong signals EFA lacks → `NCCL_SYM_GIN_KERNELS_ENABLE=0` (D7).
- EFA driver `>=3.3.0` required (completion-counter API); verify `modinfo efa`.
- `NCCL_NET_PLUGIN` must be the **absolute** `libnccl-net-ofi.so` or NCCL silently
  falls back to TCP Socket (no RDMA).
