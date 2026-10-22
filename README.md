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
| `awslabs-contrib/architectures/containers/deepv2-nixl-efa/Dockerfile` | the image (multi-arch, all build fixes + license hygiene) |
| `.../setup_deepep_gin.sh` | DeepEP GIN build helper |
| `.../check_licenses.py` + `.../validation-gate.yaml` | throughput floors + AWS OSS license gate |
| `.../README.md` | recipe-level detail |
| `deepep-nixl-pipeline-design.md` | full design doc (decisions, build fixes, pipeline) |
| `*.yaml` (top level) | k8s validation manifests + node probes |

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

The build is self-checking: in-build gates fail the build if the GIN NCCL
library is stale or a backend plugin is missing, so a green build is already
partly validated. Push to your registry when it succeeds:

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
  Bake these into the nodegroup launch template userData so replacement nodes
  keep them. Verify on a node:
  ```bash
  grep -E "EnableStreamMemOPs|RegistryDwords" /proc/driver/nvidia/params
  ```
- The kubeflow **mpi-operator** CRD installed (`kubectl get crd mpijobs.kubeflow.org`).
- The built image pushed to a registry your cluster can pull.

Set the image reference in the manifests below to your pushed image before applying.

### Run the three suites

Each is a 2-node / 16-rank job. Apply, wait for the launcher pod to reach
`Completed`, then read its logs for the success markers.

```bash
# 1. NCCL collectives — all-reduce, all-gather, all-to-all over EFA
kubectl apply -f nccl-tests-validate.yaml

# 2. DeepEP V2 — expert-parallel dispatch/combine over NCCL GIN / EFA-GDA
kubectl apply -f deepep-validate.yaml

# 3. NIXL — point-to-point transfer over the LIBFABRIC/EFA backend
kubectl apply -f nixl-test-target-validate.yaml      # start the target first
kubectl apply -f nixl-test-initiator-validate.yaml   # then the initiator
```

### Success markers (what a pass looks like)

| Suite | Pass condition |
|---|---|
| NCCL | every rank logs `NET/OFI Selected provider is efa … (found 16 nics)`; busbw reported; exit 0 |
| DeepEP V2 | `Loaded gin plugin Libfabric_GDAKI`; no `host_register_iomem failed`; dispatch + combine complete on all 16 ranks |
| NIXL | LIBFABRIC backend selected; `Transfer completed and Buffers match`; exit 0 on both pods |

Throughput floors for sign-off live in `validation-gate.yaml`; the gate compares
a 3-run mean against each floor and blocks a regression.

See `deepep-nixl-pipeline-design.md` for the full design, the six build fixes,
and the publish/provenance flow.
