#!/usr/bin/env bash
# Environment for the NIXL + DeepEP V2 + EFA benchmarks.
#
# Copy to env.sh, fill in YOUR values, then:
#   source env.sh
#   envsubst < benchmarks/<suite>/<file>.yaml | kubectl apply -f -
#
# Every manifest, the recipe README, the design doc, and validation-gate.yaml
# use these ${VAR} placeholders instead of hardcoded account/cluster values.
# env.sh is gitignored; this .example file is the committed template.

# --- AWS account / regions -------------------------------------------------
export AWS_ACCOUNT_ID="123456789012"      # account that owns the ECR repo + cluster
export AWS_REGION="us-west-2"             # region of the dev ECR repo + the EKS cluster
export ECR_REGION="us-east-1"            # region of the PUBLISHED image repo
export ECR_TOKEN_REGION="us-east-1"      # region used only to mint the ECR auth token
                                          #   (some accounts require a specific one)
export OTHER_ACCOUNT_ID="000000000000"    # a different account (doc reference only)

# --- AWS auth --------------------------------------------------------------
export AWS_PROFILE="default"             # CLI profile with ECR push + cluster access

# --- ECR repositories ------------------------------------------------------
export ECR_REPO="my-team/nixl-deepep-efa"              # dev image repo path
export PUBLISH_REPO="my-org/deepepv2-nixl-efa"         # published image repo path
export IMAGE_DIGEST="sha256:0000000000000000000000000000000000000000000000000000000000000000"

# --- Kubernetes ------------------------------------------------------------
export KUBE_CONTEXT="my-eks-context"     # kubectl context for the cluster
export EKS_CLUSTER="my-eks-cluster"      # EKS cluster name
export K8S_NAMESPACE="default"           # namespace the benchmark pods run in

# --- Benchmark nodes (2-node jobs pin to specific nodes) -------------------
# Full Kubernetes node names, e.g. ip-10-0-1-23.us-west-2.compute.internal
export NODE_0="<node-0-name>"
export NODE_1="<node-1-name>"
