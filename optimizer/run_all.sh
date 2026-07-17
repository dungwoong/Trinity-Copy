#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR/optimizer"

models=(
  "DecAttn"
  "Decoder"
  "EncAttn"
  "encdec_crossattn_swiglu"
  "encoder_alibi_prenorm"
  "falcon7b"
  "FFN"
  "keyformer_prenorm"
  "keyformer"
  "llama3_8b_gqa"
  "llama8b"
  "prenorm"
  "qknorm_prenorm"
  "qknorm"
  "rmsnorm"
  "Roco_prenorm"
  "Roco"
)

for model_name in "${models[@]}"; do
  echo "==> TRINITY_MODEL_NAME=$model_name"
  TRINITY_MODEL_NAME="$model_name" cargo test --test test_custom_model count_all -- --nocapture
  echo
done
TRINITY_MODEL_NAME="rmsnorm" cargo test --test test_custom_model optimizer_custom_model -- --nocapture