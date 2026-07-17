#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR/backend"

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

# Default cost/kern if not provided by env.
cost="${cost:-6}"
kern="${kern:-2}"

for model_name in "${models[@]}"; do
  echo "==> ${model_name} (cost=${cost}, kern=${kern})"
  python -m profile.benchmark \
    --ir "../optimizer/expressions/${model_name}_cost${cost}_kern${kern}.txt" \
    --shapes "../frontend/outputs/trinity/${model_name}/shapes.json"
  echo
done

# Example of running a single model:
# python -m profile.benchmark \
#    --ir "../optimizer/expressions/Roco_cost6_kern2.txt" \
#    --shapes "../frontend/outputs/trinity/Roco/shapes.json"