#!/bin/bash

MODELS=("falcon" "llama")
METHODS=("vanilla" "prenorm" "qknorm" "keyformer" "roco" "ffn")

for model in "${MODELS[@]}"; do
    for method in "${METHODS[@]}"; do
        if [ "$method" = "ffn" ]; then
            CMD="RUSTFLAGS=\"-A warnings\" cargo test --test ${method} ${model}_extract_ffn_expressions -- --nocapture"
        else
            CMD="RUSTFLAGS=\"-A warnings\" cargo test --test ${method} ${model}_extract_rmsnorm_qkv_attn_expressions -- --nocapture"
        fi
        echo ">>> Running: $model + $method"
        eval $CMD
    done
done