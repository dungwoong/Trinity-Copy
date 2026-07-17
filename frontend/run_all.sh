#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# not working yet
# echo
# python "$ROOT_DIR/model/decoder_gqa_rotary_kv.py"
# echo
# python "$ROOT_DIR/model/moe_lora_parallel.py"

echo
python "$ROOT_DIR/model/encoder_alibi_prenorm.py"
echo
python "$ROOT_DIR/model/encdec_crossattn_swiglu.py"
echo


echo
python "$ROOT_DIR/model/Decoder.py"
echo
python "$ROOT_DIR/model/Roco.py"
echo
python "$ROOT_DIR/model/falcon7b.py"
echo
python "$ROOT_DIR/model/llama3_8b_gqa.py"
echo
python "$ROOT_DIR/model/llama8b.py"
echo
python "$ROOT_DIR/model/keyformer.py"
echo
python "$ROOT_DIR/model/keyformer_prenorm.py"
echo
python "$ROOT_DIR/model/Roco_prenorm.py"
echo
python "$ROOT_DIR/model/qknorm.py"
echo
python "$ROOT_DIR/model/qknorm_prenorm.py"
echo
python "$ROOT_DIR/model/prenorm.py"
echo
python "$ROOT_DIR/model/rmsnorm.py"
echo
python "$ROOT_DIR/model/FFN.py"
echo
python "$ROOT_DIR/model/EncAttn.py"
echo
python "$ROOT_DIR/model/DecAttn.py"

