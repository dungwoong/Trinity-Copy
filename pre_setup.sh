REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p $REPO_ROOT/third_party/wheels

curl -L --output-dir "$REPO_ROOT" -O https://github.com/mlc-ai/package/releases/download/v0.9.dev0/mlc_ai_cpu-0.20.0-py3-none-manylinux_2_28_x86_64.whl
curl -L --output-dir "$REPO_ROOT" -O https://files.pythonhosted.org/packages/72/9c/af12a5e796a672664f2f18eca989222697b91974a9c9e98d4ec2ecfeeb83/apache_tvm_ffi-0.1.11-cp311-cp311-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl
mv $REPO_ROOT/apache_tvm_ffi-0.1.11-cp311-cp311-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl $REPO_ROOT/third_party/wheels/apache_tvm_ffi-0.1.10rc1-cp311-cp311-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl
mv $REPO_ROOT/mlc_ai_cpu-0.20.0-py3-none-manylinux_2_28_x86_64.whl $REPO_ROOT/third_party/wheels/