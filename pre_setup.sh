REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p $REPO_ROOT/third_party/wheels

curl -L -o $REPO_ROOT -O https://github.com/mlc-ai/package/releases/download/v0.9.dev0/mlc_ai_cpu-0.20.0-py3-none-manylinux_2_28_x86_64.whl
curl -L -o $REPO_ROOT -O https://files.pythonhosted.org/packages/93/e9/923843463730aa1add10c26b45110fb6a13a68dfdb48e1cd9e325b04e331/apache_tvm_ffi-0.1.12-cp311-cp311-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl
mv $REPO_ROOT/apache_tvm_ffi-0.1.12-cp311-cp311-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl $REPO_ROOT/third_party/wheels/apache_tvm_ffi-0.1.10rc1-cp311-cp311-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl
mv $REPO_ROOT/mlc_ai_cpu-0.20.0-py3-none-manylinux_2_28_x86_64.whl $REPO_ROOT/third_party/wheels/