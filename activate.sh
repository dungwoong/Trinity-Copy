source /opt/conda/etc/profile.d/conda.sh
conda activate trinity

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$PYTHONPATH:$REPO_ROOT"
export PATH="/opt/rust/cargo/bin:$PATH"
