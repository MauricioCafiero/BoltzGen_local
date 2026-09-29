#!/bin/zsh
# Fetch BoltzGen, patch it for Apple silicon, install it, and check the install.
#
# Everything this does is spelled out in README.md; this is just the same steps in one
# command. Safe to re-run: it refuses to clobber an existing src_boltzgen.
#
#   ./setup.sh [target-dir]        default: src_boltzgen beside this script
set -e

HERE=${0:A:h}
DIR=${1:-$HERE/src_boltzgen}
PIN=a3149cf                       # boltzgen 0.3.2, the commit these fixes were written against

if [[ -e $DIR ]]; then
    echo "$DIR already exists -- remove it first, or pass another target." >&2
    exit 1
fi

git clone https://github.com/HannesStark/boltzgen.git $DIR
git -C $DIR checkout --quiet $PIN
echo "upstream pinned at $(git -C $DIR log --oneline -1)"

git -C $DIR apply --verbose $HERE/mps-fixes.patch
echo "MPS fixes applied"

if [[ ! -d $HERE/.venv ]]; then
    # numpy is pinned at 2.0.2 upstream and has no wheels above CPython 3.12
    if command -v uv >/dev/null; then
        uv venv --python 3.12 $HERE/.venv
    else
        python3.12 -m venv $HERE/.venv
    fi
fi
source $HERE/.venv/bin/activate

if command -v uv >/dev/null; then
    uv pip install -e $DIR
else
    pip install -e $DIR
fi

export PYTORCH_ENABLE_MPS_FALLBACK=1
python - <<'PY'
import torch
print(f"torch {torch.__version__}  mps available: {torch.backends.mps.is_available()}")
PY

# Validates a design spec end to end without loading a model. Downloads the 391 MB
# component dictionary on first run.
boltzgen check $HERE/specs/octinoxate33.yaml

cat <<'EOF'

Installed. Add this to any shell that runs it:

    source .venv/bin/activate
    export PYTORCH_ENABLE_MPS_FALLBACK=1

Then see README.md for the run commands. --devices 1 is not optional.
EOF
