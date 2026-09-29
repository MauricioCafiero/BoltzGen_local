#!/bin/zsh
# One BoltzGen run, logged, with the machine held awake for its duration.
#
#   ./run.sh <spec> <output-name> [num_designs]
#
# e.g.  ./run.sh specs/octinoxate33.yaml octx33 6
#
# Logs to <output-name>.log. Re-running the same output name resumes rather than
# restarting: --reuse keeps whatever the previous attempt finished.
set -e

HERE=${0:A:h}
SPEC=${1:?usage: run.sh <spec> <output-name> [num_designs]}
NAME=${2:?usage: run.sh <spec> <output-name> [num_designs]}
N=${3:-6}
PROTOCOL=${PROTOCOL:-peptide-anything}

cd $HERE
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OMP_NUM_THREADS=4

# Dies with this script, and keeps the machine from idling out mid-run.
caffeinate -w $$ &

{
    date
    boltzgen run $SPEC \
        --output workbench/$NAME \
        --protocol $PROTOCOL \
        --num_designs $N \
        --devices 1 \
        --reuse
    echo "EXIT=$?"
    date
} 2>&1 | tee $NAME.log
