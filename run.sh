#!/bin/zsh
# One BoltzGen run, logged, with the machine held awake for its duration.
#
#   ./run.sh <spec> <output-name> [num_designs]
#
# e.g.  ./run.sh specs/octinoxate33.yaml octx33 6
#
# Logs to <output-name>.log. Re-running the same output name resumes rather than
# restarting: --reuse keeps whatever the previous attempt finished.
#
# IF_NUM_SEQ=<n> raises the inverse-folding ensemble. BoltzGen's own default is 1, which
# is why the first octinoxate and oxybenzone sets came back half poly-alanine: one
# sequence per backbone leaves the filtering stage nothing to select among. Upstream's
# other code path (cli/boltzgen.py:1029) assumes 10. Folding cost scales with it, since
# every sequence in the ensemble is folded.
set -e

HERE=${0:A:h}
SPEC=${1:?usage: run.sh <spec> <output-name> [num_designs]}
NAME=${2:?usage: run.sh <spec> <output-name> [num_designs]}
N=${3:-6}
PROTOCOL=${PROTOCOL:-peptide-anything}
IF_NUM_SEQ=${IF_NUM_SEQ:-}
# An array, not ${IF_NUM_SEQ:+--flag $IF_NUM_SEQ}: zsh does not word-split an unquoted
# expansion, so that form reaches argparse as one argv element with a space in it and is
# rejected as an unknown option. An empty array expands to no words at all.
# An `if`, not `[[ ... ]] && ...`: under `set -e` a false test makes the whole line the
# script's last status and exits it, which would break every run that leaves IF_NUM_SEQ unset.
typeset -a IF_ARGS
if [[ -n $IF_NUM_SEQ ]]; then
    IF_ARGS=(--inverse_fold_num_sequences $IF_NUM_SEQ)
fi

cd $HERE
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OMP_NUM_THREADS=4

# Dies with this script, and keeps the machine from idling out mid-run.
caffeinate -w $$ &

{
    date
    # precision=32 is not optional on MPS. The shipped bf16-mixed leaves the model's
    # ~50 torch.autocast("cuda", enabled=False) float32 guards inert under an MPS
    # autocast, and the ligand comes out with bond lengths wrong by up to 0.9 A --
    # measured, see the Traps section of README.md.
    boltzgen run $SPEC \
        --output workbench/$NAME \
        --protocol $PROTOCOL \
        --num_designs $N \
        --devices 1 \
        --reuse \
        $IF_ARGS \
        --config design trainer.precision=32 \
        --config folding trainer.precision=32
    echo "EXIT=$?"
    date
} 2>&1 | tee $NAME.log
