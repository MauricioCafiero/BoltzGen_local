#!/bin/zsh
# Compute the MM/GBSA windows while the dynamics is still running.
#
# The dynamics is on OpenCL and holds one CPU core; a window is analyze + prmtop build +
# MMPBSA.py over a few hundred frames of a 400-atom solute, which took 92 s in this project's
# own records. So the windows are free on compute. What is not free is memory: the machine is
# already swapping under the run, so each window waits for its turn, reports the memory state
# before it starts, and slices by streaming rather than loading the whole trajectory.
#
# run_md20.sh skips any window whose FINAL_RESULTS_MMPBSA.dat already exists, so whatever this
# finishes early is simply not redone at the end.
#
#   md/windows_live.sh <structure> [total_ns]
cd /Users/cafierom/python_mac/boltzgen_local
OMD=~/miniforge3/envs/openmm-md/bin/omd
PY=~/miniforge3/envs/openmm-md/bin/python
ROOT=$PWD              # absolute, so MM/GBSA can be run from inside its own output dir

n=${1:?usage: windows_live.sh <structure> [total_ns]}
TOTAL=${2:-20}
M=md/$n
P=$M/prod_${TOTAL}ns

for ns in 5 10 15; do
    W=$M/first_${ns}ns
    if [[ -f $W/mmgbsa/FINAL_RESULTS_MMPBSA.dat ]]; then
        echo "[$ns ns] already done"
        continue
    fi
    # Wait until the trajectory is past this window, with a margin so the slice is not at the
    # very head of the file being appended to.
    echo "[$ns ns] waiting for the run to pass $((ns + 1)) ns"
    while :; do
        [[ -f $P/energy.csv ]] || { sleep 60; continue }
        step=$(tail -1 $P/energy.csv | cut -d, -f1)
        [[ $step -gt $(( (ns + 1) * 500000 )) ]] && break
        sleep 120
    done
    mkdir -p $W
    echo "[$ns ns] starting at $(date), step $step, $(memory_pressure | tail -1)"
    $PY md/window_live.py $P/traj.dcd $M/system/complex.pdb $W/traj.dcd $ns $P/energy.csv || continue
    $OMD analyze --traj $W/traj.dcd --topology $M/system/complex.pdb --out-dir $W
    # MMPBSA.py writes its scratch into the *working* directory, not --out-dir: one 20 ns
    # window left a 6 GB reference.frc and a dozen _MMPBSA_* files in the repo root. Running
    # it from inside the window's own directory keeps that where it belongs.
    ( cd $W && $OMD mmgbsa --protein $ROOT/$M/protein_fixed.pdb \
                           --ligand $ROOT/$M/ligand_prepped.sdf \
                           --traj traj_wrapped.xtc --topology traj_wrapped.pdb \
                           --out-dir mmgbsa --no-auto-cofactors --run )
    echo "[$ns ns] done at $(date), $(memory_pressure | tail -1)"
    grep -A3 "DELTA TOTAL" $W/mmgbsa/FINAL_RESULTS_MMPBSA.dat 2>/dev/null | head -4
done
echo "WINDOWS_LIVE_DONE[$n]"
