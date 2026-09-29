#!/bin/zsh
# 10 ns of dynamics plus MM/GBSA for two BoltzGen designs, using peptidebuilder's protocol.
#
# Order is bg33_3 (bundle model 4) then bg33_4 (model 1). Every stage is guarded by its own
# output, so re-running this resumes rather than restarting: an interrupted run loses at most
# the stage it was in.
#
# The 5 ns window is computed from the same trajectory because peptidebuilder's own
# convergence test says every short window reads too negative -- orig_f12 gave -16.86 at 10 ns
# against -13.71 at 20 ns. Having 5 and 10 side by side at least shows which way it is still
# moving.
cd /Users/cafierom/python_mac/boltzgen_local
OMD=~/miniforge3/envs/openmm-md/bin/omd
PY=~/miniforge3/envs/openmm-md/bin/python
REPO=/Users/cafierom/python_mac/peptidebuilder
STEPS=5000000            # 10 ns at 2 fs
caffeinate -w $$ &
date

for n in bg33_3 bg33_4; do
    M=md/$n
    P=$M/prod_10ns
    echo "################ $n"

    if [[ ! -f $P/traj.dcd ]]; then
        $OMD run --system $M/system/system.xml --topology $M/system/complex.pdb \
                 --out-dir $P --steps $STEPS --platform OpenCL
        echo "RUN_EXIT[$n]=$?"
        date
    else
        echo "$P/traj.dcd exists, skipping the dynamics"
    fi

    if [[ ! -f $P/traj_wrapped.xtc ]]; then
        $OMD analyze --traj $P/traj.dcd --topology $M/system/complex.pdb --out-dir $P
        echo "ANALYZE_EXIT[$n]=$?"
    fi

    if [[ ! -f $P/mmgbsa/FINAL_RESULTS_MMPBSA.dat ]]; then
        $OMD mmgbsa --protein $M/protein_fixed.pdb --ligand $M/ligand_prepped.sdf \
                    --traj $P/traj_wrapped.xtc --topology $P/traj_wrapped.pdb \
                    --out-dir $P/mmgbsa --no-auto-cofactors --run
        echo "MMGBSA_EXIT[$n]=$?"
        date
    fi

    # Leading half of the same trajectory, as the convergence check.
    W=$M/first_5ns
    if [[ ! -f $W/mmgbsa/FINAL_RESULTS_MMPBSA.dat ]]; then
        mkdir -p $W
        [[ -f $W/traj.dcd ]] || $PY - <<PY
import mdtraj as md
t = md.load("$P/traj.dcd", top="$M/system/complex.pdb")
half = t[: len(t) // 2]
half.save_dcd("$W/traj.dcd")
print(f"kept {len(half)} of {len(t)} frames (leading 5 ns)")
PY
        $OMD analyze --traj $W/traj.dcd --topology $M/system/complex.pdb --out-dir $W
        $OMD mmgbsa --protein $M/protein_fixed.pdb --ligand $M/ligand_prepped.sdf \
                    --traj $W/traj_wrapped.xtc --topology $W/traj_wrapped.pdb \
                    --out-dir $W/mmgbsa --no-auto-cofactors --run
        echo "WINDOW_EXIT[$n]=$?"
        date
    fi

    # What the ligand did over the run, which one dG cannot show.
    $PY $REPO/code/md_contacts.py $P --csv $M/md_contacts.csv || true
    echo "DONE[$n]"
    date
done
echo MD_ALL_DONE
