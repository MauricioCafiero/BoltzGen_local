#!/bin/zsh
# 20 ns of dynamics plus MM/GBSA for two BoltzGen designs, using peptidebuilder's protocol.
#
# Order is bg33_3 (bundle model 4) then bg33_4 (model 1). Every stage is guarded by its own
# output, so re-running this resumes rather than restarting: an interrupted run loses at most
# the stage it was in. `omd run` itself cannot resume -- dynamics.py writes a checkpoint but
# nothing loads it -- so an interrupted trajectory restarts from zero.
#
# Measured here: 1.11 ms/step at 5,088 particles, 156 ns/day, against the 1.72 ms/step at
# 8,647 particles in peptidebuilder's README. So 20 ns is about 3.1 h for bg33_3 and 4.4 h
# for bg33_4.
cd /Users/cafierom/python_mac/boltzgen_local
OMD=~/miniforge3/envs/openmm-md/bin/omd
PY=~/miniforge3/envs/openmm-md/bin/python
REPO=/Users/cafierom/python_mac/peptidebuilder
STEPS=10000000           # 20 ns at 2 fs
ROOT=$PWD              # absolute, so MM/GBSA can be run from inside its own output dir
caffeinate -w $$ &
date

# Legs default to the three recorded here; pass names to tail only specific legs, e.g.
# `./run_md20.sh bg33_1` after fetching a new trajectory whose stages are otherwise unprocessed.
# (bg33_2's tail, including its windows and contacts, ran 2026-10-04.)
LEGS="${*:-bg33_2 bg33_3 bg33_4}"
for n in ${=LEGS}; do
    M=md/$n
    P=$M/prod_20ns
    echo "################ $n"

    # Skip the run if EITHER the full dcd (local legs) or the wrapped solute (racc legs, stripped on
    # the cluster -- their full dcd never comes to this Mac) is present: both mean dynamics is done.
    # Without the traj_wrapped.xtc clause a fetched racc leg starts a fresh 20 ns run on the laptop,
    # which is the same trap run_dock_pose_md.sh documents at its own run guard.
    if [[ ! -f $P/traj.dcd && ! -f $P/traj_wrapped.xtc ]]; then
        $OMD run --system $M/system/system.xml --topology $M/system/complex.pdb \
                 --out-dir $P --steps $STEPS --platform OpenCL
        echo "RUN_EXIT[$n]=$?"
        date
    else
        echo "trajectory present in $P (traj.dcd or traj_wrapped.xtc), skipping the dynamics"
    fi

    if [[ ! -f $P/traj_wrapped.xtc ]]; then
        $OMD analyze --traj $P/traj.dcd --topology $M/system/complex.pdb --out-dir $P
        echo "ANALYZE_EXIT[$n]=$?"
    fi

    if [[ ! -f $P/mmgbsa/FINAL_RESULTS_MMPBSA.dat ]]; then
        # from inside the output dir: MMPBSA.py scatters scratch into the working directory
        ( cd $P && $OMD mmgbsa --protein $ROOT/$M/protein_fixed.pdb \
                               --ligand $ROOT/$M/ligand_prepped.sdf \
                               --traj traj_wrapped.xtc --topology traj_wrapped.pdb \
                               --out-dir mmgbsa --no-auto-cofactors --run )
        echo "MMGBSA_EXIT[$n]=$?"
        date
    fi

    # Leading windows of the same trajectory. peptidebuilder's convergence test on orig_f12
    # gave -25.37 at 40 ps, -21.52 at 5 ns, -16.86 at 10 ns, -14.61 at 15 ns and -13.71 at
    # 20 ns: every short window reads too negative, because each is still measuring the
    # predicted pose rather than the ensemble. The series is what shows whether 20 ns sufficed.
    for ns in 5 10 15; do
        W=$M/first_${ns}ns
        [[ -f $W/mmgbsa/FINAL_RESULTS_MMPBSA.dat ]] && continue
        mkdir -p $W
        # Two sources, by lane. A local leg has the full-box $P/traj.dcd, so slice and wrap it. A racc
        # leg has only the already-wrapped solute -- the full box stays on cluster scratch -- so slice
        # that directly: no full-box load, which is what OOM'd this 8 GB Mac at 4-5 GB once.
        if [[ ! -f $W/traj_wrapped.xtc ]]; then
            if [[ -f $P/traj.dcd ]]; then
                if [[ ! -f $W/traj.dcd ]]; then
                    $PY -c "
import mdtraj as md
t = md.load('$P/traj.dcd', top='$M/system/complex.pdb')
keep = t[: int(len(t) * $ns / 20)]
keep.save_dcd('$W/traj.dcd')
print(f'kept {len(keep)} of {len(t)} frames (leading $ns ns)')
"
                fi
                $OMD analyze --traj $W/traj.dcd --topology $M/system/complex.pdb --out-dir $W
            else
                $PY -c "
import mdtraj as md
t = md.load('$P/traj_wrapped.xtc', top='$P/traj_wrapped.pdb')
keep = t[: int(len(t) * $ns / 20)]
keep.save_xtc('$W/traj_wrapped.xtc')
keep[0].save_pdb('$W/traj_wrapped.pdb')
print(f'kept {len(keep)} of {len(t)} wrapped frames (leading $ns ns)')
"
            fi
        fi
        ( cd $W && $OMD mmgbsa --protein $ROOT/$M/protein_fixed.pdb \
                               --ligand $ROOT/$M/ligand_prepped.sdf \
                               --traj traj_wrapped.xtc --topology traj_wrapped.pdb \
                               --out-dir mmgbsa --no-auto-cofactors --run )
        echo "WINDOW_EXIT[$n:${ns}ns]=$?"
        date
    done

    # What the ligand did over the run, which one dG cannot show.
    $PY $REPO/code/md_contacts.py $P --csv $M/md_contacts.csv || true
    # End-of-run ensemble (pipeline step per CLAUDE.md: contacts and frames, both, on every leg).
    $PY $REPO/code/md_frames.py $P || true
    echo "DONE[$n]"
    date
done
echo MD_ALL_DONE
