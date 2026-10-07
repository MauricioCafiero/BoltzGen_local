#!/bin/zsh
# Prepare co-folded BoltzGen legs for dynamics: PDBFixer on the peptide, bond orders and hydrogens
# on the ligand, then a solvated dodecahedral system.
#
#   md/prep.sh <leg> [<leg> ...]        e.g.  md/prep.sh bgox31_3 bgox31_4
#
# Legs are named explicitly, never globbed: a directory scan would silently pull in a leg whose
# system is already built and running. Each stage is guarded by its own output, so a re-run resumes.
#
# --box-shape dodecahedron is not optional. `omd build` defaults to a cube and the co-folded
# baselines in peptidebuilder's runs/*/md are dodecahedral, so leaving it to the default both wastes
# ~30% more waters and breaks comparability (CLAUDE.md, Dynamics runs).
set -u
OMD=~/miniforge3/envs/openmm-md/bin/omd
cd /Users/cafierom/python_mac/boltzgen_local

if (( $# == 0 )); then
  echo "usage: md/prep.sh <leg> [<leg> ...]" >&2
  exit 2
fi

for n in "$@"; do
  M=md/$n
  echo "################ $n"
  if [[ ! -d $M ]]; then
    echo "no such leg directory: $M" >&2
    exit 2
  fi
  [[ -f $M/protein_fixed.pdb ]] || \
    $OMD prep-protein --pdb $M/${n}_protein.pdb --out $M/protein_fixed.pdb
  [[ -f $M/ligand_prepped.sdf ]] || \
    $OMD prep-ligand  --sdf $M/${n}_ligand.sdf  --out $M/ligand_prepped.sdf
  if [[ ! -f $M/system/system.xml ]]; then
    $OMD build --protein $M/protein_fixed.pdb --ligand $M/ligand_prepped.sdf \
               --out-dir $M/system --no-auto-cofactors --box-shape dodecahedron
    echo "BUILD_EXIT[$n]=$?"
  else
    echo "system already built, skipping: $M/system/system.xml"
  fi
done
echo PREP_DONE
