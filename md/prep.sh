#!/bin/zsh
OMD=~/miniforge3/envs/openmm-md/bin/omd
cd /Users/cafierom/python_mac/boltzgen_local
for n in bg33_3 bg33_4; do
  M=md/$n
  echo "################ $n"
  $OMD prep-protein --pdb $M/${n}_protein.pdb --out $M/protein_fixed.pdb
  $OMD prep-ligand  --sdf $M/${n}_ligand.sdf  --out $M/ligand_prepped.sdf
  $OMD build --protein $M/protein_fixed.pdb --ligand $M/ligand_prepped.sdf \
             --out-dir $M/system --no-auto-cofactors --box-shape dodecahedron
  echo "BUILD_EXIT[$n]=$?"
done
echo PREP_DONE
