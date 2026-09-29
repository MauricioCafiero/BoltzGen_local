#!/bin/zsh
# Put BoltzGen refolds through peptidebuilder's own fold check, then bundle them for viewing.
#
#   tools/metrics.sh <out-dir-for-the-check> <refold cif> [<refold cif> ...]
#
# peptidebuilder's check_fold.py wants a run directory laid out the way Boltz writes one, and
# reads cif fields positionally, so each refold is first converted out of BoltzGen's column
# layout (which carries an extra label_entity_id) and dropped into that structure.
set -e

HERE=${0:A:h}
REPO=${PEPTIDEBUILDER:-$HOME/python_mac/peptidebuilder}
PY=$REPO/.venv/bin/python
SMILES=${SMILES:-'CCCC[C@H](CC)OC(=O)/C=C/c1ccc(OC)cc1'}

DEST=${1:?usage: metrics.sh <out-dir> <refold cif>...}
shift

rm -rf $DEST
i=0
typeset -a CONVERTED
for f in "$@"; do
    name=$(basename $f .cif)
    d=$DEST/boltz/boltz_results_$name/predictions/$name
    mkdir -p $d
    $PY $HERE/bg_to_boltz_cif.py $f $d/${name}_model_0.cif
    CONVERTED+=("$d/${name}_model_0.cif")
    i=$((i+1))
done

$PY $REPO/code/check_fold.py $DEST --quiet
$PY $HERE/lig_geom.py $SMILES "$@"
$PY $HERE/bg_bundle_pdb.py $DEST/bundle.pdb $DEST/boltz/fold_check.csv $SMILES "${CONVERTED[@]}"
