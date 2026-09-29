"""Is the ligand geometry in a predicted complex actually broken, or does it only look it?

Bond lengths are measured against an MMFF-relaxed conformer of the same SMILES. The atom
names BoltzGen and Boltz write are element plus a per-element count over the heavy-atom
molecule in RDKit order, which is what lets a name in the cif be matched to a bond in the
reference molecule.

Usage: python lig_geom.py <smiles> <cif> [<cif> ...]
"""
import sys

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def named_mol(smiles):
    """Heavy-atom Mol with the atom names a Boltz-family cif uses, and a 3D conformer."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
    AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    mol = Chem.RemoveHs(mol)
    counts = {}
    for atom in mol.GetAtoms():
        s = atom.GetSymbol()
        counts[s] = counts.get(s, 0) + 1
        atom.SetProp("name", f"{s}{counts[s]}")
    return mol


def read_ligand(path, order=None):
    """{atom name: xyz} for the HETATM records of a cif or pdb.

    `order` replaces the file's own atom names with that list, positionally. Boltz names
    SMILES ligand atoms by a global index (`C38`) where BoltzGen names them by a per-element
    count (`C17`), but both write the atoms in the molecule's own order, so passing the
    reference names in order matches one file's atoms to the other's naming scheme.
    """
    out, cols, in_loop = {}, [], False
    for line in open(path):
        s = line.rstrip("\n")
        if s.strip().startswith("_atom_site."):
            cols.append(s.strip().split(".", 1)[1])
            in_loop = True
        elif in_loop and s.startswith("HETATM"):
            f = s.split()
            a = dict(zip(cols, f))
            name = order[len(out)] if order else a["label_atom_id"]
            out[name] = np.array(
                [float(a["Cartn_x"]), float(a["Cartn_y"]), float(a["Cartn_z"])])
        elif s.startswith("HETATM") and not in_loop:          # plain PDB
            name = order[len(out)] if order else s[12:16].strip()
            out[name] = np.array([float(s[30:38]), float(s[38:46]), float(s[46:54])])
    return out


def main(smiles, paths):
    by_order = "--by-order" in paths
    paths = [p for p in paths if not p.startswith("--")]
    ref = named_mol(smiles)
    conf = ref.GetConformer()
    bonds = []
    for b in ref.GetBonds():
        i, j = b.GetBeginAtom(), b.GetEndAtom()
        ideal = float(np.linalg.norm(np.array(conf.GetAtomPosition(i.GetIdx())) -
                                     np.array(conf.GetAtomPosition(j.GetIdx()))))
        bonds.append((i.GetProp("name"), j.GetProp("name"), ideal, str(b.GetBondType())))
    ring = [a.GetProp("name") for a in ref.GetAtoms() if a.GetIsAromatic()]
    print(f"reference: {ref.GetNumAtoms()} heavy atoms, {len(bonds)} bonds, "
          f"aromatic ring {'-'.join(ring)}")
    print(f"\n{'structure':<24}{'n':>4}{'bond err A':>12}{'worst':>8}{'>0.15':>7}"
          f"{'>0.30':>7}{'ring rmsd':>11}  worst bond")
    for path in paths:
        names = [a.GetProp("name") for a in ref.GetAtoms()] if by_order else None
        lig = read_ligand(path, order=names)
        missing = [n for n, *_ in bonds if n not in lig] + [n for _, n, *_ in bonds if n not in lig]
        if missing:
            print(f"{path.split('/')[-1][:24]:<24}  atom names not matched: "
                  f"{sorted(set(missing))[:6]}")
            continue
        errs, worst = [], ("", 0.0, 0.0)
        for n1, n2, ideal, kind in bonds:
            d = float(np.linalg.norm(lig[n1] - lig[n2]))
            errs.append(abs(d - ideal))
            if abs(d - ideal) > abs(worst[2] - worst[1]) or not worst[0]:
                worst = (f"{n1}-{n2} ({kind.lower()})", ideal, d)
        errs = np.array(errs)
        # Ring planarity: rmsd of the six aromatic atoms from their own best-fit plane.
        R = np.array([lig[n] for n in ring])
        R = R - R.mean(0)
        plane_rmsd = float(np.sqrt((np.linalg.svd(R, compute_uv=False)[-1] ** 2) / len(R)))
        print(f"{path.split('/')[-1][:24]:<24}{len(errs):>4}{errs.mean():>12.3f}"
              f"{errs.max():>8.3f}{int((errs > 0.15).sum()):>7}{int((errs > 0.30).sum()):>7}"
              f"{plane_rmsd:>11.3f}  {worst[0]} {worst[2]:.2f} vs {worst[1]:.2f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
