"""Which residues touch which part of the ligand, per fold.

Wrapping counts ligand atoms without saying what is being gripped, and for this ligand that
matters: a peptide can stack the methoxyphenyl ring, bury the 2-ethylhexyl tail, or sit on the
ester in the middle, and those are different binding modes with different prospects in water.
The ligand is split into moieties from its own SMILES, so nothing is hard-coded.

Usage: python contact_map.py <smiles> <cif> [<cif> ...]
"""
import sys
from collections import defaultdict

import numpy as np
from rdkit import Chem

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lig_geom import read_ligand  # noqa: E402

AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
       "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
       "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def moieties(smiles):
    """{atom name: moiety} for the ligand, named as a Boltz-family cif names its atoms."""
    mol = Chem.RemoveHs(Chem.MolFromSmiles(smiles))
    counts, names = {}, []
    for atom in mol.GetAtoms():
        s = atom.GetSymbol()
        counts[s] = counts.get(s, 0) + 1
        names.append(f"{s}{counts[s]}")

    group = {}
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        if atom.GetIsAromatic():
            group[names[i]] = "ring"
        elif atom.GetSymbol() == "O" and any(n.GetIsAromatic() for n in atom.GetNeighbors()):
            group[names[i]] = "methoxy"          # the ring's own ether oxygen
    for atom in mol.GetAtoms():                  # methyl hanging off that oxygen
        i = atom.GetIdx()
        if names[i] in group:
            continue
        if any(group.get(names[n.GetIdx()]) == "methoxy" for n in atom.GetNeighbors()):
            group[names[i]] = "methoxy"
    for atom in mol.GetAtoms():                  # the ester, carbonyl carbon outwards
        i = atom.GetIdx()
        if atom.GetSymbol() != "C" or names[i] in group:
            continue
        os_ = [n for n in atom.GetNeighbors() if n.GetSymbol() == "O"]
        if len(os_) == 2:
            group[names[i]] = "ester"
            for n in os_:
                group[names[n.GetIdx()]] = "ester"
    for bond in mol.GetBonds():                  # the cinnamate C=C
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        if (str(bond.GetBondType()) == "DOUBLE" and not a.GetIsAromatic()
                and not b.GetIsAromatic() and a.GetSymbol() == b.GetSymbol() == "C"):
            for at in (a, b):
                group.setdefault(names[at.GetIdx()], "vinyl")
    for n in names:
        group.setdefault(n, "tail")              # 2-ethylhexyl, and its ester oxygen's carbon
    return group, ["ring", "methoxy", "vinyl", "ester", "tail"]


def read_peptide(path):
    """[(resseq, one-letter, atom name, xyz)] for the ATOM records of a cif."""
    out, cols, in_loop = [], [], False
    for line in open(path):
        s = line.strip()
        if s.startswith("_atom_site."):
            cols.append(s.split(".", 1)[1])
            in_loop = True
        elif in_loop and s.startswith("ATOM"):
            a = dict(zip(cols, s.split()))
            out.append((int(a["label_seq_id"]), AA3.get(a["label_comp_id"], "X"),
                        a["label_atom_id"],
                        np.array([float(a["Cartn_x"]), float(a["Cartn_y"]), float(a["Cartn_z"])])))
    return out


def main(smiles, paths, cutoff=4.5):
    group, order = moieties(smiles)
    counts = defaultdict(int)
    for g in group.values():
        counts[g] += 1
    print("ligand moieties: " + ", ".join(f"{g} {counts[g]}" for g in order if counts[g]))
    for path in paths:
        lig = read_ligand(path)
        pep = read_peptide(path)
        engaged = defaultdict(set)      # moiety -> ligand atoms contacted
        by_res = defaultdict(set)       # residue -> moieties it contacts
        backbone = defaultdict(int)
        for name, lp in lig.items():
            for resseq, aa, atom, pp in pep:
                if np.linalg.norm(lp - pp) < cutoff:
                    engaged[group[name]].add(name)
                    by_res[(resseq, aa)].add(group[name])
                    if atom in ("N", "CA", "C", "O"):
                        backbone[group[name]] += 1
        print(f"\n{path.split('/')[-1]}")
        print("  " + "  ".join(f"{g} {len(engaged[g])}/{counts[g]}" for g in order if counts[g]))
        tot = sum(len(v) for v in engaged.values())
        bb = sum(backbone.values())
        print(f"  {len(by_res)} residues in contact, {tot}/{len(lig)} ligand atoms, "
              f"{bb} of the contacts made by backbone atoms")
        for (resseq, aa), gs in sorted(by_res.items()):
            print(f"    {aa}{resseq:<3} {' '.join(sorted(gs))}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
