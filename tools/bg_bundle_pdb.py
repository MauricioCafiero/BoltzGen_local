"""One multi-model PDB from a set of BoltzGen refolds, all superposed on the ligand.

Each MODEL is one design: peptide as ATOM in chain A, ligand as HETATM in chain B. The
ligand is the same molecule in every structure and its atom names are identical, so every
model is rotated onto the first model's ligand -- scrolling through the models then shows
the peptides moving around a ligand that stays put.

REMARKs carry the sequence and the fold-check numbers for the model above them, so the
viewer shows which structure is on screen.

Usage: python bg_bundle_pdb.py <out.pdb> <fold_check.csv> <smiles|-> <cif> [<cif> ...]

Passing the ligand SMILES writes CONECT records for its bonds; passing `-` leaves the
viewer to infer them by distance.
"""
import csv
import sys

import numpy as np

AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
       "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
       "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def read_cif(path):
    """(atoms, sequence) from a BoltzGen-written model cif."""
    cols, atoms = [], []
    in_loop = False
    for line in open(path):
        s = line.strip()
        if s.startswith("_atom_site."):
            cols.append(s.split(".", 1)[1])
            in_loop = True
            continue
        if in_loop:
            if s.startswith(("ATOM", "HETATM")):
                f = s.split()
                a = dict(zip(cols, f))
                a["group"] = f[0]
                atoms.append(a)
            elif s and not s.startswith("#"):
                break
    # Sequence from the ATOM records, so this works on a converted cif that carries no
    # _entity_poly block as well as on one BoltzGen wrote directly.
    seen = {}
    for a in atoms:
        if a["group"] == "ATOM":
            seen.setdefault(int(a["label_seq_id"]), a["label_comp_id"])
    seq = "".join(AA3.get(seen[k], "X") for k in sorted(seen))
    return atoms, seq


def xyz(atoms):
    return np.array([[float(a["Cartn_x"]), float(a["Cartn_y"]), float(a["Cartn_z"])]
                     for a in atoms])


def kabsch(mobile, target):
    """Rotation and translation putting `mobile` onto `target` (both N x 3)."""
    mc, tc = mobile.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((mobile - mc).T @ (target - tc))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1, 1, d]) @ u.T
    return rot, tc - rot @ mc


def ligand_bonds(smiles):
    """[(name, name)] for the ligand's bonds, named as a Boltz-family cif names them."""
    if smiles == "-":
        return []
    from rdkit import Chem
    mol = Chem.RemoveHs(Chem.MolFromSmiles(smiles))
    counts, names = {}, []
    for atom in mol.GetAtoms():
        s = atom.GetSymbol()
        counts[s] = counts.get(s, 0) + 1
        names.append(f"{s}{counts[s]}")
    return [(names[b.GetBeginAtomIdx()], names[b.GetEndAtomIdx()]) for b in mol.GetBonds()]


def main(out, csv_path, smiles, cifs):
    metrics = {r["name"]: r for r in csv.DictReader(open(csv_path))} if csv_path != "-" else {}
    bonds = ligand_bonds(smiles)
    ref_lig = None
    lines = []
    for model, path in enumerate(cifs, 1):
        atoms, seq = read_cif(path)
        lig = [a for a in atoms if a["group"] == "HETATM"]
        pep = [a for a in atoms if a["group"] == "ATOM"]
        lig.sort(key=lambda a: a["label_atom_id"])          # same order in every model
        pos_lig, pos_pep = xyz(lig), xyz(pep)
        if ref_lig is None:
            ref_lig = pos_lig
            rot, shift = np.eye(3), np.zeros(3)
            rmsd = 0.0
        else:
            rot, shift = kabsch(pos_lig, ref_lig)
            rmsd = float(np.sqrt(((pos_lig @ rot.T + shift - ref_lig) ** 2).sum(1).mean()))
        name = path.split("/")[-1].replace("_model_0.cif", "").replace(".cif", "")
        r = metrics.get(name, {})
        lines.append(f"MODEL     {model:>4d}")
        lines.append(f"REMARK   1 {model}. {name}  {seq}")
        if r:
            lines.append(f"REMARK   1 enclosed {float(r['enclosed_fraction']):.2f}  "
                         f"wrapped {float(r['wrapped_fraction']):.2f}  "
                         f"engaged {r['engaged']}/{r['ligand_heavy_atoms']}  "
                         f"Rg {float(r['peptide_rg']):.1f} A  "
                         f"centroid sep {float(r['centroid_separation']):.1f} A  "
                         f"closest {float(r['closest_approach']):.2f} A  "
                         f"helical {float(r['helical_fraction']):.2f}  "
                         f"nonlocal H-bonds {r['nonlocal_hbonds']}")
        lines.append(f"REMARK   1 ligand superposition on model 1: {rmsd:.3f} A")
        serial = 0
        # The ligand is written FIRST, so its serial numbers are 1..N in every model and one
        # CONECT block at the end of the file is valid for all of them. The peptides differ in
        # length, so ligand-last would give it different serials in each model and the
        # connectivity would apply to the wrong atoms from model 2 on.
        lig_serials = {}
        for group, group_atoms, positions, chain in (("HETATM", lig, pos_lig, "B"),
                                                     ("ATOM  ", pep, pos_pep, "A")):
            moved = positions @ rot.T + shift
            for a, (x, y, z) in zip(group_atoms, moved):
                serial += 1
                nm = a["label_atom_id"]
                if group == "HETATM":
                    lig_serials[nm] = serial
                nm = f" {nm:<3s}" if len(nm) < 4 else nm
                comp = "LIG" if group == "HETATM" else a["label_comp_id"]
                res = 1 if group == "HETATM" else int(a["label_seq_id"])
                lines.append(f"{group}{serial:5d} {nm} {comp:>3s} {chain}{res:4d}    "
                             f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          "
                             f"{a['type_symbol']:>2s}")
        lines.append("TER")
        lines.append("ENDMDL")

    # Explicit ligand bonds, so a viewer does not have to guess them from distance. It
    # matters here: some of these predictions have bond lengths wrong by tenths of an
    # angstrom, and a viewer left to guess draws the molecule in pieces rather than
    # drawing the distortion.
    if bonds:
        for n1, n2 in bonds:
            if n1 in lig_serials and n2 in lig_serials:
                lines.append(f"CONECT{lig_serials[n1]:5d}{lig_serials[n2]:5d}")
    lines.append("END")
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"{out}: {len(cifs)} models, {len(bonds)} ligand bonds as CONECT")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
