"""Rewrite a BoltzGen-written .cif into the column layout Boltz-2 writes.

The two differ by one column: BoltzGen's `_atom_site` loop carries `label_entity_id`,
Boltz's does not, so `code/uma_binding.py:parse_cif` -- which reads fields positionally --
would take the entity id as the residue number and collapse every residue into one.
Everything else (element, atom name, comp id, coordinates) already lines up.

Usage: python bg_to_boltz_cif.py <in.cif> <out.cif>
"""
import sys


def read_atom_site(path):
    cols, rows = [], []
    in_loop = False
    for line in open(path):
        s = line.strip()
        if s.startswith("_atom_site."):
            cols.append(s.split(".", 1)[1])
            in_loop = True
            continue
        if in_loop:
            if s.startswith(("ATOM", "HETATM")):
                rows.append(s.split())
            elif s and not s.startswith("#"):
                break
    return [dict(zip(cols, r)) | {"group": r[0]} for r in rows]


def main(src, dst):
    atoms = read_atom_site(src)
    if not atoms:
        sys.exit(f"no _atom_site rows in {src}")
    with open(dst, "w") as fh:
        fh.write("data_model\n_entry.id model\n#\nloop_\n")
        for c in ("group_PDB", "id", "type_symbol", "label_atom_id", "label_alt_id",
                  "label_comp_id", "label_alt_id2", "label_seq_id", "pdbx_PDB_ins_code",
                  "label_asym_id", "Cartn_x", "Cartn_y", "Cartn_z", "occupancy",
                  "label_entity_id", "auth_asym_id", "auth_comp_id", "B_iso_or_equiv",
                  "pdbx_PDB_model_num"):
            fh.write(f"_atom_site.{c}\n")
        for i, a in enumerate(atoms, 1):
            fh.write(" ".join([
                a["group"], str(i), a["type_symbol"], a["label_atom_id"], ".",
                a["label_comp_id"], ".", a["label_seq_id"], "?", a["label_asym_id"],
                a["Cartn_x"], a["Cartn_y"], a["Cartn_z"], "1",
                a.get("label_entity_id", "1"), a["label_asym_id"], a["label_comp_id"],
                a.get("B_iso_or_equiv", "0.00"), "1"]) + "\n")
        fh.write("#\n")
    n_het = sum(a["group"] == "HETATM" for a in atoms)
    print(f"{src} -> {dst}: {len(atoms)} atoms, {n_het} ligand")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
