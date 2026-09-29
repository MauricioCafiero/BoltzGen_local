"""One multi-model PDB of end-of-run frames from several trajectories, superposed on the ligand.

`md_frames.py` writes each leg's frames superposed on its own peptide, which is right for showing
one ligand moving in one binding site. Comparing two designs needs the opposite: the ligand held
still so the two peptides can be seen around it. The ligand is the same molecule in every leg and
comes from the same prepared SDF, so its atom names match and give the correspondence.

Metrics passed with --label are written into REMARKs above each model, so the viewer shows which
structure is on screen and what it scored.

Usage:
    python md_endpoints.py out.pdb LEG=frame.pdb [LEG=frame.pdb ...] [--label LEG="text"]
"""
import sys

import numpy as np

LIG_RES = "LIG"


def read_pdb(path):
    """(records, ligand atom name -> xyz). Records keep their original lines and coordinates."""
    recs, lig = [], {}
    for line in open(path):
        if not line.startswith(("ATOM", "HETATM")):
            continue
        name = line[12:16].strip()
        res = line[17:20].strip()
        xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        recs.append({"line": line.rstrip("\n"), "xyz": xyz, "res": res, "name": name})
        if res == LIG_RES and not name.startswith("H"):
            lig[name] = xyz
    if not lig:
        raise SystemExit(f"no {LIG_RES} heavy atoms in {path}")
    return recs, lig


def kabsch(mobile, target):
    mc, tc = mobile.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((mobile - mc).T @ (target - tc))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1, 1, d]) @ u.T
    return rot, tc - rot @ mc


def main(argv):
    out = argv[0]
    labels, legs = {}, []
    i = 1
    while i < len(argv):
        if argv[i] == "--label":
            k, _, v = argv[i + 1].partition("=")
            labels[k] = v
            i += 2
        else:
            k, _, v = argv[i].partition("=")
            legs.append((k, v))
            i += 1

    ref, lines = None, []
    for model, (tag, path) in enumerate(legs, 1):
        recs, lig = read_pdb(path)
        names = sorted(lig)
        pos = np.array([lig[n] for n in names])
        if ref is None:
            ref, ref_names = pos, names
            rot, shift, rmsd = np.eye(3), np.zeros(3), 0.0
        else:
            if names != ref_names:
                raise SystemExit(f"{tag}: ligand atom names differ from the first leg")
            rot, shift = kabsch(pos, ref)
            rmsd = float(np.sqrt(((pos @ rot.T + shift - ref) ** 2).sum(1).mean()))
        lines.append(f"MODEL     {model:>4d}")
        lines.append(f"REMARK   1 {model}. {tag}")
        if tag in labels:
            lines.append(f"REMARK   1 {labels[tag]}")
        lines.append(f"REMARK   1 ligand superposition on model 1: {rmsd:.3f} A")
        for serial, r in enumerate(recs, 1):
            x, y, z = r["xyz"] @ rot.T + shift
            lines.append(f"{r['line'][:6]}{serial:5d}{r['line'][11:30]}"
                         f"{x:8.3f}{y:8.3f}{z:8.3f}{r['line'][54:]}")
        lines.append("TER")
        lines.append("ENDMDL")
    lines.append("END")
    open(out, "w").write("\n".join(lines) + "\n")
    print(f"{out}: {len(legs)} models, superposed on {len(ref)} ligand heavy atoms")


if __name__ == "__main__":
    main(sys.argv[1:])
