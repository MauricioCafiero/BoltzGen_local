"""What shape is a predicted peptide: a rod, a helical hairpin, or a real sheet?

Two measurements, because neither alone separates the cases this ligand's designs fall into.

`peptidebuilder`'s README makes the point that a high beta_fraction means an extended backbone
and not a sheet, since a sheet needs paired strands. A count of non-local backbone hydrogen
bonds does not separate them either. What does is the *register*: in antiparallel pairing the
pairs (i, j) march in opposite directions, so i + j is roughly constant, and that constant is
twice the turn centre. Two registers in one fold mean two pairings, which is a three-stranded
sheet rather than a hairpin.

A helical hairpin has no non-local hydrogen bonds at all -- a helix bonds only to itself -- so
it is invisible to the first measurement. The second one fits an axis to each half of the chain
and reports how antiparallel they are and how close the arms sit.

Usage: python architecture.py <cif> [<cif> ...]
"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from contact_map import read_peptide  # noqa: E402


def axis(points):
    """Centroid and unit direction of the best-fit line, pointing along the chain."""
    c = points.mean(0)
    v = np.linalg.svd(points - c)[2][0]
    return c, (-v if (points[-1] - points[0]) @ v < 0 else v)


def registers(res, keys):
    """[(i+j, [(i, j)])] for the non-local backbone N...O pairs, grouped by register."""
    pairs = []
    for i in keys:
        for j in keys:
            if abs(i - j) <= 5 or "N" not in res[i] or "O" not in res[j]:
                continue
            if np.linalg.norm(res[i]["N"] - res[j]["O"]) < 3.3:
                pairs.append((i, j))
    groups = {}
    for i, j in pairs:
        # Pairs within 2 of each other in i+j belong to the same ladder.
        key = next((k for k in groups if abs(k - (i + j)) <= 2), i + j)
        groups.setdefault(key, []).append((i, j))
    return sorted(groups.items()), pairs


def main(paths):
    for path in paths:
        res, seq = {}, {}
        for num, aa, name, xyz in read_peptide(path):
            res.setdefault(num, {})[name] = xyz
            seq[num] = aa
        keys = sorted(res)
        ca = np.array([res[k]["CA"] for k in keys])
        letters = "".join(seq[k] for k in keys)

        cut, cos = min(
            ((c, float(axis(ca[:c])[1] @ axis(ca[c:])[1])) for c in range(6, len(keys) - 6)),
            key=lambda t: t[1])
        c1, _ = axis(ca[:cut])
        c2, _ = axis(ca[cut:])
        gap = min(np.linalg.norm(a - b) for a in ca[:cut] for b in ca[cut:])
        ends = float(np.linalg.norm(ca[-1] - ca[0]))

        print(f"\n{path.split('/')[-1]}  {letters}")
        print(f"  two arms: split after {seq[keys[cut - 1]]}{keys[cut - 1]}, axes at "
              f"{np.degrees(np.arccos(np.clip(cos, -1, 1))):.0f} deg, centroids "
              f"{np.linalg.norm(c1 - c2):.1f} A apart, closest CA-CA across the split {gap:.1f} A")
        print(f"  end to end {ends:.1f} A over {len(keys)} residues")

        groups, pairs = registers(res, keys)
        if not pairs:
            print("  no non-local backbone N...O under 3.3 A -- if the arms are antiparallel "
                  "and packed, this is a helical hairpin")
            continue
        print(f"  {len(pairs)} non-local N...O pairs in {len(groups)} register(s):")
        for total, group in groups:
            named = ", ".join(f"{seq[i]}{i}->{seq[j]}{j}" for i, j in sorted(group))
            print(f"    i+j~{total}: {named}")
            # The strands are the residues actually paired, grouped into contiguous runs --
            # not the halves either side of the register's centre. A long-range pairing
            # (N-terminus onto C-terminus) has no turn between its strands at all.
            involved = sorted({r for pair in group for r in pair})
            runs = [[involved[0]]]
            for r in involved[1:]:
                (runs[-1] if r - runs[-1][-1] <= 4 else runs.append([r]) or runs[-1]).append(r)
            strands = ", ".join(f"{r[0]}-{r[-1]}" for r in runs)
            print(f"      strands {strands}")
            if len(runs) == 2 and runs[1][0] - runs[0][-1] <= 9:
                between = list(range(runs[0][-1] + 1, runs[1][0]))
                turn = " ".join(f"{seq[r]}{r}" for r in between) or "none"
                print(f"      a hairpin, closed by a {len(between)}-residue loop: {turn}")
            elif len(runs) == 2:
                print(f"      {runs[1][0] - runs[0][-1] - 1} residues between them: two distant "
                      f"segments paired, so a third strand rather than a hairpin")
        if len(groups) > 1:
            print("    more than one register: a sheet of three or more strands, not one hairpin")


if __name__ == "__main__":
    main(sys.argv[1:])
