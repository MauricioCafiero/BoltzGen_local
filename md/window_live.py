"""Slice the leading N ns out of a trajectory that is still being written.

Two reasons not to use `mdtraj.load` here. It reads the whole file, which for a 5,000-particle
system at 20 ns is on the order of 700 MB resident and grows as the run does -- on a machine
already swapping under the dynamics, that is the expensive part of a window, not MMPBSA.py.
And the file is being appended to, so the last frame on disk can be half-written; `iterload`
with an explicit frame budget reads only what is asked for and stops.

A margin of a few frames is left at the end for the same reason.

The frame budget has to come from how much has been *written*, not from how long the run
will eventually be: the file grows, so a fraction of the final length is meaningless while it
is still going. Elapsed nanoseconds are read from energy.csv, whose first column is the step.

Usage: python window_live.py <traj.dcd> <topology.pdb> <out.dcd> <want_ns> <energy.csv>
"""
import sys

import mdtraj as md
from mdtraj.formats import DCDTrajectoryFile

MARGIN = 5          # frames left untouched at the head of the file being written


def elapsed_ns(energy_csv, timestep_fs=2.0):
    """Nanoseconds written so far, from the last step in the energy log."""
    with open(energy_csv) as fh:
        last = fh.readlines()[-1]
    return int(last.split(",")[0]) * timestep_fs * 1e-6


def main(traj, top, out, want_ns, energy_csv):
    with DCDTrajectoryFile(traj) as fh:
        on_disk = len(fh)
    usable = on_disk - MARGIN
    done_ns = elapsed_ns(energy_csv)
    per_ns = on_disk / done_ns
    want = int(round(want_ns * per_ns))
    print(f"{on_disk} frames on disk over {done_ns:.2f} ns written "
          f"({per_ns:.1f} frames/ns)")
    if want > usable:
        sys.exit(f"only {usable} usable frames of {on_disk} on disk, need {want} for "
                 f"{want_ns} ns -- not there yet")

    kept = []
    for chunk in md.iterload(traj, top=top, chunk=200):
        kept.append(chunk if len(chunk) <= want - sum(len(k) for k in kept)
                    else chunk[: want - sum(len(k) for k in kept)])
        if sum(len(k) for k in kept) >= want:
            break
    traj_out = kept[0] if len(kept) == 1 else kept[0].join(kept[1:])
    traj_out.save_dcd(out)
    print(f"{want_ns} ns window: kept {len(traj_out)} frames = "
          f"{len(traj_out) / per_ns:.2f} ns -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5])
