# BoltzGen on Apple silicon

BoltzGen designs binders — proteins, peptides, nanobodies — for a target that can itself be
a protein, a peptide, a nucleic acid or a small molecule. It ships CUDA-only and assumes a
CUDA device exists, so it does not install and does not run on a Mac.

Seven files need changing, 23 inserted lines and 6 removed. This repository holds those
changes as a patch, the reasoning behind each one in [MPS_FIXES.md](MPS_FIXES.md), and the
design specifications and results produced with it.

**BoltzGen's own source is not copied here.** Get it from upstream and patch it, which is
what the directions below do. The interesting content is 23 lines; a vendored copy of a
63 MB tree under active development goes stale with the next release.

Verified on macOS 26.5, 8 GB unified memory, Python 3.12, torch 2.14: `design`,
`inverse_folding`, `folding`, `analysis` and `filtering` all complete for a designed peptide
against a small-molecule target given as SMILES.

## Prior art, and where to look when something else breaks

The four fixes here were worked out from scratch, and then found to duplicate
[PR #145](https://github.com/HannesStark/boltzgen/pull/145), which @fnachon opened in January
2026 and still maintains. It covers all four — including the same diagnosis of the RDKit
pickle problem, reached independently — and more besides: `pin_memory` on MPS,
`persistent_workers`, the autocast device type ([#258](https://github.com/HannesStark/boltzgen/pull/258),
which forces float32 on CPU and MPS rather than bf16-mixed), MPS SVD
([#261](https://github.com/HannesStark/boltzgen/pull/261)) and the macOS libomp conflict
([#260](https://github.com/HannesStark/boltzgen/pull/260)). Two of its choices are better than
ours: it keeps the CUDA dependencies behind `; platform_system != 'Darwin'` environment
markers instead of deleting them, and it gates the float64 cast on
`torch.backends.mps.is_available()` instead of casting unconditionally.

We keep the patch here because it is four changes we understand and can re-derive. But
**[github.com/fnachon/boltzgen](https://github.com/fnachon/boltzgen)** — `main` at `628506be5`,
11 commits ahead of upstream and not behind — is the first place to look when something else
breaks on a Mac, and [issue #146](https://github.com/HannesStark/boltzgen/issues/146) is the
thread where Mac users compare notes. The benchmarks there are all `1g13prot.yaml`, a
~200-token protein target, so they are not comparable with the ~35-token peptide-and-ligand
timings in [FEASIBILITY.md](FEASIBILITY.md).

---

## Install

One command:

```sh
./setup.sh
```

which does exactly this, and nothing else:

```sh
# 1. Fetch BoltzGen at the commit these fixes were written against.
git clone https://github.com/HannesStark/boltzgen.git src_boltzgen
git -C src_boltzgen checkout a3149cf          # boltzgen 0.3.2

# 2. Apply the Mac fixes.
git -C src_boltzgen apply mps-fixes.patch

# 3. Install it. Python 3.12, because upstream pins numpy at 2.0.2 and that has no
#    wheels above CPython 3.12.
uv venv --python 3.12 .venv                   # or python3.12 -m venv .venv
source .venv/bin/activate
uv pip install -e src_boltzgen                # or pip install -e src_boltzgen

# 4. Every shell that runs it wants this, so that any operation MPS lacks falls back
#    to the CPU instead of raising.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 5. Check the install without loading a model. Downloads a 391 MB component
#    dictionary on first run.
boltzgen check specs/octinoxate33.yaml
```

The pin matters. The patch is against `a3149cf`, and if upstream has since touched any of
those seven files `git apply` will refuse. When that happens, drop the pin and make the four
changes by hand — [MPS_FIXES.md](MPS_FIXES.md) describes each one well enough to redo from
the prose, and regenerate the patch afterwards with
`git -C src_boltzgen diff > mps-fixes.patch`.

Storage: about 6 GB of model checkpoints land in `~/.cache/huggingface` on the first real
run (`--cache` or `$HF_HOME` moves them), plus 1.3 GB of virtualenv and 63 MB of source.

## Run

```sh
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1

boltzgen run specs/octinoxate33.yaml \
  --output workbench/<name> \
  --protocol peptide-anything \
  --num_designs 6 \
  --devices 1
```

**`--devices 1` is not optional.** Without it the CLI asks CUDA how many devices there are,
gets zero, and configures the trainer with none. Everything else is stock BoltzGen, and its
own README documents the rest.

`./run.sh <spec> <output-name> [num_designs]` wraps that with a log, `--reuse`, and a
`caffeinate` that dies with the script, since a multi-hour run otherwise stops when the
machine idles:

```sh
./run.sh specs/octinoxate33.yaml octx33 6
```

Three details that cost time to find:

- `--reuse` restarts an interrupted run without losing work.
- `--steps <names>` runs part of the pipeline. The names are `design inverse_folding
  design_folding folding affinity analysis filtering` — both `--steps` and `--config` want
  `inverse_folding`, even though the config file is `inverse_fold.yaml`.
- `--config` keys differ by step: `data.num_workers` for `design`, `data.cfg.num_workers`
  for the others. Setting those to 0 was needed before the RDKit fix in `data/mol.py`; it is
  not needed now, and the runs recorded here that pass it were made before that fix.

The `peptide-anything` protocol skips the affinity model, so the 2 GB affinity checkpoint is
downloaded and never loaded.

## The design specification

A small-molecule-only target is one `ligand` entity and one designed `protein` entity, where
the range is the length to sample:

```yaml
entities:
  - protein:
      id: A
      sequence: 33..33          # 12..21 samples a length; 33..33 fixes it
  - ligand:
      id: B
      smiles: "CCCC[C@H](CC)OC(=O)/C=C/c1ccc(OC)cc1"
```

`boltzgen check <spec>` validates it and writes a cif showing what was asked for, without
loading a model. Run it first; it costs seconds and catches the whole class of mistakes where
the wrong chain ends up designed.

```
specs/
├── octinoxate.yaml       peptide of 12-21 residues, ligand-only target
├── octinoxate33.yaml     the same at 33 residues, matching peptidebuilder's designs
└── ifold/                redesigning the sequence of an existing complex
```

## What it cost and what it produced

[FEASIBILITY.md](FEASIBILITY.md) has the measured per-step timings, the memory and disk
footprint, and a comparison of eight designs against the 38 folds in `peptidebuilder` on the
cheap geometric metrics.

`bg_designs.pdb` holds all eight designs in one multi-model PDB, superposed on the ligand and
ordered by enclosure, with the sequence and the fold-check numbers in `REMARK` lines above
each model. The ligand is flexible and each fold has its own conformer, so the superposition
is 1.5 to 2.1 Å rather than exact.

`results/` keeps the eight refolded complexes as BoltzGen wrote them — with the per-residue
confidences the PDB bundle drops — beside the `fold_check.csv` those numbers come from.
Whole pipeline output goes to `workbench/`, which is not committed.

## Reusing structures between this and peptidebuilder

Two format details stand between the two projects, and both are silent rather than loud.

**BoltzGen reads ligands in a cif by CCD code.** Boltz writes a SMILES ligand as `LIG1`,
which is not in the CCD, so `boltzgen check` on a peptidebuilder fold fails with
`KeyError: "There is no item named 'LIG1.pkl' in the archive"`. The fix is a molecule
directory containing a `LIG1.pkl` — an RDKit `Mol` pickled with atom properties.
`~/.boltz/mols` is already a compatible directory of 45,000 CCD entries to extend, and
`data.cfg.moldir` points at it.

**The atom names have to match, and a mismatch does not raise.**
`parse_ccd_residue_from_smiles` matches cif atoms to the reference molecule by name,
generating names as element plus a per-element count (`C1`…`C17`, `O1`…`O3`). Boltz's cifs
use a different scheme (`C38`). Any name that does not match is treated as an atom that is not
present and its coordinates become `(0, 0, 0)`, so the ligand pose is lost without an error.
Rename the ligand atoms in a copy of the cif, then assert the parsed coordinates are non-zero
before trusting anything downstream.

**In the other direction**, BoltzGen's `_atom_site` loop carries a `label_entity_id` column
that Boltz's does not. `peptidebuilder`'s `code/uma_binding.py:parse_cif` reads fields
positionally, so it would take the entity id as the residue number and collapse every residue
into one. Converting the column layout is enough to make `check_fold.py` and everything
downstream of it work on BoltzGen output unchanged.

## Licence

This repository is MIT, © 2026 Mauricio Cafiero — see [LICENSE](LICENSE). BoltzGen itself is
MIT, © 2025 Hannes Stärk; `mps-fixes.patch` modifies that code and carries its terms, and the
specifications, results and documentation here are this repository's own.
