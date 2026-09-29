# BoltzGen on this laptop: what was installed, what it cost, what it can be used for

Written 2026-09-29. Machine: Apple silicon, **8 GB unified memory**, macOS 26.5, torch 2.14, MPS.
BoltzGen 0.3.2 from `github.com/HannesStark/boltzgen`, patched clone in `src_boltzgen/`, uv venv
(Python 3.12) in `.venv/`.

## It runs

The whole pipeline — design, inverse folding, refolding, analysis, filtering — completed on MPS for
the octinoxate analogue used throughout `peptidebuilder`, with the ligand as the *only* target.

| step | 33-residue run, 6 designs | 12-21-residue run, 2 designs |
|---|---|---|
| `design` | 236.8 s | 85.3 s |
| `inverse_folding` | 15.6 s | 15.7 s |
| `folding` (Boltz-2) | 305.5 s | 103.5 s |
| `analysis` | 20.1 s | 12.0 s |
| `filtering` | 7.4 s | 6.9 s |
| **per design, end to end** | **97.6 s** | **111.7 s** |

Those are at `precision=32`, which is what this has to run at (see the fourth fix below). The
same pipeline at the shipped `bf16-mixed` was about 15% faster and produced wrong geometry.
Peak memory left 23% of an 8 GB machine free. The checkpoints are about 6 GB in `~/.cache`,
plus 1.3 GB of virtualenv.

## Four things had to be fixed

1. **`pip install boltzgen` cannot work on macOS.** `cuequivariance_ops_cu12` and
   `cuequivariance_ops_torch_cu12` publish manylinux wheels only, so resolution fails before
   anything is built. Removing those two, `cuequivariance_torch` and `nvidia-ml-py` from
   `pyproject.toml` and installing from source is enough: the cuEquivariance imports are lazy and
   are only reached under `use_kernels=True`, which is off without a CUDA device.
2. **Two unconditional CUDA calls in the CLI.** `src/boltzgen/cli/boltzgen.py` calls
   `torch.cuda.get_device_capability()` (crashes) and `torch.cuda.device_count()` (returns 0, so the
   trainer is configured with zero devices). Both guarded.
3. **`KeyError: 'name'` — macOS only.** Multiprocessing defaults to *spawn* here, and RDKit's
   default pickle drops atom properties, so DataLoader workers receive molecules whose `name`
   property is gone and `featurizer.py` fails on the first item. Fixed properly with
   `Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)` at the top of
   `data/mol.py`, verified by a design run at the default `num_workers=4` (58.3 s, exit 0).
   `num_workers=0` is the workaround if that line is absent, and the runs recorded below were
   made with it before the real fix went in. Linux forks its workers, so upstream never sees this.
4. **float64 tensors reaching MPS.** `transfer_batch_to_device` moves feature tensors as-is and MPS
   has no float64. Patched the four `task/predict/data_*.py` modules to cast to float32 first.

`accelerator: gpu` in the shipped configs resolves to Lightning's MPSAccelerator by itself, so no
device override is needed. `bf16-mixed` works on MPS in torch 2.14, but note that every
`torch.autocast("cuda", enabled=False)` guard in the model — and there are ~50, protecting the
triangle ops, attention softmaxes and confidence heads — is a **no-op under MPS autocast**. Those
blocks therefore run in bf16 rather than the fp32 the authors intended. Nothing looked wrong in the
two designs produced, but `--config design trainer.precision=32` is the fallback if it ever does.

## What it produced

Design specs: a designed chain of `12..21` or `33..33` residues, target = the ligand alone,
SMILES `CCCC[C@H](CC)OC(=O)/C=C/c1ccc(OC)cc1` (the C17H24O3 analogue, taken from
`runs/octinoxate/ligand.xyz`), protocol `peptide-anything`. Eight designs, all at float32.

Each refold was put through `peptidebuilder`'s own `check_fold.py`, after converting it into
the column layout Boltz-2 writes -- BoltzGen's `_atom_site` loop carries an extra
`label_entity_id`, which `parse_cif` reads positionally and would take as the residue number.
`tools/metrics.sh` does that end to end.

| structure | len | encl | wrap | eng | sep | Rg | sep/Rg | cont | closest | hel | nonlocal | sequence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bg33_4` | 33 | 0.59 | 0.95 | 19/20 | 6.3 | 9.5 | 0.66 | 18 | 3.16 | 0.90 | 0 | `GAAAARALAVVLAAAALAAGLITAEEALAAIAA` |
| `bg33_2` | 33 | 0.58 | 0.75 | 15/20 | 5.2 | 14.5 | 0.36 | 12 | 3.40 | 0.97 | 0 | `GAAAAAAAIAAAAAAAAAAAAAAAAAAAAALAA` |
| `bg33_5` | 33 | 0.56 | 0.50 | 10/20 | 7.7 | 9.2 | 0.84 | 7 | 3.35 | 0.65 | 0 | `MLTLEELIELARQKGKGARGEPLSLEEMRRIAA` |
| **`bg33_3`** | 33 | 0.48 | **1.00** | **20/20** | 8.6 | 8.5 | 1.01 | 19 | 2.69 | **0.32** | **6** | `AIVLKNISEEEAAEIARKLGGGIEKVGDSYIVY` |
| `bg33_1` | 33 | 0.38 | 0.75 | 15/20 | 17.3 | 14.4 | 1.20 | 32 | **2.34** | 1.00 | 0 | `GAAAVAAALAAAGVAAVAAVAAAAALAALLAAA` |
| `bgA` | 13 | 0.35 | 0.75 | 15/20 | 6.8 | 6.2 | 1.10 | 17 | 3.32 | 1.00 | 0 | `AAAAAAAALAALA` |
| `bgB` | 20 | 0.34 | 0.60 | 12/20 | 14.8 | 9.2 | 1.61 | 13 | 3.38 | 1.00 | 0 | `AAAAAAVAAGAAATLAALLL` |
| `bg33_0` | 33 | 0.28 | 0.55 | 11/20 | 19.9 | 14.6 | 1.36 | 9 | 3.50 | 0.97 | 0 | `AAAAAAAAAAAAVAAAVAAAAAAAAAAAALALA` |

Against the 38 folds in `runs/octinoxate/boltz/fold_check*.csv`:

| | BoltzGen (n=8) | this repo, 38 folds | the shuffled null |
|---|---|---|---|
| `enclosed` | 0.28-0.59 | 0.36-0.96, median 0.66 | 0.74 |
| `wrapped` | 0.50-1.00 | 0.40-1.00 | 0.75 |
| `engaged` | 10-20 / 20 | 8-20 / 20 | 15/20 |
| `contacts_under_cutoff` | 7-32 | 5-67, median 22 | 22 |
| `centroid_sep` / `Rg` | 0.36-1.61 | 0.25-1.53 | 0.96 |

**Every one of the eight is less enclosed than the shuffled null control**, and the best of them
(0.59) is well short of this repository's best (0.96). Six of eight have the ligand at or outside
the peptide surface. `bg33_1` has a heavy-atom contact at 2.34 A, under the 2.6 A that this
repository treats as a broken interface.

**Five of the eight came back near poly-alanine**, one of them 29 alanines of 33. That is the
inverse-folding step collapsing exactly as ESM2's argmax collapses to leucine, and it is the first
thing to fix: `--inverse_fold_num_sequences` defaults to 1, so there was nothing to select among.

Two structures are not like that, and both come from the float32 regeneration rather than the
first bf16 attempt:

- **`bg33_3`**, `AIVLKNISEEEAAEIARKLGGGIEKVGDSYIVY` -- 32% helical with **6 non-local backbone
  hydrogen bonds**, Rg 8.5 A, wrapping **all twenty** ligand heavy atoms across 19 contacts. It is
  the only fold in the set with real tertiary structure, and the only one that engages the whole
  ligand. Its closest contact, 2.69 A, is just above the clash threshold.
- **`bg33_5`**, `MLTLEELIELARQKGKGARGEPLSLEEMRRIAA` -- 65% helical, Rg 9.2 A, and reads like a
  designed helix-loop-helix. It wraps only half the ligand, so it is interesting as a sequence
  rather than as a complex.

### How the three structured folds hold it

`tools/contact_map.py` splits the ligand into moieties from its own SMILES and reports which
residues contact which -- wrapping counts atoms without saying what is gripped, and for this
ligand a peptide can stack the methoxyphenyl ring, bury the 2-ethylhexyl tail or sit on the
ester in the middle. Ring 6 atoms, methoxy 2, vinyl 2, ester 3, tail 7:

| | ring | methoxy | vinyl | ester | tail | residues | backbone contacts |
|---|---|---|---|---|---|---|---|
| `bg33_4` | 6/6 | 2/2 | 2/2 | 2/3 | 7/7 | 8 | 29 |
| `bg33_5` | 3/6 | **0/2** | 1/2 | 2/3 | 4/7 | 10 | 14 |
| **`bg33_3`** | 6/6 | 2/2 | 2/2 | **3/3** | 7/7 | 7 | 22 |

`bg33_3` covers every moiety from seven residues, and they fall in two sequence-distant blocks:
`I2 V3 L4 K5` on the ester and the tail, `I15 K18 L19` on the ring and the methoxy. A gap of 10
to 17 residues between the two segments holding opposite ends of the ligand is the non-adjacent
pairwise engagement `pair_contacts.py` argues is the informative content, and here it is backed
by structure rather than luck -- 6 non-local backbone hydrogen bonds at Rg 8.5 A. Its 22
backbone contacts echo `orig_f12`'s signature, carbonyls onto the ester carbons rather than
charged side chains.

`bg33_4` reaches 19 of 20 atoms the other way, from a 90% helix with no non-local hydrogen
bonds and contacts spread from residue 5 to residue 30: the ligand lies along the helix. That is
the groove mode, though its `centroid_sep`/`Rg` of 0.66 argues against that reading.

`bg33_5` is the weakest of the three despite the most designed-looking sequence. The
methoxyphenyl head is entirely uncontacted, 10 of 20 atoms are engaged across 10 residues, and
three of the contacts are `K14`, `K16` and `R30` against a lipophile -- the mode that pays no
desolvation penalty in a gas-phase interaction energy and is the first to weaken in water.

Ranking these three by wrapping quality gives `bg33_3` > `bg33_4` > `bg33_5`, the reverse of
ranking them by how designed the sequences look.

**The comparison is only fair as "what BoltzGen gives at laptop scale".** Eight designs is the
regime its own README warns against; the filtering stage that does the work had nothing to filter,
and the published results come from 10,000-20,000. Note also that this repository's folds were
given contact constraints while these are unconstrained controls.

### The bf16 detour, kept as evidence

The first eight designs were made at the shipped `bf16-mixed` and their ligands came out visibly
broken. Bond lengths against an MMFF conformer of the same SMILES:

| | mean error | worst | bonds off >0.15 A |
|---|---|---|---|
| bf16-mixed, 8 folds | 0.053-0.244 A | **0.88 A** | 0-14 of 20 |
| float32, same 6 backbones refolded | 0.021-0.047 A | 0.13 A | **0 of 20** |
| float32, regenerated designs | 0.023-0.070 A | 0.18 A | 0-1 of 20 |
| Boltz-2 in this repo, 8 folds | 0.012-0.025 A | 0.07 A | 0 of 20 |

One C-O bond was stretched to 2.32 A against an ideal of 1.44, which is why a viewer drew the
molecule in pieces. Aromatic bonds went the other way, compressed to 0.93-1.08 A against 1.39.
Rings stayed planar throughout, so it is bond lengths specifically. Nothing warned: the runs
completed and the confidences looked ordinary.

The bf16 structures are kept in `results/bf16_precision_artifact/` because they are the evidence
for that, and `tools/lig_geom.py` is the check -- worth running on any predicted complex before
its geometry is trusted.

## Scale is the limit, not the hardware

BoltzGen's README asks for **10,000–60,000 intermediate designs** per target, and the paper's own
small-molecule campaigns used **140–180-residue proteins** at **10,000–20,000 designs** per target,
yielding 30–250 µM binders (rucaparib: 5 of 6 tested bound at 50–150 µM; a rhodamine derivative:
4 of 4 at 30–250 µM). There is **no peptide-against-small-molecule campaign** in the paper — the
peptide protocols were validated against protein and peptide targets.

At ~100 s per design this laptop buys:

| designs | wall time |
|---|---|
| 100 | ~3 h |
| 1,000 | ~28 h |
| 10,000 | ~12 days |

So a few hundred designs overnight is the realistic local budget: enough for a baseline or a
comparison, not for a campaign. On a rented A100 the paper's numbers imply seconds per design per
stage, so 10,000 designs is on the order of 10–20 GPU-hours — well beyond the ~$4/month Modal
allocation, but a 500–1,000-design set would fit.

## Three ways this repository could use it

Ranked by cost against what they would settle.

1. **As the strong baseline the repo currently lacks.** The null model in `NEXT_STEPS.md` is random
   sequences of matched length, and the best random peptide already beat the best design by 2.5× on
   UMA interaction energy. BoltzGen designs are a far more informative comparator: generate a few
   hundred peptides for the same ligand, fold them, and score them with the machinery that already
   exists (`check_fold.py`, contact counts, MM/GBSA on the survivors). That answers "is the
   reachability-constrained shell worth anything against a generative binder model" — and a baseline
   needs hundreds of designs, not ten thousand.
2. **Inverse folding on backbones this repo built.** The `--only_inverse_fold` mode designs a
   sequence for a *fixed* backbone with the ligand present, in seconds, which is exactly what ESM2
   cannot do (ESM2 never sees the ligand — the substitution experiment improved Boltz's dG while
   leaving or breaking the actual complex). Blocker: BoltzGen reads ligands in a `.cif` by CCD code,
   and Boltz writes SMILES ligands as `LIG1`, which is not in the CCD — `boltzgen check` fails with
   `KeyError: "There is no item named 'LIG1.pkl' in the archive"`. The fix is a moldir containing a
   `LIG1.pkl` (an RDKit mol pickled with atom properties; `~/.boltz/mols` is already a compatible
   45,000-entry directory to extend). **Trap:** `parse_ccd_residue_from_smiles` matches cif atoms to
   the mol by name, generating names as element + per-element count (`C1`…`C17`, `O1`…`O3`), whereas
   Boltz's cifs use a different scheme (`C38`). Names that do not match are silently treated as
   absent and their coordinates become (0, 0, 0) — the ligand pose would be lost without an error.
   Rename the ligand atoms in a copy of the cif and assert the parsed coordinates are non-zero.
3. **A full peptide campaign for octinoxate.** Legal to specify and it runs, but it is off-label
   twice over: peptide binders for a small molecule are not a validated BoltzGen protocol, and the
   design count that made its published results work is 30–100× what this machine can produce. Worth
   doing only on rented compute, and only after (1) says the comparison is interesting.

## Reproducing

```sh
cd ~/python_mac/boltzgen_local
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
boltzgen run specs/octinoxate.yaml --output workbench/<name> \
  --protocol peptide-anything --num_designs 2 --devices 1 \
  --config design trainer.precision=32 \
  --config folding trainer.precision=32
```

or `./run.sh specs/octinoxate33.yaml <name> 6`, which passes those and logs the run. Then
`tools/metrics.sh scratch/<name>_check workbench/<name>/intermediate_designs_inverse_folded/refold_cif/*.cif`
reproduces every number in this file.

`--reuse` restarts an interrupted run without losing work. `--steps <names>` runs part of the
pipeline; step names are `design inverse_folding design_folding folding affinity analysis filtering`
(note `inverse_folding` for `--steps` and `--config`, even though the config file is
`inverse_fold.yaml`). The `peptide-anything` protocol skips the affinity model, so the 2.06 GB
affinity checkpoint is downloaded but never loaded.
