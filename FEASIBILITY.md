# BoltzGen on this laptop: what was installed, what it cost, what it can be used for

Written 2026-09-29. Machine: Apple silicon, **8 GB unified memory**, macOS 26.5, torch 2.14, MPS.
BoltzGen 0.3.2 from `github.com/HannesStark/boltzgen`, patched clone in `src_boltzgen/`, uv venv
(Python 3.12) in `.venv/`.

## It runs

The whole pipeline — design, inverse folding, refolding, analysis, filtering — completed on MPS for
the octinoxate analogue used throughout `peptidebuilder`, with the ligand as the *only* target.

| step | time | note |
|---|---|---|
| `design` | **89.7 s / 2 designs** | 500 diffusion steps, 3 recycles, plus one 2 GB checkpoint swap |
| `inverse_folding` | 5.7 s / 2 | 12.6 MB model; essentially free |
| `folding` (Boltz-2) | 81.3 s / 2 | ~40 s per complex |
| `analysis` | 20.5 s | |
| `filtering` | 7.0 s | |

About **100 s per design end to end** for a 12–15-residue peptide against a 20-heavy-atom ligand
(~35 tokens). Peak memory left 23% of the machine free — the pipeline runs each step as a separate
process and swaps the two design checkpoints in place, so only one ~2 GB model is ever resident.

Disk: ~6 GB of checkpoints in `~/.cache/huggingface` (2 × 1.93 GB design, 2.09 GB folding,
2.06 GB affinity, 12.6 MB inverse folding) + 391 MB `mols.zip` + 1.3 GB venv.

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

Design spec: designed chain `12..21`, target = the ligand alone, SMILES
`CCCC[C@H](CC)OC(=O)/C=C/c1ccc(OC)cc1` (the C17H24O3 analogue, taken from
`runs/octinoxate/ligand.xyz`), protocol `peptide-anything`.

| | design model output | after inverse folding | refold |
|---|---|---|---|
| 1 | `GLLEAIIALLLS` (6/12 residues within 4.5 Å of the ligand, 12/20 ligand atoms contacted) | `SLAALALAAALA` | 5/12 residues, 13/20 ligand atoms, backbone RMSD to design **0.38 Å** |
| 2 | `GPAELIAAAVLLLLS` (7/15, 14/20) | `MVAGLLAAALGILLA` | 6/15 residues, 14/20 ligand atoms, RMSD **0.51 Å** |

Both are 83–87% helix, bury ~170 Å² of ligand surface, and make 0–1 hydrogen bonds — a helix laid
along a lipophilic ester, which is chemically reasonable for this ligand. `design_to_target_iptm`
is 0.39–0.44, which is low.

The number worth noticing is `filter_rmsd`: **0.38 and 0.51 A** between the designed backbone and
its own refold. Some of that is self-agreement -- the refolder shares training and weights with the
designer -- and it is *not* the measure this repository is trying to win: the README states that
shell reproduction is the wrong question, since `s3_orig_f12` is the strongest binder measured and
reproduces none of its twelve designed positions. What the repository delivers is pairwise contact
options, judged over a trajectory.

So the designs were put through `check_fold.py` itself, converting each refold into the column
layout Boltz-2 writes (BoltzGen's `_atom_site` loop carries an extra `label_entity_id`, which
`parse_cif` would read as the residue number). Six further designs were generated at `33..33` to
match this repository's length. Against the 38 folds in `runs/octinoxate/boltz/fold_check*.csv`:

| | BoltzGen, 33-mers (n=6) | this repo, 38 folds | the shuffled null |
|---|---|---|---|
| `enclosed` | 0.27-0.61 | 0.36-0.96, median 0.66 | 0.74 |
| `wrapped` | 0.30-0.85 | 0.40-1.00 | 0.75 |
| `engaged` | 6-17 / 20 | 8-20 / 20 | 15/20 |
| `contacts_under_cutoff` | 3-55 | 5-67, median 22 | 22 |
| `centroid_sep` / `Rg` | 0.84-1.53 | 0.25-1.53 | 0.96 |

**Every one of the six is less enclosed than the shuffled null control**, and all six have the
ligand at or outside the peptide surface. Two carry heavy-atom contacts under the 2.6 A that this
repository treats as a broken interface (2.01 and 2.23 A). Four of the six came back near
poly-alanine -- one is 30 alanines of 33 -- which is the inverse-folding step collapsing exactly as
ESM2's argmax collapses to leucine, and is fixable by sampling more sequences per backbone
(`--inverse_fold_num_sequences`).

The exception is worth keeping: `bg33_5`, `MVAPGANGNTVVVNHEAGKVEILDKDGKVVDVR`, is **10% helical with
12 non-local backbone hydrogen bonds** -- more tertiary structure than any fold in this repository
except `s2_orig_f12`'s 11 -- at Rg 9.3 A, 0.85 wrapped, 17 of 20 ligand atoms engaged and no clash.
One structure out of six, and the only one in either set that is neither a helix nor an unfolded
chain.

**The comparison is only fair as "what BoltzGen gives at laptop scale".** Six designs is the regime
its own README warns against; the filtering stage that does the work had nothing to filter, and the
published results come from 10,000-20,000.

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
  --protocol peptide-anything --num_designs 2 --devices 1
```

`--reuse` restarts an interrupted run without losing work. `--steps <names>` runs part of the
pipeline; step names are `design inverse_folding design_folding folding affinity analysis filtering`
(note `inverse_folding` for `--steps` and `--config`, even though the config file is
`inverse_fold.yaml`). The `peptide-anything` protocol skips the affinity model, so the 2.06 GB
affinity checkpoint is downloaded but never loaded.
