# Running BoltzGen on Apple silicon (MPS)

BoltzGen ships CUDA-only and assumes a CUDA device exists. The patch in `mps-fixes.patch`
changes seven files, 23 inserted lines and 6 removed, so that the pipeline runs on Apple
silicon through PyTorch's MPS backend. Nothing here is Mac-specific in effect: three of the
four changes are bugs that would bite any non-CUDA host, and one is a bug that bites any host
whose multiprocessing default is `spawn`.

Verified on macOS 26.5, 8 GB unified memory, torch 2.14, Python 3.12: `design`,
`inverse_folding`, `folding`, `analysis` and `filtering` all complete, for a designed
peptide against a small-molecule target given as SMILES.

> **This duplicates [PR #145](https://github.com/HannesStark/boltzgen/pull/145).** The changes
> were worked out here from the tracebacks, and only afterwards found to match a pull request
> @fnachon opened in January 2026 and still maintains, whose fork is at
> [github.com/fnachon/boltzgen](https://github.com/fnachon/boltzgen). It reaches the same four
> places and goes further. The notes below say where its version differs and why it is
> generally the better one; `README.md` says why this copy is kept anyway. Nothing here is
> offered as novel.

## What was changed

### 1. `pyproject.toml` — CUDA-only dependencies made installable elsewhere

`cuequivariance_ops_cu12` and `cuequivariance_ops_torch_cu12` publish `manylinux` wheels
only, so `pip install boltzgen` cannot resolve on macOS at all — resolution fails before
anything is built. They are removed here, together with `cuequivariance_torch` and
`nvidia-ml-py`.

This costs nothing on a machine without CUDA: the cuEquivariance import in
`model/layers/triangular.py` is already lazy and is only reached under `use_kernels=True`,
which requires device capability >= 8. A CUDA user who wants the kernels installs the three
packages directly.

**PR #145 does this better:** it keeps all four lines and appends
`; platform_system != 'Darwin'` to each, so a Linux install is untouched and only macOS skips
them. It does the same for the `numpy==2.0.2` and `numba==0.61.0` pins. Deleting the lines, as
here, also drops the kernels for any CUDA user who installs from this patched tree.

### 2. `src/boltzgen/cli/boltzgen.py` — two unconditional CUDA calls

```python
device_capability = torch.cuda.get_device_capability()   # raises without CUDA
devices = args.devices if args.devices is not None else torch.cuda.device_count()   # 0
```

The first raises, and the second configures the trainer with zero devices. Both are guarded:
capability falls back to `(0, 0)`, which makes `--use_kernels auto` resolve to `False`, and
the device count floors at 1.

**PR #145 is the same change.** It also reports the crash this fixes as
[issue #187](https://github.com/HannesStark/boltzgen/issues/187), `AssertionError: Torch not
compiled with CUDA enabled`, which is the same bug on a CPU-only Linux host. Note that with
either version `--devices 1` stops being strictly necessary; it is still in this repository's
run commands because the runs recorded here were made before the guard went in.

### 3. `src/boltzgen/data/mol.py` — RDKit atom properties lost to `spawn`

This one is not about the GPU. Every molecule in the pipeline carries a per-atom `name`
property, and `data/feature/featurizer.py` looks it up:

```python
atom_name_to_ref = {a.GetProp("name"): a for a in mol.GetAtoms()}
```

RDKit does not include atom properties when it pickles a molecule unless asked to. Where
DataLoader workers are started with `fork`, as on Linux, the dataset is inherited rather
than pickled and the properties survive. Where they start with `spawn` — the default on
macOS — the dataset is pickled into each worker, the names are silently dropped, and the
first batch dies with `KeyError: 'name'`.

The fix is one line at import:

```python
Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
```

Without it, every step needs `num_workers=0` to run.

**PR #145 reached the same diagnosis and fixed it elsewhere:** rather than changing RDKit's
pickle default, it stops using the pickled `self.canonicals` in
`task/predict/data_from_yaml.py` and reloads the molecules from `moldir` inside the worker.
That avoids mutating RDKit global state, which is the better instinct; the one line here
covers every module that might pickle a molecule rather than the one that was observed to.

### 4. `src/boltzgen/task/predict/data_*.py` — float64 features reaching MPS

`transfer_batch_to_device` moves feature tensors to the device as they are, and some are
float64:

```
TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't
support float64. Please use float32 instead.
```

Patched in all four predict data modules (`data_from_yaml`, `data_from_generated`,
`data_ligands`, `data_protein_binder`) to cast float64 to float32 before the move.
Inference runs in float32 or bf16 regardless, so nothing is lost.

**PR #145 gates the same cast** on `torch.backends.mps.is_available()`, so a CUDA or CPU host
keeps its float64 tensors. That is the more careful form, since on those devices the cast is
a silent change in precision rather than a fix.

## What was not changed, and is worth knowing

**`accelerator: gpu` already works.** Lightning resolves it to `MPSAccelerator` when CUDA is
absent and MPS is available, so the shipped configs need no device override.

**`bf16-mixed` runs, and it quietly produces wrong geometry.** There are around fifty
`torch.autocast("cuda", enabled=False)` blocks protecting the triangle operations, attention
softmaxes and confidence heads. Under an MPS autocast those disable *CUDA* autocast and have
no effect, so those blocks run in bf16 rather than the float32 they were written to require.

This is not theoretical. The first eight designs made here came out with visibly broken
ligands, and measuring the bond lengths against an MMFF conformer of the same SMILES gave
mean errors of 0.05 to 0.24 A, up to 14 of 20 bonds wrong by more than 0.15 A, and one C-O
bond stretched to 2.32 A against an ideal 1.44. Refolding the same backbones with
`--config folding trainer.precision=32` brought that to 0.02-0.05 A mean with no bond wrong
by more than 0.15 A -- the same range as Boltz-2's own output for this ligand. Nothing warned
about any of it: the run completed, the confidences looked ordinary, and only the picture gave
it away.

So `precision=32` for the `design` and `folding` steps is part of running this on a Mac, not a
fallback. `inverse_fold.yaml` already ships at 32. The patch here does not change the defaults,
`run.sh` passes the overrides, and the two upstream routes are
[PR #258](https://github.com/HannesStark/boltzgen/pull/258), which forces float32 on CPU and
MPS, or fixing the guards themselves with
`torch.autocast(device_type=x.device.type, enabled=False)`.

**Checkpoints are swapped, not co-resident.** Each pipeline step is its own process and the
two design checkpoints are loaded in turn at a switch point, so only one ~2 GB model is ever
in memory. This is what makes an 8 GB machine viable.

## Installing

```sh
uv venv --python 3.12 .venv          # or python -m venv
source .venv/bin/activate
uv pip install -e .                  # or pip install -e .
export PYTORCH_ENABLE_MPS_FALLBACK=1 # lets any op MPS lacks fall back to CPU
```

`numpy` is pinned at `2.0.2`, which has no wheels above CPython 3.12, so use 3.12 or relax
the pin.

## Measured cost

A designed peptide of 12 to 15 residues against a 20-heavy-atom ligand, about 35 tokens:

| step | time |
|---|---|
| `design` | 89.7 s for 2 designs (500 diffusion steps, 3 recycles, one checkpoint swap) |
| `inverse_folding` | 5.7 s for 2 |
| `folding` | 81.3 s for 2 |
| `analysis` | 20.5 s |
| `filtering` | 7.0 s |

About 100 s per design end to end. At 33 residues the design step is 42 s per design. Peak
memory left 23% of an 8 GB machine free. The checkpoints are about 6 GB in `~/.cache`.

For scale: the project's README asks for 10,000 to 60,000 intermediate designs, which is
roughly twelve days at this rate. A laptop buys a few hundred overnight, which is enough for
a baseline and not for a campaign.
