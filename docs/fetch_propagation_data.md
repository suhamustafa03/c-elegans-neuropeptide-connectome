# Generating the propagation target data (manual, local step)

> **Current status: on hold, pending author permission.** A permission
> request was sent to the corresponding author (Andrew Leifer) asking
> whether (1) downloading and analyzing the data locally, (2) publishing
> aggregate results/figures, and (3) redistributing a derived per-pair
> table are permitted. **No data has been downloaded, accessed, or
> processed.** See `docs/DATA_LICENSES.md` §3 for the full reasoning and
> §5 for the request itself. Do not proceed with any step below until
> that reply arrives and is reviewed.

This step is **not automated by any script in this repository** and its
output is **never committed**. See `docs/DATA_LICENSES.md` for why: the
underlying OSF data and the `worm-functional-connectivity` code have no
stated license, and the cleanest processing path (`pumpprobe`) is
GPL-3.0 — a different, stricter license than the rest of this project
(MIT). `pumpprobe` is used here only as an external tool you run yourself,
never copied into this repository's source.

**I have not run these steps myself** — the OSF archives are 85 MB–1.1 GB
and this derivation depends on a third-party GPL-licensed package whose
exact API I verified only from its README and source listing, not by
executing it. Treat the commands below as a documented starting point to
verify and adjust, not a tested recipe. If something here is wrong once
you try it, that's expected — update this file once you've gotten it
working so it stays accurate.

## 1. Clone `pumpprobe` as a separate, external tool (not inside this repo)

```bash
cd ~/somewhere-outside-this-repo
git clone https://github.com/leiferlab/pumpprobe.git
cd pumpprobe
pip install -e .
```

Also install its stated dependencies `wormdatamodel` and `wormbrain`
(linked from the pumpprobe README; exact install commands weren't
confirmed — check that repo's own README).

## 2. Download the raw OSF data

```bash
# ~500 MB-1.1 GB depending on which archive(s) you need.
# Project: https://osf.io/e2syt/ ("Neural signal propagation atlas of C. elegans")
# Download via the OSF web UI, or the OSF CLI (`osfclient`), into a folder
# OUTSIDE this repository, e.g.:
mkdir -p ~/osf-data/e2syt
# osf -p e2syt clone ~/osf-data/e2syt   # if using osfclient
```

## 3. Build the Funatlas and export response matrices

Per the `pumpprobe` README (not independently verified by running it),
the `Funatlas` class exposes, after loading the exported dataset:

- `occ1[i, j]` — number of times neuron `i` responded to stimulation of
  neuron `j`
- `occ3[i, j]` — total number of stimulation trials of `j` with `i`
  observed (so `occ1 / occ3` is a per-pair response rate)
- `occ2[i, j]` — list of per-trial trace/response detail (amplitude,
  kinetics)

A script along these lines should produce a long-format table (verify
against the actual `pumpprobe`/`wormdatamodel` API before trusting it):

```python
# Run this OUTSIDE this repo, in an environment with pumpprobe installed.
# Sketch only - confirm exact class/method names against pumpprobe's
# actual source before running.
from pumpprobe.funatlas import Funatlas  # path/name not independently verified

funatlas = Funatlas.from_exported_data("~/osf-data/e2syt/exported_data_full")
import csv
with open("propagation_pairs.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["from_neuron", "to_neuron", "n_responses", "n_trials",
                "response_rate", "mean_amplitude"])
    for i, from_n in enumerate(funatlas.neuron_ids):
        for j, to_n in enumerate(funatlas.neuron_ids):
            n_trials = funatlas.occ3[i, j]
            if n_trials == 0:
                continue
            n_resp = funatlas.occ1[i, j]
            amps = [trial.get("amplitude") for trial in funatlas.occ2[i, j] if trial.get("amplitude") is not None]
            mean_amp = sum(amps) / len(amps) if amps else None
            w.writerow([from_n, to_n, n_resp, n_trials, n_resp / n_trials, mean_amp])
```

## 4. Define the binary "functionally connected" target

Randi et al. define this as `q < 0.05` from their own significance testing
(Kolmogorov-Smirnov + Fisher's method + Storey-Tibshirani q-values), which
is **not** simply `response_rate` thresholded — it accounts for
trial-to-trial variability properly. If `pumpprobe`/`wormdatamodel`
exposes the fitted q-values directly, prefer pulling those rather than
re-deriving a cruder threshold from `occ1/occ3` alone. Check for a
`funatlas.q` or similarly named attribute before falling back to a rate
threshold.

## 5. Place the output where this repo expects it

```bash
mkdir -p data/derived
cp propagation_pairs.csv data/derived/propagation_pairs.csv
```

`data/derived/` is `.gitignore`d — this file stays local to your machine
unless you later confirm the source data's license permits redistribution
(see `docs/DATA_LICENSES.md`) and deliberately remove it from
`.gitignore`.

## 6. Once you have it, run

```bash
python src/build_pair_dataset.py
```

This script now exists (`src/build_pair_dataset.py`) and joins
`data/derived/propagation_pairs.csv` against the connectome layers in
`data/raw/`, using `src/neuron_id.py` for alignment, to produce the final
modeling dataset. **It has only been tested against synthetic fixture
rows matching the documented schema above** (see
`tests/test_build_pair_dataset.py`) — it has never been run against real
propagation data, because that data has not been obtained. It also
deliberately refuses to run if the input table has no `q` column (Randi
et al.'s significance value): it will not silently substitute a cruder
threshold derived from `response_rate` alone, since that would
misrepresent the paper's actual "functionally connected" definition.
