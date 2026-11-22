# Data sources and their license status

This is a factual record, not legal advice. Where a source states no
license, that is reported as such; it is not a claim that reuse is
therefore permitted or forbidden. Decisions below (e.g. not committing a
derived table) are this project's own conservative choice, not a legal
conclusion.

All checks were performed 2026-11-19 by fetching the live pages/APIs
listed; a source's terms can change after that date.

## 1. Chemical-synapse + gap-junction connectome

- **Source:** White et al. 1986 (as reanalyzed by Varshney et al. 2011,
  the "whole" variant, pharynx included), redistributed as
  `cect/data/aconnectome_white_1986_whole.csv` in
  [`openworm/ConnectomeToolbox`](https://github.com/openworm/ConnectomeToolbox).
- **Repository license:** MIT (`LICENSE` file, "Copyright (c) 2026
  OpenWorm"), confirmed by fetching the raw LICENSE file.
- **Caveat:** the MIT license covers the repository's code and
  redistributed files; it does not itself assert that OpenWorm holds or
  grants rights to the *underlying* connectome data from the original 1986
  and 2011 publications. No separate statement about the upstream data's
  terms was found. Treated here as: fine to use for non-commercial
  academic/portfolio analysis with citation, which is standard practice
  for this widely-reused, decades-old, publicly-deposited dataset; not
  independently verified as a formal grant of redistribution rights for
  any other use.
- **Retrieved:** `src/fetch_connectomes.py`, pinned to this exact raw-file
  URL on the `main` branch (see that script for the URL).
- **Committed to this repo?** Yes — 68 KB, clearly licensed, small enough
  per the "don't commit bulky raw datasets" rule not really applying here.

## 2. Neuropeptide connectome (short/mid/long-range models)

- **Source:** Ripoll-Sanchez et al. 2023 (Neuron), adjacency matrices from
  [`LidiaRipollSanchez/Neuropeptide-Connectome`](https://github.com/LidiaRipollSanchez/Neuropeptide-Connectome).
- **Repository license:** MIT (`LICENSE` file, "Copyright (c) 2024 Lidia
  Ripoll-Sanchez"), confirmed by fetching the raw LICENSE file. This repo
  is explicitly described by its own README as containing "scripts, input
  data, processed data files... for all analysis presented in" the paper —
  i.e. the data files themselves, not just code, are covered by this MIT
  license, issued by the data's own author.
- **Retrieved:** `src/fetch_connectomes.py`, pinned to the exact raw-file
  URLs on the `main` branch (last updated by the authors 2024-02-02 per
  the repo's commit history).
- **Committed to this repo?** Yes — 3 files, ~184 KB each, MIT-licensed by
  the data's own creator. Clean.

## 3. Measured signal-propagation data (Randi et al. 2023, Nature) — NOT committed

- **Source:** raw calcium-imaging traces and stimulation records on
  [OSF](https://osf.io/e2syt/) ("Neural signal propagation atlas of C.
  elegans"); processing code/pre-fit atlases in
  [`leiferlab/worm-functional-connectivity`](https://github.com/leiferlab/worm-functional-connectivity)
  and [`leiferlab/pumpprobe`](https://github.com/leiferlab/pumpprobe).
- **OSF project license:** checked via the OSF API
  (`node_license` and `relationships.license` both `null`). **No license
  is stated on the OSF project.** The Nature paper's own Data Availability
  Statement, which might state separate reuse terms, could not be read —
  the article page redirects to an institutional/subscription login and no
  open-access (PMC) copy could be confirmed accessible.
- **`worm-functional-connectivity` repository license:** none found — no
  `LICENSE` file in the repo listing, no license shown in the GitHub
  sidebar.
- **`pumpprobe` repository license:** **GPL-3.0** (`LICENSE` file present,
  confirmed via GitHub's sidebar and file listing).

### What this means for this project

Per your explicit instruction, because the *data's* reuse terms (OSF
project, `worm-functional-connectivity`) are unclear — separately from
`pumpprobe`'s GPL-3.0 *code* license — **no derived propagation table is
committed to this repository.** Only:

- Fetch/derivation **instructions** (`docs/fetch_propagation_data.md`) are
  committed.
- `pumpprobe` itself is never vendored or copied into this repository; it
  is cloned by you, separately, outside this repo's tracked files, as an
  external dependency, and the derivation script in this repo only
  *reads its output*, never includes its source.
- The derived table (`data/derived/propagation_pairs.csv` once you
  generate it) is `.gitignore`d. You generate it locally by following
  `docs/fetch_propagation_data.md`.

If you later confirm (e.g. by reading the Nature paper's Data Availability
Statement directly, or by contacting the authors) that the OSF data carries
an open license such as CC0 or CC-BY, this restriction can be relaxed and
the derived table committed — update this file with the citation for that
confirmation first.

### §3a. OSF Terms of Use review (checked against the E2SYT project specifically)

Before any decision to proceed, the current OSF/COS Terms of Use were
checked against the E2SYT project's actual metadata (`public: true`,
`node_license: null`). Source: `TERMS_OF_USE.md` in
[`CenterForOpenScience/cos.io`](https://github.com/CenterForOpenScience/cos.io)
(self-dated "last updated November 15, 2026") — the live `osf.io/terms_of_use`
page is JavaScript-rendered and could not be independently confirmed
identical to this mirror.

Key findings (verbatim, §8 unless noted): *"If no license is applied by
the administrator, then no license is included in the metadata and it
should be considered copyrighted by default"* — which is E2SYT's exact
situation — in tension with a separate clause in the same section
granting OSF users a broad license to use public content. §5 explicitly
permits downloading/modifying content "in a manner consistent with
copyright law." The Terms do not clearly resolve this tension for a
specific no-license public project. Assessment (not legal advice):
local, non-redistributed analysis is reasonably supported; publishing
aggregate results/figures is a different and more clearly supported
question under general copyright norms (facts/conclusions, not
protected expression); **redistributing a derived per-pair table
publicly remains the least-supported of the three and stays blocked.**

### §3b. Permission requested from the corresponding author — awaiting reply

Given the ambiguity above, a direct permission request was sent to
Andrew Leifer (leifer@princeton.edu, Leifer Lab, Princeton University —
identity verified by directly fetching `leiferlab.princeton.edu/contact.php`,
corroborated by the lab's GitHub organization matching `pumpprobe`'s
host), asking explicitly about the same three actions: local
download/analysis, publishing aggregate results, and redistributing a
derived per-pair table. **As of this update, no reply has been
received, and no data has been downloaded, accessed, or processed.**
This project will not proceed with any propagation-data step until a
reply arrives and is reviewed; no further contact attempts will be made
while waiting, and no assumption will be drawn from silence.

## 4. Active study's data source: Atanas, Kim et al. 2023 — used, license verified

The original neuropeptide-connectome question is paused (§3 above). The
project's **active, completed study** (behavior forecasting from
whole-brain activity — see `docs/BEHAVIOR_FORECAST_RESULTS.md`) uses a
different, separately-licensed dataset, documented here for the same
reason as every other source in this file.

- **Source:** Atanas, Kim et al., "Brain-wide representations of behavior
  spanning multiple timescales and states in *C. elegans*," *Cell* 186(19),
  4134–4151.e31 (2023). DOI: `10.1016/j.cell.2023.07.035`. PMCID:
  PMC10836760.
- **Data record used:** Zenodo record **8185377, version 2**
  ("CePNEM model analysis data and ANTSUN and microscopy neural network
  weights"), file `processed_h5.tar.bz2` only (747,188,437 bytes, MD5
  `f76bb089208db62eb4dd30785b525eda`, verified against the record before
  use).
- **License:** **CC BY 3.0 US**, independently confirmed two ways —
  (1) the rendered Zenodo record page states the license explicitly, and
  (2) the record's machine-readable API
  (`https://zenodo.org/api/records/8185377`) returns
  `metadata.license.id` consistent with CC BY 3.0 US. Both checks were
  done directly against the record, not inferred from the paper or from
  other Zenodo records in the same project.
- **Attribution required under CC BY:** cite the paper above. This
  project's own site and docs cite it throughout; anyone reusing this
  project's derived results/code should cite the same paper, not this
  repository, for the underlying data.
- **Reuse conditions:** CC BY permits copying, redistribution, and
  adaptation, including commercial use, with attribution. No additional
  restriction beyond attribution was found on this specific record.
- **A newer revision of the same project was checked and explicitly NOT
  used:** Zenodo record 19388374 (v4 of the same concept DOI,
  `10.5281/zenodo.8150514`), which adds NeuroPAL neuron-identity label
  files. That record's license metadata is **null/missing** — confirmed
  via both its API response (`metadata.license: null`,
  `metadata.rights: None`) and its rendered page (no License section, no
  CC badge, no SPDX identifier). Per this project's own data-licensing
  rule (never assume a license carries across versions), **nothing from
  v4 is used or committed**, including the neuron-identity label files
  that would otherwise have let this analysis test neuron-identity-aware
  features.
- **Committed to this repository?** **No raw or extracted data.** The
  747 MB archive and all extracted `.h5` recordings are kept in an
  external, isolated location outside Git, per this project's standing
  rule to keep raw/derived propagation-style data out of version control.
  Only small, aggregated summary statistics (medians/IQRs across
  recordings — `results/behavior_forecast_summary.txt`,
  `results/behavior_forecast_paired_comparisons.txt`) are committed; no
  per-frame, per-neuron, or per-recording raw table is.
- **What was and wasn't usable:** the archive contains 91 files (80 real
  recordings used here + 11 "scrambled-control" files in a separate
  subfolder, kept unused — see `BEHAVIOR_FORECAST_RESULTS.md`). Neuron
  identities are not usable from the main files (anonymous per-recording
  indices only) and the identity-mapping file on the v2 record
  (`dict_neuropal_label.jld2`, also CC BY 3.0 US) could not be decoded
  with `h5py` alone (its contents use Julia's JLD2 object-serialization
  format); installing Julia to decode it was explicitly ruled out. This is
  why all neural features in the active study are population-level, never
  tied to a specific neuron identity.

## 5. Synaptic connectome caveat

Randi et al. 2023 themselves compared propagation against White et al.
1986 **and** Witvliet et al. 2021 connectomes (three adult + one L4
reconstruction, scaled/averaged). This project currently uses only the
single White 1986 "whole" reconstruction as the synaptic/gap-junction
baseline layer, for simplicity and because it was the cleanest
single-file, clearly-licensed source found. This is a scope simplification
versus the original paper's connectome-averaging approach, documented here
rather than silently done; revisit if a cleanly licensed Witvliet 2021
matrix is found and the comparison would benefit from it.
