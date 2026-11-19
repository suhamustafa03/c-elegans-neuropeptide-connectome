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

## 4. Synaptic connectome caveat

Randi et al. 2023 themselves compared propagation against White et al.
1986 **and** Witvliet et al. 2021 connectomes (three adult + one L4
reconstruction, scaled/averaged). This project currently uses only the
single White 1986 "whole" reconstruction as the synaptic/gap-junction
baseline layer, for simplicity and because it was the cleanest
single-file, clearly-licensed source found. This is a scope simplification
versus the original paper's connectome-averaging approach, documented here
rather than silently done; revisit if a cleanly licensed Witvliet 2021
matrix is found and the comparison would benefit from it.
