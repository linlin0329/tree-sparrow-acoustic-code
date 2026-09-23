# Acoustic data tools 1.0.1

Processing and reproducibility tools for the Eurasian tree sparrow acoustic
dataset. Python 3.10 is the tested scientific environment. The original code is
[MIT licensed](LICENSE); the bundled numerical inputs are [CC BY 4.0](ASSET_LICENSE.md).

From `code/analysis/`, install the pinned scientific stack and this component:

```bash
python -m pip install -r requirements-observed.txt
python -m pip install --no-deps .
sparrow-data --help
```

The examples assume sibling `data/` and `code/` directories. Replace `../../data`
with another data location when needed. Acquisition and figure rendering use
their own environments.

```bash
sparrow-data inspect --package ../../data
sparrow-data reconstruct --package ../../data
sparrow-data export-raven --package ../../data --output-dir ../../outputs/raven
sparrow-data assessment inspect
```

`inspect` checks seven tables, source identities, the 27-family/41-leaf label
hierarchy and all 138 Raven tables. `reconstruct` verifies the decoded samples
of 2,556 syllables using zero-based, half-open integer frames. Add
`--syllable-id ID` to select one or `--output-dir NEW_DIR` to export FLOAT WAVs.
`export-raven` checks byte equality with the distributed annotations. Original
Selection values and original Raven row identities are distinct. Unverified
raw-MP3 offsets are never used as crop instructions.

## Features and internal assessment

```bash
sparrow-data features --package ../../data --kind acoustic \
  --syllable-id ID --output-dir ../../outputs/features
sparrow-data assessment taxonomy --output-dir ../../outputs/label-assessment
sparrow-data assessment songiness --output-dir ../../outputs/songiness
```

These commands print plans; add `--run` to execute them in new directories.
Feature kinds are `acoustic`, `105d` and `logmel`. The label assessment uses the
matched R1 feature/patch arrays, current R2 taxonomy and recording-blocked
evaluation settings (`2556-taxonomy-r2`, `retry-seed`). Its
input bounds are linked to the released bounds by
`data/validation/label_assessment/assessment_input_bounds.csv`; do not replace
its arrays with features recomputed from different boundaries.

Songiness uses 35 matched features and supplied reviewed-recording flags:
9,015 fitting records and 55,182 deployment records. Some features encode an
auxiliary eight-class vocabulary, separate from the final family taxonomy.
Its positives/negatives do not define the prevalence of song in the released
archive. No serialized random forest is supplied; computation fits the defined
model from the matched inputs.

## Clustering and reviewed labels

```bash
python -m sparrow_dataset.cluster --mode robust_hd \
  --output-dir ../../outputs/clustering --dry-run
python -m sparrow_dataset.annotation.archive \
  --output-dir ../../outputs/reviewed-labels --dry-run
```

The clustering aid uses RobustScaler, UMAP and HDBSCAN leaf selection, including
the specified parameter grids and seed-stability ranking. It loads 2,893 × 105
features with aligned RMS and source identities. The `embedding_2d` comparison
uses the defined RMS gate; `robust_hd` retains RMS as QC. Cluster numbers are
candidates for review, not final family labels.

The reviewed mapping applies the supplied retain, merge, renumber, explicit
structure-reclassification and delete decisions to 2,560 reviewed syllables, yielding 2,556 retained syllables and a
cumulative per-ID audit. It does not regenerate the preceding human judgments.
Remove `--dry-run` to write new outputs. Without `--audio-root`, label processing
writes CSVs only; with matching audio it checks format and source/destination
hashes. Missing source-row identities remain missing. This stage uses historical
research boundaries. Its74 structural corrections are explicit; it does not
apply the two later R1 boundary replacements. For current R1 syllables, use
`reconstruct` or `export-raven` with the companion data package.

## Fixed MP3 assessment

```bash
python -m sparrow_dataset.assessments.mp3_assessment \
  --fixtures ../../data/validation/mp3_encoding/fixtures \
  --output-dir ../../outputs/mp3-assessment --dry-run
```

This validates eleven fixed WAV/MP3 pairs and their numerical references.
Remove `--dry-run` to reproduce spectral measurements and compare every output
field with the references. It preserves alignment, 2,400-frame edge removal,
frequency bands, peaks, centroids and 90%-energy bandwidth. It never encodes a
new MP3. The fixtures are not pre-encoding references for the released field MP3s.

## Data preparation

`stage-help NAME` explains inputs. `process NAME -- ARGUMENTS` prints a command;
`process --run NAME -- ARGUMENTS` executes it. Stages include `inventory`,
`inventory-audit`, `prepare-qc`, `audio-qc`, `summarize-qc`, `prepare-corpus`,
`assemble-release`, `recording-features`, `all-recording-features`, `cluster`,
`reviewed-labels` and `mp3-assessment`. FFmpeg/ffprobe are needed for codec checks.

Assembly uses additional accepted QC/review inputs described in
[INPUTS.md](INPUTS.md), including sources excluded from the public subset.
The public subset cannot recreate missing source recordings or human decisions.
The working QC pool uses a score gate of >0.5; the released raw selection uses
the declared >=0.7 policy and retains known-clock flags. Do not filter the
already prepared public WAVs again.

Run `python -m unittest discover -s tests -q` for regression checks. The tests
use small synthetic signals/cohorts; they do not fit models to the real corpus.

[中文说明](README.zh-CN.md)

Historical clustering uses a separate `assets/clustering/historical_run/` library matched to its unchanged RMS vector. The current R1 label-assessment library must not replace those historical clustering inputs.
