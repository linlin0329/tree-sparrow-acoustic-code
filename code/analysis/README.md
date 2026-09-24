# Acoustic data tools

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
matched feature/patch arrays, the supplied taxonomy and excerpt-grouped
evaluation settings (`canonical2556`, `retry-seed`). Its
input bounds are linked to the released bounds by
`data/validation/label_assessment/assessment_input_bounds.csv`
(`released_integer_frames`); do not replace
its arrays with features recomputed from different boundaries.

Songiness uses 35 matched features and supplied reviewed-recording flags:
9,015 fitting records and 55,182 deployment records. Some features encode an
auxiliary eight-class vocabulary, separate from the final family taxonomy.
Its positives/negatives do not define the prevalence of song in the released
archive. No serialized random forest is supplied; computation fits the defined
model from the matched inputs.

## Clustering support

```bash
python -m sparrow_dataset.cluster --mode robust_hd \
  --output-dir ../../outputs/clustering --dry-run
```

The clustering aid uses RobustScaler, UMAP and HDBSCAN leaf selection, including
the specified parameter grids and seed-stability ranking. It loads 2,893 × 105
features with aligned RMS and source identities from
`assets/clustering/candidate_inputs/`. Use this feature library together with
its matching RMS vector; do not substitute the label-assessment feature library.
The `embedding_2d` comparison uses the defined RMS gate; `robust_hd` retains RMS
as QC. Cluster numbers are candidates for human review, not final family labels.
Remove `--dry-run` to write outputs to a new directory.

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
`assemble-release`, `recording-features`, `all-recording-features`, `cluster`
and `mp3-assessment`. FFmpeg/ffprobe are needed for codec checks.

Assembly uses additional accepted QC/review inputs described in
[INPUTS.md](INPUTS.md), including sources excluded from the public subset.
The public subset cannot recreate missing source recordings or human decisions.
The working QC pool uses a score gate of >0.5; the released raw selection uses
the declared >=0.7 policy and retains known-clock flags. Do not filter the
already prepared public WAVs again.

Run `python -m unittest discover -s tests -q` for regression checks. The tests
use small synthetic signals/cohorts; they do not fit models to the real corpus.

[中文说明](README.zh-CN.md)
