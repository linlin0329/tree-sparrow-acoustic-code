# Processing inputs

For ordinary reuse, supply the sibling `data/` directory with `--package`.
No separate source archive is needed for metadata checks, Raven export or
sample-exact syllable reconstruction. The installed `sparrow_dataset/assets/`
contains matched model-assessment and clustering inputs;
`assessment inspect` checks their hashes and identity joins.

## Rebuilding the data package

`assemble-release` requires accepted QC/review inputs beyond the public subset.
It accepts three roots explicitly:

- `--evidence-root`: the logical layout below;
- `--data-root`: source archive files referenced by the QC inventory, plus
  `review_audio/seed/` and `review_audio/manual/` when required to establish
  source equivalence;
- `--prepared-root`: prepared parents referenced by `processing/parents.csv`.

```text
evidence-root/
  qc/final/
    summary.json
    validation.json
    provenance.json
    final_logical_inventory.csv.gz
  annotations/metadata/
    file_manifest.json
    tables/
      manual_recordings.csv
      prepared_recordings.csv
      source_links.csv
      raven_tables.csv
      syllables.csv
      label_families.csv
      label_leaves.csv
  annotations/reference/
    manifest.csv
    annotations/raven/
    <reference syllable audio named by the manifest>
  processing/parents.csv
  review/
    release_scope.json
    song_seed.csv
    song_candidates.csv
    high.list
    medium.list
    candidate_labels.csv
    rule_candidates.csv
    fragment_reviews.csv
    researcher_review.csv
    researcher_review.json
```

These inputs describe accepted recording selection, researcher decisions and
sample identities. The assembler checks the declared file hashes, uses source
recording IDs to deduplicate the song index, joins Raven rows by their original
row identity, and compares integer-frame cuts against reference audio. It does
not infer missing researcher decisions or promote candidate alignment offsets.
The public tables use waveform/coordinate `data_version=1.0.0` and
`label_version=1.0.0`. The table schema is specified in
`data/metadata/schema.json`; it is distinct from the data-package version.

The raw inventory/QC stages also need the complete archive and companion
BirdNET CSVs, including excluded records. Their `relative_path` values are
preserved because they participate in source identities. The source archive
layout uses `birdsongs_YEAR/data/wav/raw/`; published audio paths use
`audio/raw/YEAR/`. Use `stage-help` to inspect each stage's arguments and run
`process` without `--run` to inspect a command before execution.

## Numerical inputs

The label-assessment inputs include a 2,893-row feature library, the current
2,556-row label input and 127 within-structure pair results. The assessment joins
rows by the supplied identities; `assessment_input_bounds.csv` in the data
package verifies the correspondence of all 2,556 analysis and extraction
intervals. Use these matched inputs for the reported internal assessment.

Clustering uses `assets/clustering/candidate_inputs/` together with its aligned
RMS vector and source identities. Its candidate feature library and the
label-assessment feature library serve different steps and must not be
interchanged. Clustering provides groups for human comparison; current labels
are supplied in the data package.
