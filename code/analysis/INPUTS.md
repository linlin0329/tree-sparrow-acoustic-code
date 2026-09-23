# Processing inputs

For ordinary reuse, supply the sibling `data/` directory with `--package`.
No separate source archive is needed for metadata checks, Raven export or
sample-exact syllable reconstruction. The installed `sparrow_dataset/assets/`
contains matched model-assessment, clustering and reviewed-label inputs;
`assessment inspect` checks their hashes and identity joins.

The reviewed-label CSVs retain stable IDs, source recording keys, available
Raven row/Selection identities, bounds, format fields, final labels, output
identifiers and source/output audio hashes. Cumulative deletion states are
retained. `source_file` is relative to the companion data directory.

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
Its final public tables retain waveform/coordinate data_version 1.0.0 and use
label_version `sdt-taxonomy-r2-2026-09-24`; the package/schema version is 1.0.1.
Internal revision-only fields are omitted from the public tables.

The raw inventory/QC stages also need the complete archive and companion
BirdNET CSVs, including excluded records. Their `relative_path` values are
preserved because they participate in source identities. The source archive
layout uses `birdsongs_YEAR/data/wav/raw/`; published audio paths use
`audio/raw/YEAR/`. Use `stage-help` to inspect each stage's arguments and run
`process` without `--run` to inspect a command before execution.

## Versioned numerical assets

The label-assessment assets contain the full 2,893-row feature library with
exactly the two R1 rows replaced, a 2,556-row current-label archive and 127
R1 within-structure pair results. Historical clustering/refinement remains an
explicit earlier stage; these artifacts do not claim that the historical
human decisions were already using the corrected structure label. The
retained historical leaf mapping is archival and is not accepted as a
current R2 contract without the explicit correction.
