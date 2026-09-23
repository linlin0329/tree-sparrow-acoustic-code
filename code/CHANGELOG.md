# 1.0.1 — structural-label correction

This release changes labels and their supporting assessment, preserving accepted
waveforms and boundary coordinates. The former Triple family 05, containing
74 syllables, is classified as Double family 05. Affected family identifiers
and their variants follow the same membership-based crosswalk; no family is
split, merged or reselected.

| Previous family | Current family |
| --- | --- |
| Double 05 | Double 06 |
| Triple 04 | Triple 05 |
| Triple 05 | Double 05 |
| Triple 06 | Triple 04 |
| Triple 07 | Triple 06 |
| Triple 08 | Triple 07 |

All other families, including Single 07, retain their labels. Current structure
totals are 1,153 Single, 768 Double and 635 Triple syllables; family counts are
14, 6 and 7, and terminal-category counts are 20, 9 and 12.

- `data_version=1.0.0` continues to identify unchanged waveform/coordinate data.
- `label_version=sdt-taxonomy-r2-2026-09-24` identifies the corrected hierarchy.
- The data/code package version is 1.0.1.
- Default label-assessment assets inherit the accepted R1 boundary corrections.
  All 2,893 feature-library rows are retained, with the two corrected rows;
  2,556 current syllables enter the label assessment and 127 within-structure
  family pairs supply pairwise evidence.
- The fold-feasibility policy inspects class coverage before fitting. Only when
  required training classes are missing does it try consecutive seeds, recording
  each attempt. The sole changed requested seed is 146 to 147 in the fifth
  nonnoise-family repeat. The criterion uses no prediction score.
- Historical clustering retains a separate original feature/RMS library.
  The reviewed-label stage combines its historical 43-to-41 mapping with an
  explicit `structure_reclassification` action for 74 rows. Its input leaf is
  historical Triple 02, corresponding to previous-release Triple 05. This
  stage uses historical research boundaries; current R1 extraction and Raven
  export use the companion data metadata.
- Figure 4 reflects the new counts; Figure 5 keeps its 12 waveforms and changes
  its last Triple label from Family 04 to Family 05. Figures 1–3 and 6 retain
  their signal/media inputs. Acquisition code is unchanged.

Validation covered all 2,556 rebuilt research-label identities, current release
input joins, complete portable assessment replay, all six figure exports,
83 analysis regression tests and six figure tests. The portable label-assessment
replay matched all 18 scientific CSVs from the canonical R1/R2 run exactly.
Historical packages and their checksums remain separate; this release does not
rewrite the old classification as if it had always been correct.
