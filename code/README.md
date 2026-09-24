# Eurasian tree sparrow dataset code

Version **1.0.0** provides recording acquisition, data processing and figure
reproduction for the Eurasian tree sparrow acoustic dataset. Use it with data
package version **1.0.0**. Both packages may be unpacked together as `data/` and `code/`;
all data locations can also be set explicitly. The public tables use
`data_version=1.0.0` and `label_version=1.0.0`.

Archived releases: [data](https://doi.org/10.5281/zenodo.22878786) and [code](https://doi.org/10.5281/zenodo.22879603).
Links beginning with `../data/` refer to a local data package unpacked
beside `code/`; use the data DOI above when browsing this repository online.

| Component | Use | Entry |
|---|---|---|
| [Acquisition](acquisition/README.md) | Capture complete 30 s recordings, run the fixed BirdNET model, retain matching MP3s, install and roll back | Python 3.9+; component CLI and configuration |
| [Analysis](analysis/README.md) | Check data, extract syllables, export Raven tables, compute features and reproduce supporting analyses | Python 3.10+; `sparrow-data` |
| [Figures](figures/README.md) | Reproduce the six figures from source tables, maps and analysis WAVs | Separate Python 3.13 environment; `render.py` |

## Read and reuse the dataset

From `analysis/`, in its supported environment:

```bash
python -m pip install '.[science]'
sparrow-data inspect --package /path/to/data
sparrow-data reconstruct --package /path/to/data
sparrow-data export-raven --package /path/to/data --output-dir /path/to/new-raven
sparrow-data assessment inspect
```

The 138 analysis WAVs support exact integer-frame extraction of 2,556 syllables.
They originate from 116 recordings within the 581-record confirmed-song index
and the 64,811-record raw collection. Exact extraction coordinates in original
MP3s are not supplied; use the analysis WAVs and their metadata.

Acquisition, analysis and GIS/figure environments are separate. Each component
describes its dependencies and input/output paths. Run dry-run/plan options
before computational workflows. Matching analysis inputs are included; missing
field recordings or human judgements cannot be regenerated from code alone.

## Checks and licence

Before installing or changing this source tree, verify it with
`python -B verify_package.py`. Checksums identify every supplied file.
Component READMEs provide test commands and describe their actual scope.
The researcher reports successful operation of the recording component on a
Raspberry Pi; newly packaged installation resources also have local regression
checks. This is not a claim of compatibility with all Pi/BirdNET versions.

Original code uses [MIT](LICENSE). Researcher data assets use CC BY 4.0.
Third-party and derived code, fonts and maps retain their own terms, listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). No field audio or model weights
are bundled with the code. The [separate data package](../data/README.md)
contains the recordings, annotations and validation measurements.

中文说明：[README.zh-CN.md](README.zh-CN.md)。

## Figures and label assessment

Figure 5 contains one main-form example from each of the 27 families: 14 Single,
6 Double and 7 Triple examples, arranged in three rows and nine columns over
0–0.7 s. Figure 6 shows the Single 08 main-form/variant pair over 0–0.4 s.
Together, these figures use 29 unique example STFT arrays. The
[figure component](figures/README.md) describes their layout and input checks.

The syllables comprise Single 1,153/14 families/20 terminal categories,
Double 768/6/9 and Triple 635/7/12, giving 2,556 syllables, 27 families and
41 terminal categories. The internal label assessment uses numerical inputs
matched to all 2,556 supplied extraction intervals and their labels, with
127 within-structure family pairs. A deterministic fold-feasibility policy
tries consecutive seeds only when a training fold lacks a required class;
class coverage is checked before fitting or scoring. Full split assignments
and reasons are exported by the assessment. The policy does not select
validation results by their scores. Details and interpretation limits are in
the [analysis component](analysis/README.md) and the
[data assessment guide](../data/validation/label_assessment/README.md).
