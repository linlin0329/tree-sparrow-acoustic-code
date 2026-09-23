# Reproduce the six manuscript figures

The figure component of code version **1.0.1** contains the drawing modules,
source tables, map layers, photograph and spectrogram arrays needed for six figures.
Field audio is supplied in the separate data package.

## Run

Use a separate Python 3.13 environment and install `requirements.txt`:

```bash
python3.13 -m venv /path/to/figure-env
/path/to/figure-env/bin/python -m pip install -r requirements.txt
MPLCONFIGDIR=/path/to/cache /path/to/figure-env/bin/python render.py \
  --data-package /path/to/data --output /path/to/new-figures
```

The data path must contain `metadata/`, `audio/analysis_clips/` and
`annotations/raven/`. The output directory must be new and outside the inputs.
Use `--check-only` to check input associations without drawing. A complete run
produces PDF, SVG and 400 dpi PNG files.

| Figure | Content |
|---|---|
| 1 | Six village references, Sentinel-2 imagery and OpenStreetMap river layers |
| 2 | Researcher photograph of the recorder and vector connection diagram |
| 3 | Recording selection, song confirmation and annotation relationships |
| 4 | Three structures, 27 families and 41 terminal categories |
| 5 | Twelve main-form examples in three rows and four columns; 0–0.5 s |
| 6 | Single 08 main form and variant; 0–0.4 s |

Both spectrogram figures use 0–16 kHz and −45 to 0 dB relative to each panel's
peak within the displayed frequency band. The 14 supplied STFT arrays are
checked against the exact samples re-extracted from analysis WAVs (zero relative
tolerance; absolute tolerance 1e-10 dB). Input checking also verifies the
116/581/64,811 recording relationships, 138 Raven tables and 2,556 syllables.
It does not repeat recognition or file-level quality screening.

`assets_manifest.json` records all resource hashes. Raster appearance may vary
slightly between Matplotlib/font versions; signal values and figure geometry
are fixed by the supplied inputs. Tests: `python -m unittest discover -s tests -v`.

Original drawing code is MIT-licensed. Researcher data and the photograph use
CC BY 4.0; external maps and fonts retain their own terms. See
[licence and attribution](../THIRD_PARTY_NOTICES.md).

The current Figure4 contains 14/6/7 families and 20/9/12 terminal categories
for Single/Double/Triple. Figure5 retains its 12 original selected waveforms;
its last panel now reads Triple family 05 (previously04). The separate
27-example author selection does not replace the 12-panel manuscript figure
in this release. Other figure media and STFT arrays are unchanged.
