# Reproduce the six manuscript figures

The figure component of code version **1.0.0** contains the drawing modules,
source tables, map layers, photograph and spectrogram arrays needed for six figures.
Use it with data package **1.0.0**, which supplies the field audio. The waveform/
coordinate identifier is 1.0.0 and the label identifier is
`1.0.0`.

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
| 5 | One main-form example from each of 27 families, in three rows and nine columns; 0–0.7 s |
| 6 | Single 08 main form and variant; 0–0.4 s |

Figure 5 follows S01–S14, D01–D06 and T01–T07 from left to right across the rows
(14 Single, 6 Double and 7 Triple examples). Each plot has a height-to-width
ratio of 1.19877049. The 6.3 pt family labels sit below a shared 0 kHz baseline
for each row. A shared greyscale legend and calibrated 0.4 s time scale bar sit
below the grid. The time scale bar does not change the 0–0.7 s display window;
spacing between examples is for layout, not natural song timing.

Both spectrogram figures use 0–16 kHz and −45 to 0 dB relative to each panel's
peak within the displayed frequency band. Figure 6 shows the two
Single 08 examples within a 0–0.4 s window. The input checker re-extracts 29
unique examples (27 for Figure 5 and 2 for Figure 6) from analysis WAVs and
compares them with the supplied STFT arrays (zero relative tolerance; absolute
tolerance 1e-10 dB). Input checking also verifies the
116/581/64,811 recording relationships, 138 Raven tables and 2,556 syllables.
It does not repeat recognition or file-level quality screening.

`assets_manifest.json` records all resource hashes. Raster appearance may vary
slightly between Matplotlib/font versions; signal values and figure geometry
are fixed by the supplied inputs. Tests: `python -m unittest discover -s tests -v`.

Original drawing code is MIT-licensed. Researcher data and the photograph use
CC BY 4.0; external maps and fonts retain their own terms. See
[licence and attribution](../THIRD_PARTY_NOTICES.md).

Figure 4 presents 14/6/7 families and 20/9/12 terminal categories for
Single/Double/Triple. Figures 5 and 6 use the supplied analysis WAVs and their
labels; the numerical source data and input checks specify each example.
