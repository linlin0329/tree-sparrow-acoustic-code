# Reproduce the six manuscript figures

The figure component contains the drawing modules, source tables, map layers,
photograph and spectrogram arrays needed for six figures. Use it with the
companion data package, which supplies the field audio. The waveform/
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
| 1 | Six village references, Sentinel-2 imagery, OpenStreetMap river layers and a Gansu provincial locator |
| 2 | Researcher photograph of the recorder and vector connection diagram |
| 3 | Recording selection, song confirmation and annotation relationships |
| 4 | Three structures, 27 families and 41 terminal categories |
| 5 | One main-form example from each of 27 families, in three rows and nine columns; 0–0.7 s |
| 6 | Single 08 main form and variant; 0–0.4 s |

Figure 3c follows 581 confirmed song recordings through manual excerpt selection
to 116 source MP3 recordings, 138 analysis excerpts and 2,556 labelled syllables.
Its four numeric cards use the same height as those in panel a.

## Map inputs

The provincial locator uses the Gansu feature from Natural Earth 1:10m Admin 1,
version 5.1.1. The source geometry is provided in
[`gansu_boundary.geojson`](assets/map/gansu_boundary.geojson), and the download
URL, version, feature identity, extraction checksums and public-domain terms
are recorded in [`gansu_source_manifest.json`](assets/map/gansu_source_manifest.json).

The labelled province area, approximately 425,800 km², is the official land-area
figure in the Gansu Yellow River Basin Water Security Plan linked in that
manifest. It is independent of the generalized boundary geometry. The locator
box shows the extent of the regional map, not a surveyed coverage polygon.
The regional and local panels share the same projection and orientation; the
north arrow is shown on the regional panel. Each main panel retains its scale bar.
The regional panel spans the full height of the two local panels, with their
outer top and bottom edges aligned. The province locator is flush with the
regional panel’s upper-left corner.

The six village-reference points retain their reported coordinate values and
undocumented source datum. Sentinel-2 crops, the OpenStreetMap layers and their
source snapshots are documented in `assets/map/imagery_manifest.json` and
`assets/map/osm_source/manifest.json`.

## Syllable examples

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

## Figure 6 terminal-sound pointer

Figure 6 uses the selected S08 main form and variant. The arrow in panel b identifies the short terminal sound; it does not define an acoustic boundary. The selected syllables, spectrogram arrays, time and frequency ranges, and relative-power scale are unchanged.

This component is distributed with code package 1.0.1 and the recommended data
distribution 1.0.1. Scientific waveform/coordinate and label identifiers remain
1.0.0.
