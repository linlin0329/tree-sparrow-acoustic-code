# Licences and attribution

Original project software and its usage documentation are licensed under
[MIT](LICENSE), Copyright (c) 2026 Researchers. This grant does not replace
licences on the third-party/derived materials listed below.

| Material | Applicable terms and attribution |
|---|---|
| `acquisition/retention.py`, `pipeline.py`, `manage.py`, original tests and configuration | MIT; see the component's `LICENSE` |
| `acquisition/model_bridge.py` | Derived BirdNET adapter: CC BY-NC-SA 4.0; see [component licence](acquisition/LICENSE) and retained source attribution |
| `acquisition/vendor/server.py`, `labels.txt` | Fixed BirdNET-Pi/BirdNET-Lite upstream resources; original terms and complete notices in [vendor/LICENSE](acquisition/vendor/LICENSE) |
| `analysis/sparrow_dataset/assets/` | Researcher numerical inputs: [CC BY 4.0](licenses/CC-BY-4.0.txt) |
| Figure source tables, spectrogram arrays and equipment photograph | Researcher materials: [CC BY 4.0](licenses/CC-BY-4.0.txt) |
| Sentinel-2 image crops | Contains modified Copernicus Sentinel data (2023), from the 14 May 2023 scenes listed in `figures/assets/map/imagery_manifest.json`; [Sentinel data terms](https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice) |
| OpenStreetMap river vectors and source snapshots | © OpenStreetMap contributors, ODbL-1.0; [copyright and licence](https://www.openstreetmap.org/copyright); source snapshots and processed GeoPackage are supplied in `figures/assets/map/` |
| Lato and Cormorant Garamond fonts | SIL Open Font License; full texts and source/derivation records are in `figures/assets/fonts/` |

Required third-party author names, copyright notices and cited references are
retained. Project contributors are credited collectively as Researcher(s).
The compatible model weights and external runtime libraries are not included;
their own distribution terms apply. The separate acoustic data package uses
CC BY 4.0.
