# Full-recording acquisition and retention — 1.0.0

[中文说明](README.zh-CN.md)

This component captures 30-second recordings, runs a fixed BirdNET model and
retains qualifying recordings as complete MP3 files with their detection CSVs
and analysis receipts. A single qualifying detection retains the whole
recording; it does not produce a three-second excerpt.

| Module | Function |
| --- | --- |
| `pipeline.py` | ALSA capture, capture receipts and serial model processing |
| `model_bridge.py` | Fixed source/model/label checks and upstream model-function adaptation |
| `retention.py` | Input validation, selection, verified encoding and atomic archival |
| `manage.py` | Isolated installation, integrity checks and recoverable uninstall |

## Requirements and parameters

The maintenance layer uses Python 3.9+. Capture requires Linux ALSA `arecord`;
encoding requires FFmpeg/ffprobe with `libmp3lame`. Inference also requires a
compatible NumPy, librosa, SciPy and TFLite environment. Reference versions are
NumPy 1.19.5, librosa 0.9.2, SciPy 1.10.1 and tflite-runtime 2.6.0 with Python 3.9.

Only BirdNET-Pi commit `6334367ac0257514967025487026d6fe6b351afe` and the fixed
`BirdNET_6K_GLOBAL_MODEL` are supported. Obtain its model weights separately
under the applicable terms. The matching Chinese label table is included as
`vendor/labels.txt`. Source, model and label SHA-256 values are checked by
`model_bridge.py`; other versions are rejected. `upstream_contract.json`
identifies the required upstream checkout files. The bundled source is never
started as a server: the adapter loads only eight allowed model functions.

Copy `config.example.json` to a private configuration file and fill in:

| Configuration | Meaning |
| --- | --- |
| `runtime_user`, `runtime_python` | Non-root Linux account and absolute runtime interpreter path |
| `recording_device`, `device_id`, `timezone` | ALSA input, unique recorder identifier and IANA time zone |
| `latitude`, `longitude` | Recording-site coordinates used by the model |
| `model_path`, `labels_path` | Absolute model and matching label paths |
| `pending_dir`, `archive_dir`, `quarantine_dir` | Separate input, archive and optional quarantine directories |

For an installed copy, set `labels_path` to
`/path/to/installation/vendor/labels.txt`. The installer includes and verifies
this table. The three data directories must be mutually disjoint and must not
overlap the source package, upstream checkout or installation. Use directories
writable only by the operator; symbolic links and reparse paths are rejected.

The default decision requires exact labels `Passer montanus` and `麻雀`, with
confidence `>= 0.2` (`comparison: gte`; `gt` means strictly greater). This
acquisition threshold is separate from the dataset's later release filter.
The default model settings are sensitivity 1.25, overlap 1.5 seconds and
privacy threshold 0; other privacy thresholds are unsupported.

## Install, verify and roll back

Commands are run from this directory. The installation parent must exist; the
installation directory must be absent or empty. The upstream checkout must
match the fixed commit and file contract. Without `--apply`, installation and
uninstallation only show a plan.

```sh
python -B manage.py install --upstream /path/to/BirdNET-Pi --prefix /path/to/installation --config /path/to/config.json
python -B manage.py install --upstream /path/to/BirdNET-Pi --prefix /path/to/installation --config /path/to/config.json --apply
python -B manage.py verify --prefix /path/to/installation
python -B manage.py uninstall --prefix /path/to/installation
python -B manage.py uninstall --prefix /path/to/installation --apply
```

Installation stages a fixed payload and commits it by directory rename.
Repeated identical installation is idempotent. A proposed systemd unit is
written for operator activation; existing services are never changed by these
commands. `verify` checks installed-file content and permissions. To change an
installed configuration, uninstall the unchanged installation and reinstall
using the revised configuration.

Uninstall verifies managed files, creates a recovery transaction, then detaches
each current file before checking it. User modifications or replacement files
are preserved on conflict, together with the recovery transaction. Unrelated
files and recordings remain untouched. Repeating `uninstall --apply` resumes
verified recovery after an interruption; conflicting recovery material is
retained for inspection.

## Run and retain

After installation, run the installed scripts with the installed `config.json`:

```sh
python -B pipeline.py --config /path/to/installation/config.json capture-once
python -B pipeline.py --config /path/to/installation/config.json analyze-pending
python -B pipeline.py --config /path/to/installation/config.json run
python -B retention.py --config /path/to/installation/config.json --once
```

`capture-once` writes one WAV and capture receipt; `analyze-pending` processes
recordings with capture receipts; `run` combines capture and a serial analysis
consumer. The retention-only command processes producer-confirmed completion
markers. Each capture reopens the device, so consecutive recordings may have
gaps. Coordinate microphone ownership and existing cleanup jobs before use.

The pipeline validates exactly 1,440,000 mono PCM16 frames at 48 kHz, runs the
model, validates every row of the five-column semicolon CSV, then publishes a
completion marker. Positive recordings are encoded at 48 kHz mono, 320 kb/s;
the decoded MP3 must still contain exactly 1,440,000 frames before archival.
Model padding does not extend or crop the retained waveform.

Defaults retain source WAVs and negative inputs. `source_wav_policy:
delete_after_commit` permits deleting a source WAV only after the committed
archive and matching source hashes pass verification. `negative_policy:
quarantine` creates a verified copy and retains the input. Invalid input,
encoding errors, timeouts and insufficient space preserve source recordings.
There is no automatic archive-purging policy. Disk reserves are controlled by
`min_free_bytes` and `capacity_buffer_bytes`.

Completion markers bind inference settings, target labels and model hashes.
Changing these settings rejects reuse of filtered CSVs and cached analysis;
retain the original outputs and explicitly analyze in a separate location.
Retention-policy changes can reuse a matching completed analysis.

## Tests and licence

```sh
python -B -m unittest discover -s tests -v
```

Tests use temporary recordings and include real FFmpeg encode/decode checks;
codec tests explicitly skip if FFmpeg is unavailable. The researcher has
reported successful manual Raspberry Pi operation of the acquisition logic.
The publication installer's bundled-label installation is checked by local
regression tests; that report does not certify this added installation step,
reboot behavior or a particular duration of unattended operation.

Original maintenance code uses MIT; the derived model adapter and bundled
BirdNET materials retain CC BY-NC-SA 4.0 and the complete upstream notices.
See [LICENSE](LICENSE), [LICENSE-MIT](LICENSE-MIT) and
[vendor/LICENSE](vendor/LICENSE) for their respective scopes.
