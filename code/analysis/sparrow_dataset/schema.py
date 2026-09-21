"""Public metadata schema for release 1.0.0."""

VERSION = "1.0.0"
OMIT_FIELDS = {
    "song_recordings.csv": {
        "historical_confirmation_evidence",
        "latest_fragment_review_group",
        "latest_review_disposition",
        "author_correction_id",
        "author_reviewed_unit",
    },
    "syllables.csv": {"boundary_status", "supersedes_syllable_id"},
}


def public_rows(name, rows):
    omitted = OMIT_FIELDS.get(name, set())
    return [
        {
            key: VERSION if key in {"data_version", "label_version"} else value
            for key, value in row.items()
            if key not in omitted
        }
        for row in rows
    ]
