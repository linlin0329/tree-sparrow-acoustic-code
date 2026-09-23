"""Approved manuscript drawing operations with explicit local dependencies."""

from __future__ import annotations
import csv
import collections

STRUCTURES = ("single", "double", "triple")
COLORS = {"single": "#3B708C", "double": "#A1763D", "triple": "#7C6D9D"}
BANDS = {"single": "#F3F7FA", "double": "#FAF7F1", "triple": "#F7F5FA"}
TINTS = {"single": "#DFEBF2", "double": "#F1E7D5", "triple": "#EAE4F2"}


def validated_data(metadata_dir):

    def read(name):
        with (metadata_dir / name).open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    families, leaves, syllables = (
        read("label_families.csv"),
        read("label_leaves.csv"),
        read("syllables.csv"),
    )
    if not (len(families) == 27 and len(leaves) == 41 and (len(syllables) == 2556)):
        raise ValueError(
            "Figure input requirement failed: len(families) == 27 and len(leaves) == 41 and (len(syllables) == 2556)"
        )
    if not len({x["syllable_id"] for x in syllables}) == 2556:
        raise ValueError(
            "Figure input requirement failed: len({x['syllable_id'] for x in syllables}) == 2556"
        )
    fs = {x["family_id"]: x for x in families}
    ls = {x["leaf_id"]: x for x in leaves}
    if not (len(fs) == 27 and len(ls) == 41):
        raise ValueError(
            "Figure input requirement failed: len(fs) == 27 and len(ls) == 41"
        )
    fc = collections.Counter((x["family_id"] for x in syllables))
    lc = collections.Counter((x["leaf_id"] for x in syllables))
    for x in syllables:
        if not (x["leaf_id"] in ls and x["family_id"] in fs):
            raise ValueError(
                "Figure input requirement failed: x['leaf_id'] in ls and x['family_id'] in fs"
            )
        if not ls[x["leaf_id"]]["family_id"] == x["family_id"]:
            raise ValueError(
                "Figure input requirement failed: ls[x['leaf_id']]['family_id'] == x['family_id']"
            )
        if (
            not x["structure"]
            == ls[x["leaf_id"]]["syllable_structure"]
            == fs[x["family_id"]]["syllable_structure"]
        ):
            raise ValueError(
                "Figure input requirement failed: x['structure'] == ls[x['leaf_id']]['syllable_structure'] == fs[x['family_id']]['syllable_structure']"
            )
    for x in families:
        if not fc[x["family_id"]] == int(x["syllable_count"]):
            raise ValueError(
                "Figure input requirement failed: fc[x['family_id']] == int(x['syllable_count'])"
            )
    for x in leaves:
        if not lc[x["leaf_id"]] == int(x["syllable_count"]):
            raise ValueError(
                "Figure input requirement failed: lc[x['leaf_id']] == int(x['syllable_count'])"
            )
    groups = []
    rows = []
    for i, structure in enumerate(STRUCTURES, 1):
        ff = sorted(
            [f for f in families if f["syllable_structure"] == structure],
            key=lambda f: f["family_id"],
        )
        gg = []
        for f in ff:
            ll = sorted(
                [l for l in leaves if l["family_id"] == f["family_id"]],
                key=lambda l: (l["is_variant"] == "true", l["leaf_id"]),
            )
            if not (
                int(f["body_element_count"]) == i and len(ll) == int(f["leaf_count"])
            ):
                raise ValueError(
                    "Figure input requirement failed: int(f['body_element_count']) == i and len(ll) == int(f['leaf_count'])"
                )
            if not (
                ll[0]["leaf_id"] == f["family_id"] and ll[0]["is_variant"] == "false"
            ):
                raise ValueError(
                    "Figure input requirement failed: ll[0]['leaf_id'] == f['family_id'] and ll[0]['is_variant'] == 'false'"
                )
            if not len(ll) in (1, 2):
                raise ValueError("Figure input requirement failed: len(ll) in (1, 2)")
            if len(ll) == 2:
                if not (
                    ll[1]["leaf_id"] == f["family_id"] + "_v1"
                    and ll[1]["is_variant"] == "true"
                ):
                    raise ValueError(
                        "Figure input requirement failed: ll[1]['leaf_id'] == f['family_id'] + '_v1' and ll[1]['is_variant'] == 'true'"
                    )
            gg.append((f, ll))
            for l in ll:
                rows.append(
                    {
                        "structure": structure,
                        "main_element_count": i,
                        "family_id": f["family_id"],
                        "family_number": f["family_id"].rsplit("_", 1)[1],
                        "family_syllables": int(f["syllable_count"]),
                        "leaf_id": l["leaf_id"],
                        "terminal_role": "v1" if l["is_variant"] == "true" else "Main",
                        "syllables": int(l["syllable_count"]),
                    }
                )
        groups.append(
            {
                "structure": structure,
                "elements": i,
                "families": gg,
                "family_count": len(ff),
                "leaf_count": sum((len(ll) for _, ll in gg)),
                "syllable_count": sum((int(f["syllable_count"]) for f in ff)),
            }
        )
    if not [
        (g["family_count"], g["leaf_count"], g["syllable_count"]) for g in groups
    ] == [(14, 20, 1153), (6, 9, 768), (7, 12, 635)]:
        raise ValueError(
            "Figure input requirement failed: [(g['family_count'], g['leaf_count'], g['syllable_count']) for g in groups] == [(14, 20, 1153), (6, 9, 768), (7, 12, 635)]"
        )
    if not (
        sum((r["terminal_role"] == "Main" for r in rows)) == 27
        and sum((r["terminal_role"] == "v1" for r in rows)) == 14
    ):
        raise ValueError(
            "Figure input requirement failed: sum((r['terminal_role'] == 'Main' for r in rows)) == 27 and sum((r['terminal_role'] == 'v1' for r in rows)) == 14"
        )
    return (groups, rows)
