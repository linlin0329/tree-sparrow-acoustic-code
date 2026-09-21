"""核对seed、排序审阅和规则候选来源，构建公开鸣唱录音索引。"""

from __future__ import annotations
import csv
import re
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_sources(root, data_root):
    ds = root
    paths = {
        "seed": root / "review/song_seed.csv",
        "candidates": data_root
        / "tmp/song_prefilter/candidates_thr0.02/candidates.csv",
        "high": root / "review/high.list",
        "medium": root / "review/medium.list",
        "labels": root / "review/candidate_labels.csv",
        "rules": root / "review/rule_candidates.csv",
        "manual": ds / "annotations/metadata/tables/manual_recordings.csv",
        "review": ds / "review/fragment_reviews.csv",
    }
    seed_rows = read_csv(paths["seed"])
    seed = defaultdict(list)
    for row in seed_rows:
        seed[row["base_recording"]].append(row)
    candidates = read_csv(paths["candidates"])
    by_rank = {int(r["rank"]): r for r in candidates}
    if len(by_rank) != len(candidates):
        raise ValueError("候选rank重复")
    high = [int(v) for v in re.findall(r"\d+", paths["high"].read_text())]
    medium = [int(v) for v in re.findall(r"\d+", paths["medium"].read_text())]
    if len(set(high)) != len(high) or len(set(medium)) != len(medium):
        raise ValueError("人工序号名单内部重复")
    later = {}
    for rank in sorted(set(high) | set(medium)):
        row = by_rank[rank]
        key = row["source_file"]
        if key in later:
            raise ValueError("不同候选rank指向同一录音键")
        later[key] = dict(
            rank=rank, review_label="usable_song" if rank in high else "unusable_song"
        )
    labels = {
        r["source_file"]: r for r in read_csv(paths["labels"]) if r["any_song"] == "1"
    }
    if set(later) != set(labels) or any(
        later[k]["review_label"] != labels[k]["review_label"] for k in later
    ):
        raise ValueError("人工序号列表与归档标签不一致")
    if set(seed) & set(later):
        raise ValueError("seed与后续候选重复，须重新解释血统")
    manual = read_csv(paths["manual"])
    core = [r for r in manual if int(r["canonical_syllable_count"]) > 0]
    reviews = read_csv(paths["review"])
    receipt = ds / "review"
    author = []
    if (receipt / "researcher_review.csv").exists():
        paths["author_rereview"] = receipt / "researcher_review.csv"
        paths["author_rereview_receipt"] = receipt / "researcher_review.json"
        author = read_csv(paths["author_rereview"])
        receipt_json = json.loads(paths["author_rereview_receipt"].read_text())
        if (
            hashlib.sha256(paths["author_rereview"].read_bytes()).hexdigest()
            != receipt_json["corrections_csv_sha256"]
        ):
            raise ValueError("作者复听对照表与原话收据哈希不符")
        if (
            len(author) != 11
            or len({r["raw_recording_id"] for r in author}) != 11
            or Counter(r["reviewed_unit"] for r in author)
            != {"same_seed_manual_wav": 9, "original_30s_raw_mp3": 2}
        ):
            raise ValueError("作者本轮9+2复听范围不符")
    rules = {r["source_file"]: r for r in read_csv(paths["rules"])}
    counts = (len(seed_rows), len(seed), len(later), len(core))
    if counts != (122, 113, 455, 138):
        raise ValueError(f"已批准的来源快照数量变化：{counts}")
    return dict(
        paths=paths,
        seed=seed,
        later=later,
        core=core,
        rules=rules,
        manual=manual,
        reviews=reviews,
        data_root=data_root,
        author_rereview={r["raw_recording_id"]: r for r in author},
        requested_keys=set(seed) | set(later),
        core_raw_ids={r["raw_recording_id"] for r in core},
        audit_raw_ids={r["raw_recording_id"] for r in manual},
        high_medium_overlap=sorted(set(high) & set(medium)),
    )


def reconcile_reviews(
    sources, base_keys, published, archive, core_counts, syllable_counts, policy
):
    """先按raw ID连接片段意见；保留独立核心阳性，不把片段判断扩为整raw阴性。"""
    if policy not in (
        "hold_conflicting_recordings",
        "retain_historical_raw_positive_with_fragment_note",
        "author_rereview_confirmed_all_11",
    ):
        raise ValueError("须明确两项历史整段阳性与最新局部意见冲突的纳入政策")
    manual = {r["source_stem"]: r for r in sources["manual"]}
    by_id = {r["raw_recording_id"]: k for k, r in archive.items()}
    current = set(base_keys)
    audit = []
    used_author = set()
    for r in sources["reviews"]:
        m = manual[r["source_stem"]]
        key = by_id[m["raw_recording_id"]]
        group = r["review_group"]
        seed_sha = ""
        action = "not_in_historical_public_song_index"
        effective_group = group
        author = (
            sources.get("author_rereview", {}).get(m["raw_recording_id"])
            if policy == "author_rereview_confirmed_all_11"
            else None
        )
        if key not in published:
            action = "raw_outside_public_score_scope"
        elif author:
            if (
                author["source_recording_key"] != key
                or author["affected_manual_source_stem"] != r["source_stem"]
                or author["previous_fragment_review_group"] != group
            ):
                raise ValueError("作者复听更正没有精确关联原片段与raw ID")
            if author["reviewed_unit"] == "same_seed_manual_wav":
                old = sources["seed"][key]
                if (
                    len(old) != 1
                    or author["reviewed_sha256"] != m["sha256"]
                    or author["current_reviewed_unit_status"] != "low_quality_song"
                ):
                    raise ValueError("作者seed复听单位或哈希不匹配")
                seed_file = sources["data_root"] / "review_audio/seed" / old[0]["file"]
                manual_file = (
                    sources["data_root"]
                    / "review_audio/manual"
                    / f"{r['source_stem']}.wav"
                )
                seed_sha = hashlib.sha256(seed_file.read_bytes()).hexdigest()
                if (
                    seed_sha != m["sha256"]
                    or hashlib.sha256(manual_file.read_bytes()).hexdigest()
                    != m["sha256"]
                ):
                    raise ValueError("作者复听的seed与manual非同一字节文件")
                effective_group = "low_quality_song"
                action = "include_author_reconfirmed_low_quality_seed_clip"
            elif author["reviewed_unit"] == "original_30s_raw_mp3":
                if (
                    key not in sources["later"]
                    or author["reviewed_sha256"] != published[key]["sha256"]
                    or author["manual_fragment_judgement_changed"] != "false"
                ):
                    raise ValueError("作者原始整段复听被错误应用到裁剪片段")
                action = (
                    "include_author_confirmed_raw_song_fragment_judgement_unchanged"
                )
            else:
                raise ValueError("未知作者复听单位")
            current.add(key)
            used_author.add(m["raw_recording_id"])
        elif group in ("missed_song", "song_quality_or_overlap"):
            current.add(key)
            action = "retain_current_fragment_positive"
        elif key in base_keys and group in ("call", "uncertain_or_not_stated"):
            if core_counts[key]:
                action = "retain_separate_positive_core_fragment"
            elif key in sources["seed"]:
                old = sources["seed"][key]
                if len(old) != 1:
                    raise ValueError("冲突raw存在多份旧seed片段，不能直接降级")
                seed_file = sources["data_root"] / "review_audio/seed" / old[0]["file"]
                manual_file = (
                    sources["data_root"]
                    / "review_audio/manual"
                    / f"{r['source_stem']}.wav"
                )
                seed_sha = hashlib.sha256(seed_file.read_bytes()).hexdigest()
                if (
                    seed_sha != m["sha256"]
                    or hashlib.sha256(manual_file.read_bytes()).hexdigest()
                    != m["sha256"]
                ):
                    raise ValueError(
                        "旧seed与最新复核片段不是同一字节文件，须另查独立阳性依据"
                    )
                current.discard(key)
                action = "withdraw_superseded_same_seed_fragment_positive"
            elif key in sources["later"]:
                if policy == "hold_conflicting_recordings":
                    current.discard(key)
                    action = "hold_historical_raw_positive_with_fragment_conflict"
                else:
                    action = "retain_historical_raw_positive_with_fragment_conflict"
            else:
                raise ValueError("未登记的最新片段意见冲突来源")
        audit.append(
            dict(
                source_stem=r["source_stem"],
                raw_recording_id=m["raw_recording_id"],
                source_recording_key=key,
                latest_fragment_review_group=effective_group,
                prior_fragment_review_group=group,
                reason_verbatim=r["reason_verbatim"],
                reviewed_manual_duration_s=m["duration_s"],
                reviewed_manual_sha256=m["sha256"],
                seed_identical_wav_sha256=seed_sha,
                prior_fragment_evidence_reference=r["current_evidence_reference"],
                prior_fragment_evidence_sha256=r["current_evidence_sha256"],
                prior_fragment_received_on=r["received_on"],
                prior_fragment_revision_received_on=r["content_revision_received_on"],
                historical_route=(
                    "seed_fragment"
                    if key in sources["seed"]
                    else (
                        "later_full_recording_review"
                        if key in sources["later"]
                        else (
                            "core_rule_supplement"
                            if core_counts[key]
                            else "outside_historical_index"
                        )
                    )
                ),
                independent_core_clip_count=core_counts[key],
                independent_core_syllables=syllable_counts[key],
                in_historical_public_index=key in base_keys,
                action=action,
                whole_raw_verified_negative=False,
                author_correction_id=author["correction_id"] if author else "",
                author_reviewed_unit=author["reviewed_unit"] if author else "",
                author_reviewed_unit_status=(
                    author["current_reviewed_unit_status"] if author else ""
                ),
                author_received_on=author["received_on"] if author else "",
                author_evidence_type=author["evidence_type"] if author else "",
            )
        )
    if policy == "author_rereview_confirmed_all_11" and used_author != set(
        sources.get("author_rereview", {})
    ):
        raise ValueError("作者11项复听证据未全部精确应用")
    for row in audit:
        row["current_song_included"] = row["source_recording_key"] in current
    return current, audit


def assemble_index(sources, published, archive, *, review_policy=None):
    """published/ archive按source_recording_key索引；未入历史名单不解释为阴性。"""
    seed, later, core = sources["seed"], sources["later"], sources["core"]
    union = set(seed) | set(later)
    if union - archive.keys():
        raise ValueError("历史鸣唱键未能映射完整逻辑raw档案")
    by_id = {r["raw_recording_id"]: key for key, r in archive.items()}
    core_counts, syllable_counts = Counter(), Counter()
    for row in core:
        key = by_id[row["raw_recording_id"]]
        core_counts[key] += 1
        syllable_counts[key] += int(row["canonical_syllable_count"])
    supplement = set(core_counts) - union
    if supplement - sources["rules"].keys():
        raise ValueError("核心补充来源不在已存规则候选清单")
    excluded = union - published.keys()
    expected_excluded = {"liujiaxia_2024-06-25-birdnet-05_52_12"}
    if excluded != expected_excluded or len(supplement) != 14:
        raise ValueError("人工批准的历史排除/核心补充分支发生变化")
    base_keys = (union & published.keys()) | supplement
    public_keys, latest_review_audit = reconcile_reviews(
        sources,
        base_keys,
        published,
        archive,
        core_counts,
        syllable_counts,
        review_policy,
    )
    latest_by_key = {r["source_recording_key"]: r for r in latest_review_audit}
    if public_keys - published.keys():
        raise ValueError("鸣唱索引含未公开raw")
    result = []
    for key in sorted(public_keys):
        raw = published[key]
        if key in seed:
            route = "early_confirmed_seed"
            detail = ";".join(sorted({r["kind"] for r in seed[key]}))
            rank, review = "", "historical_seed_confirmation"
        elif key in later:
            route, detail = (
                "songiness_deployment_review",
                "second_reviewer_confirmed_song",
            )
            rank, review = str(later[key]["rank"]), later[key]["review_label"]
        else:
            route, detail = (
                "legacy_rule_candidate_core_supplement",
                "retained_in_final_human_syllable_core",
            )
            rank, review = sources["rules"][key]["rank"], "core_annotation_evidence"
        update = latest_by_key.get(key, {})
        current_evidence = (
            update["author_evidence_type"] + ":" + update["author_reviewed_unit_status"]
            if update.get("author_correction_id")
            else review
        )
        result.append(
            dict(
                song_recording_id="song:" + raw["raw_recording_id"],
                raw_recording_id=raw["raw_recording_id"],
                source_recording_key=key,
                archive_year=raw["archive_year"],
                site_id=raw["site_id"],
                audio_path=raw["audio_path"],
                provenance_route=route,
                route_detail=detail,
                route_rank=rank,
                confirmation_evidence=current_evidence,
                historical_confirmation_evidence=review,
                high_quality_clip_count=str(core_counts[key]),
                canonical_syllable_count=str(syllable_counts[key]),
                time_status=raw["time_status"],
                latest_fragment_review_group=latest_by_key.get(key, {}).get(
                    "latest_fragment_review_group", ""
                ),
                latest_review_disposition=update.get(
                    "action", "no_later_fragment_review_in_current22"
                ),
                author_correction_id=update.get("author_correction_id", ""),
                author_reviewed_unit=update.get("author_reviewed_unit", ""),
            )
        )
    if (
        len(base_keys) != 581
        or len(core_counts) != 116
        or sum(core_counts.values()) != 138
        or sum(syllable_counts.values()) != 2556
    ):
        raise ValueError("历史来源/116/138/2556集合关系不闭合")
    evidence = dict(
        seed_clip_rows=122,
        seed_unique_recordings=len(seed),
        later_confirmed_recordings=len(later),
        high_medium_overlap=sources["high_medium_overlap"],
        seed_later_overlap=0,
        historical_union=len(union),
        historical_available_in_raw_release=len(union & published.keys()),
        excluded_historical=[
            dict(source_recording_key=k, **archive[k]) for k in sorted(excluded)
        ],
        supplement_rule_recordings=len(supplement),
        supplementary_high_quality_clips=sum(core_counts[k] for k in supplement),
        supplementary_syllables=sum(syllable_counts[k] for k in supplement),
        public_song_recordings=len(result),
        core_raw_route_counts=dict(
            Counter(
                (
                    "seed"
                    if k in seed
                    else "later_review" if k in later else "rule_supplement"
                )
                for k in core_counts
            )
        ),
        core_clip_route_counts={
            route: sum(
                core_counts[k]
                for k in core_counts
                if (
                    "seed"
                    if k in seed
                    else "later_review" if k in later else "rule_supplement"
                )
                == route
            )
            for route in ("seed", "later_review", "rule_supplement")
        },
        public_song_route_counts=dict(Counter(r["provenance_route"] for r in result)),
        core_syllable_route_counts={
            route: sum(
                syllable_counts[k]
                for k in core_counts
                if (
                    "seed"
                    if k in seed
                    else "later_review" if k in later else "rule_supplement"
                )
                == route
            )
            for route in ("seed", "later_review", "rule_supplement")
        },
        historical_public_union_before_latest_review=len(base_keys),
        latest_review_policy=review_policy,
        withdrawn_after_latest_review=sorted(base_keys - public_keys),
        added_after_latest_review=sorted(public_keys - base_keys),
        latest_review_reconciliation=latest_review_audit,
        author_rereview_applied=sum(
            bool(r["author_correction_id"]) for r in latest_review_audit
        ),
        list_absence_is_not_a_negative_label=True,
    )
    return result, evidence
