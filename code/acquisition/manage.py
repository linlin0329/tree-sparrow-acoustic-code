#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Stage an isolated retention installation. Never modifies BirdNET-Pi or services.

The command defaults to a read-only plan. --apply creates a separate prefix;
starting the proposed service is deliberately outside this tool's scope.
Python 3.9+, standard library only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone

import pipeline
import retention

PINNED_COMMIT = "6334367ac0257514967025487026d6fe6b351afe"
RECEIPT = ".retention-install.json"
RECEIPT_HASH = ".retention-install.sha256"
SCHEMA = 1


class InstallError(RuntimeError):
    """A safe precondition was not met; no overwrite is permitted."""


@dataclass(frozen=True)
class InstallerContract:
    commit: str
    files: dict


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def checked_relative(value):
    if not isinstance(value, str) or "\\" in value:
        raise InstallError("Manifest paths must be relative POSIX paths")
    part = PurePosixPath(value)
    if (
        part.is_absolute()
        or not part.parts
        or any(p in (".", "..") for p in value.split("/"))
        or ":" in value
    ):
        raise InstallError("Unsafe relative path: " + str(value))
    return part


def checked_path(value):
    """Reject symbolic links and Windows junctions before resolving a boundary."""
    path = Path(os.path.abspath(os.fspath(value)))
    if any(ord(character) < 32 for character in str(path)):
        raise InstallError("Control characters are not allowed in installation paths")
    for ancestor in (path,) + tuple(path.parents):
        try:
            info = ancestor.lstat()
        except FileNotFoundError:
            continue
        reparse = getattr(info, "st_file_attributes", 0) & getattr(
            stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400
        )
        if stat.S_ISLNK(info.st_mode) or reparse:
            raise InstallError(
                "Symlink/junction paths are not allowed: " + str(ancestor)
            )
    return path


def inside(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def load_contract(package):
    manifest = json.loads(
        (package / "upstream_contract.json").read_text(encoding="utf-8")
    )
    if manifest.get("upstream_commit") != PINNED_COMMIT:
        raise InstallError(
            "Production provenance does not name the pinned upstream commit"
        )
    raw = manifest.get("files")
    if isinstance(raw, list):
        files = {item["path"]: item["sha256"] for item in raw}
        if len(files) != len(raw):
            raise InstallError("Duplicate upstream manifest paths")
    elif isinstance(raw, dict):
        files = raw
    else:
        raise InstallError("Missing upstream file hashes")
    if not files:
        raise InstallError("Empty upstream file contract")
    for name, sha in files.items():
        checked_relative(name)
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise InstallError("Invalid SHA-256 for " + name)
    return InstallerContract(PINNED_COMMIT, files)


def git(upstream, *args):
    environment = dict(os.environ, GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0")
    result = subprocess.run(
        ["git", "-C", str(upstream), *args],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    if result.returncode:
        raise InstallError("Git check failed: " + result.stderr.strip())
    return result.stdout.strip()


def check_upstream(upstream, contract):
    upstream = checked_path(upstream)
    if not upstream.is_dir():
        raise InstallError("Upstream must be an existing clean Git checkout")
    actual = git(upstream, "rev-parse", "HEAD")
    if actual != contract.commit:
        raise InstallError("Wrong upstream commit: " + actual)
    root = checked_path(git(upstream, "rev-parse", "--show-toplevel"))
    if root != upstream:
        raise InstallError("--upstream must name the checkout root")
    if git(
        upstream,
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
        "--",
        *contract.files,
    ):
        raise InstallError("Required upstream files contain tracked changes")
    for relative, expected in contract.files.items():
        checked_relative(relative)
        candidate = checked_path(upstream / relative)
        if not candidate.is_file() or digest(candidate.read_bytes()) != expected:
            raise InstallError("Upstream file SHA-256 mismatch: " + relative)
    return upstream


def read_config(config_path):
    config_path = checked_path(config_path)
    data = config_path.read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):
        raise InstallError(
            "Configuration must be UTF-8 without a BOM, matching the runtime reader"
        )
    config = json.loads(data.decode("utf-8"))
    if not isinstance(config, dict):
        raise InstallError("Configuration must be a JSON object")
    user = config.get("runtime_user")
    if not isinstance(user, str) or not re.fullmatch(r"[a-z_][a-z0-9_-]*[$]?", user):
        raise InstallError(
            "Set runtime_user to the non-root Linux account that will run capture"
        )
    if user == "root":
        raise InstallError("runtime_user must not be root")
    runtime_python = config.get("runtime_python")
    if (
        not isinstance(runtime_python, str)
        or not runtime_python
        or any(ord(c) < 32 for c in runtime_python)
        or not (
            PurePosixPath(runtime_python).is_absolute()
            or PureWindowsPath(runtime_python).is_absolute()
        )
    ):
        raise InstallError(
            "Set runtime_python to the absolute Python executable in the validated runtime environment"
        )
    return data, config


def validate_configuration(package, config):
    """Use the runtime schema without importing code from arbitrary packages."""
    return pipeline.validate_config(config, require_runtime=False)


def systemd_quote(value):
    # Specifiers are expanded even inside quoted systemd arguments.
    return (
        '"'
        + str(value)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("%", "%%")
        .replace("$", "$$")
        + '"'
    )


def proposed_unit(prefix, config):
    return (
        "# PROPOSED ONLY: review old capture/analysis/extraction/cleanup/disk jobs before activation.\n"
        "# Installing this package never enables, starts, stops or reloads a service.\n"
        "[Unit]\nDescription=BirdNET-Pi retained recording pipeline (manual activation)\nAfter=sound.target\n\n"
        "[Service]\nType=simple\nUser=" + config["runtime_user"] + "\n"
        "WorkingDirectory=" + systemd_quote(prefix) + "\n"
        "ExecStart="
        + systemd_quote(config["runtime_python"])
        + " "
        + systemd_quote(prefix / "pipeline.py")
        + " --config "
        + systemd_quote(prefix / "config.json")
        + " run\n"
        "Restart=on-failure\nRestartSec=10\nUMask=0077\n\n"
        "[Install]\nWantedBy=multi-user.target\n"
    ).encode("utf-8")


def collect_payload(package, prefix, config_data, config):
    """Install an explicit runtime set; never recursively copy history or caches."""
    required = (
        "pipeline.py",
        "model_bridge.py",
        "retention.py",
        "manage.py",
        "vendor/server.py",
        "vendor/labels.txt",
        "vendor/LICENSE",
        "LICENSE",
        "LICENSE-MIT",
        "upstream_contract.json",
    )
    payload = {}
    for relative in required:
        candidate = checked_path(package / relative)
        if not candidate.is_file():
            raise InstallError(
                "Required runtime/provenance file is missing: " + relative
            )
        payload[relative] = (candidate.read_bytes(), 0o644)
    if digest(payload["vendor/labels.txt"][0]) != pipeline.LABELS_SHA256:
        raise InstallError(
            "Packaged labels.txt does not match the supported Chinese label SHA-256"
        )
    payload["config.json"] = (config_data, 0o600)
    payload["systemd/birdnet-retention.service.proposed"] = (
        proposed_unit(prefix, config),
        0o644,
    )
    return payload


def read_receipt(prefix):
    path = checked_path(prefix / RECEIPT)
    if not path.is_file():
        raise InstallError(
            "Nonempty prefix has no installation receipt; refusing to overwrite"
        )
    receipt_data = path.read_bytes()
    hash_path = checked_path(prefix / RECEIPT_HASH)
    if not hash_path.is_file() or hash_path.read_text(
        encoding="ascii"
    ).strip() != digest(receipt_data):
        raise InstallError("Receipt integrity check failed; no files will be changed")
    if os.name != "nt" and any(
        stat.S_IMODE(p.stat().st_mode) != 0o600 for p in (path, hash_path)
    ):
        raise InstallError("Receipt permissions changed; no files will be changed")
    receipt = json.loads(receipt_data.decode("utf-8"))
    if not isinstance(receipt, dict):
        raise InstallError("Receipt must be a JSON object")
    if receipt.get("schema") != SCHEMA or receipt.get("prefix") != str(prefix):
        raise InstallError("Receipt schema or prefix does not match")
    entries = receipt.get("files")
    if not isinstance(entries, list) or not entries:
        raise InstallError("Receipt has no installed file list")
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise InstallError("Invalid receipt file entry")
        relative = entry.get("path")
        checked_relative(relative)
        if relative in (RECEIPT, RECEIPT_HASH) or relative in seen:
            raise InstallError("Invalid receipt file list")
        seen.add(relative)
        if not re.fullmatch(
            r"[0-9a-f]{64}", str(entry.get("sha256"))
        ) or not isinstance(entry.get("mode"), int):
            raise InstallError("Invalid receipt digest or mode")
    dirs = receipt.get("directories", [])
    if not isinstance(dirs, list):
        raise InstallError("Invalid receipt directory list")
    for directory in dirs:
        checked_relative(directory)
    return receipt


def verify_installation(prefix):
    prefix = checked_path(prefix)
    receipt = read_receipt(prefix)
    problems = []
    for entry in receipt["files"]:
        path = checked_path(prefix / entry["path"])
        if not path.is_file():
            problems.append(entry["path"] + ": missing/not a file")
        elif digest(path.read_bytes()) != entry["sha256"]:
            problems.append(entry["path"] + ": content changed")
        elif os.name != "nt" and stat.S_IMODE(path.stat().st_mode) != entry["mode"]:
            problems.append(entry["path"] + ": permissions changed")
    if problems:
        raise InstallError(
            "Refusing mutation; installed files changed:\n" + "\n".join(problems)
        )
    return receipt


def write_file(path, data, mode):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, mode)
    os.replace(temporary, path)


def install(package, upstream, prefix, config_path, contract, apply=False):
    """Plan without writes; serialize mutation with a stable sibling lock."""
    prefix = checked_path(prefix)
    if not apply:
        return _install(package, upstream, prefix, config_path, contract, False)
    if not prefix.parent.is_dir():
        raise InstallError("Prefix parent must already exist")
    with retention._lock(prefix.parent, 30, "." + prefix.name + ".management.lock"):
        return _install(package, upstream, prefix, config_path, contract, True)


def _install(package, upstream, prefix, config_path, contract, apply=False):
    package = checked_path(package)
    upstream = check_upstream(upstream, contract)
    prefix = checked_path(prefix)
    if _transaction_path(prefix).exists():
        raise InstallError(
            "Interrupted uninstall: run uninstall --apply to recover before installation"
        )
    if any(
        inside(prefix, root) or inside(root, prefix) for root in (upstream, package)
    ):
        raise InstallError(
            "Prefix must be separate from the upstream checkout and source package"
        )
    if not prefix.parent.is_dir():
        raise InstallError("Prefix parent must already exist: " + str(prefix.parent))
    if prefix.exists() and not prefix.is_dir():
        raise InstallError("Prefix is not a directory")
    config_data, config = read_config(config_path)
    config = validate_configuration(package, config)
    for name in ("pending_dir", "archive_dir", "quarantine_dir"):
        if config.get(name):
            data_root = checked_path(config[name])
            for protected in (prefix, package, upstream):
                if inside(data_root, protected) or inside(protected, data_root):
                    raise InstallError(
                        name
                        + " must be separate from installation and source directories"
                    )
    payload = collect_payload(package, prefix, config_data, config)
    for relative in ("scripts/server.py", "LICENSE"):
        expected = contract.files.get(relative)
        embedded = (
            "vendor/server.py" if relative == "scripts/server.py" else "vendor/LICENSE"
        )
        if expected is not None and digest(payload[embedded][0]) != expected:
            raise InstallError(
                "Packaged historical file does not match upstream contract: " + embedded
            )
    if prefix.exists() and any(prefix.iterdir()):
        receipt = verify_installation(prefix)
        current = {
            entry["path"]: entry
            for entry in receipt["files"]
            if entry["path"] != "backup/preinstall.json"
        }
        same = set(current) == set(payload) and all(
            current[name]["sha256"] == digest(data) and current[name]["mode"] == mode
            for name, (data, mode) in payload.items()
        )
        if not same:
            raise InstallError(
                "Prefix contains a different installation; verify and uninstall it first"
            )
        return {
            "status": "already_installed",
            "prefix": str(prefix),
            "service_changed": False,
        }
    existed = prefix.exists()
    previous_mode = stat.S_IMODE(prefix.stat().st_mode) if existed else None
    backup = {
        "prefix_existed": existed,
        "prefix_mode": previous_mode,
        "previous_files": [],
        "note": "No existing files are overwritten; prefix was absent or empty.",
    }
    payload["backup/preinstall.json"] = (json_bytes(backup), 0o600)
    directories = set()
    for relative in payload:
        checked_relative(relative)
        for parent in PurePosixPath(relative).parents:
            if str(parent) != ".":
                directories.add(str(parent))
    receipt = {
        "schema": SCHEMA,
        "software_version": retention.VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prefix": str(prefix),
        "upstream": str(upstream),
        "upstream_commit": contract.commit,
        "preinstall": backup,
        "files": [
            {"path": name, "sha256": digest(data), "mode": mode}
            for name, (data, mode) in sorted(payload.items())
        ],
        "directories": sorted(directories),
    }
    report = {
        "status": "planned",
        "prefix": str(prefix),
        "files": len(payload),
        "service_changed": False,
        "runtime_validation": "deferred: no microphone, model inference, executable dependencies or service activation checked",
        "preinstall": backup,
    }
    if not apply:
        return report
    stage = Path(
        tempfile.mkdtemp(prefix="." + prefix.name + ".staging-", dir=str(prefix.parent))
    )
    try:
        for relative, (data, mode) in payload.items():
            write_file(stage / relative, data, mode)
        receipt_data = json_bytes(receipt)
        write_file(stage / RECEIPT, receipt_data, 0o600)
        write_file(
            stage / RECEIPT_HASH, (digest(receipt_data) + "\n").encode("ascii"), 0o600
        )
        for directory in sorted(stage.rglob("*"), reverse=True):
            if directory.is_dir():
                os.chmod(directory, 0o755)
                retention._fsync_dir(directory)
        os.chmod(stage, previous_mode if existed else 0o755)
        checked_path(prefix)
        if existed:
            # rmdir refuses if another process put anything in the empty prefix.
            prefix.rmdir()
        elif prefix.exists():
            raise InstallError("Prefix appeared while staging; no overwrite attempted")
        try:
            retention._fsync_dir(stage)
            os.rename(stage, prefix)
            retention._fsync_dir(prefix.parent)
        except BaseException:
            if existed and not prefix.exists():
                prefix.mkdir(mode=previous_mode)
            raise
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    report["status"] = "staged"
    return report


def _transaction_path(prefix):
    return checked_path(prefix.parent / ("." + prefix.name + ".uninstall"))


def _remove_transaction(transaction, entries):
    """Never remove unexpected additions from the recovery directory."""
    expected_files = {entry["path"] for entry in entries} | {"journal.json"}
    expected_files.update("removed/" + entry["path"] for entry in entries)
    expected_dirs = set()
    for name in expected_files:
        expected_dirs.update(
            str(parent) for parent in PurePosixPath(name).parents if str(parent) != "."
        )
    for path in transaction.rglob("*"):
        checked_path(path)
        relative = path.relative_to(transaction).as_posix()
        if relative not in (expected_dirs if path.is_dir() else expected_files):
            raise InstallError(
                "Unexpected recovery content; inspect " + str(transaction)
            )
    shutil.rmtree(transaction)
    retention._fsync_dir(transaction.parent)


def _prepare_uninstall(prefix, receipt):
    transaction = _transaction_path(prefix)
    transaction.mkdir(mode=0o700)
    entries = list(receipt["files"]) + [
        {"path": name, "sha256": digest((prefix / name).read_bytes()), "mode": 0o600}
        for name in (RECEIPT, RECEIPT_HASH)
    ]
    for entry in entries:
        write_file(
            transaction / entry["path"],
            (prefix / entry["path"]).read_bytes(),
            entry["mode"],
        )
    journal = {
        "schema": 1,
        "prefix": str(prefix),
        "prefix_mode": stat.S_IMODE(prefix.stat().st_mode),
        "files": entries,
    }
    write_file(transaction / "journal.json", json_bytes(journal), 0o600)
    for directory in sorted(transaction.rglob("*"), reverse=True):
        if directory.is_dir():
            retention._fsync_dir(directory)
    retention._fsync_dir(transaction)
    retention._fsync_dir(transaction.parent)
    return transaction, entries


def _move_owned_file(prefix, transaction, entry):
    """Detach the current object atomically, then verify the detached object.

    A concurrent replacement at the original pathname is never unlinked. Any
    unexpected detached bytes remain available to recovery rather than being
    discarded on the strength of an earlier pathname check.
    """
    current = checked_path(prefix / entry["path"])
    moved = checked_path(transaction / "removed" / entry["path"])
    moved.parent.mkdir(parents=True, exist_ok=True)
    if moved.exists():
        raise InstallError("Recovery destination already exists; refusing overwrite")
    os.rename(current, moved)
    retention._fsync_dir(current.parent)
    retention._fsync_dir(moved.parent)
    moved = checked_path(moved)
    if (
        not moved.is_file()
        or digest(moved.read_bytes()) != entry["sha256"]
        or (os.name != "nt" and stat.S_IMODE(moved.stat().st_mode) != entry["mode"])
    ):
        raise InstallError(
            "Installed file changed during removal; retained in " + str(moved)
        )


def _restore_moved_files(prefix, transaction, entries):
    """Restore actual detached objects without overwriting replacement paths."""
    conflicts = []
    for entry in entries:
        moved = transaction / "removed" / entry["path"]
        if not moved.exists() and not moved.is_symlink():
            continue
        current = prefix / entry["path"]
        checked_path(current.parent)
        current.parent.mkdir(parents=True, exist_ok=True)
        try:
            # link is atomic and refuses an existing destination. Keeping the
            # detached link until transaction cleanup also preserves both
            # versions if a writer has replaced the original pathname.
            os.link(moved, current, follow_symlinks=False)
        except FileExistsError:
            if not os.path.samestat(current.lstat(), moved.lstat()):
                conflicts.append(entry["path"])
        retention._fsync_dir(current.parent)
    if conflicts:
        raise InstallError(
            "Recovery found replacement files; both versions retained in "
            + str(transaction)
            + ": "
            + ", ".join(conflicts)
        )


def _restore_uninstall(prefix):
    """Recover interrupted removal only after verifying backups and all survivors."""
    transaction = _transaction_path(prefix)
    if not transaction.exists():
        return
    journal_path = checked_path(transaction / "journal.json")
    if not journal_path.is_file():
        raise InstallError(
            "Incomplete backup requires inspection before retry: " + str(transaction)
        )
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    if (
        not isinstance(journal, dict)
        or journal.get("schema") != 1
        or journal.get("prefix") != str(prefix)
    ):
        raise InstallError("Recovery journal does not match installation")
    entries = journal.get("files")
    if not isinstance(entries, list) or not entries:
        raise InstallError("Invalid recovery file list")
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise InstallError("Invalid recovery entry")
        relative = str(checked_relative(entry.get("path")))
        if relative == "journal.json" or relative in seen:
            raise InstallError("Invalid duplicate/reserved recovery path")
        seen.add(relative)
        if type(entry.get("mode")) is not int or not 0 <= entry["mode"] <= 0o777:
            raise InstallError("Invalid recovery permissions")
        backup = checked_path(transaction / relative)
        if not backup.is_file() or digest(backup.read_bytes()) != entry.get("sha256"):
            raise InstallError(
                "Recovery backup changed; original files will not be overwritten"
            )
    prefix.mkdir(mode=journal["prefix_mode"], exist_ok=True)
    _restore_moved_files(prefix, transaction, entries)
    for entry in entries:
        current = checked_path(prefix / entry["path"])
        if current.exists() and (
            not current.is_file()
            or digest(current.read_bytes()) != entry["sha256"]
            or (
                os.name != "nt"
                and stat.S_IMODE(current.stat().st_mode) != entry["mode"]
            )
        ):
            raise InstallError(
                "Installed file changed during recovery; refusing overwrite"
            )
    for entry in entries:
        current = prefix / entry["path"]
        current.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(transaction / entry["path"], current, follow_symlinks=False)
        except FileExistsError:
            # A newer pathname must survive even if it appeared during restore.
            if digest(current.read_bytes()) != entry["sha256"]:
                raise InstallError(
                    "Installed file changed during recovery; refusing overwrite"
                )
    retention._fsync_dir(prefix)
    _remove_transaction(transaction, entries)


def uninstall(prefix, apply=False):
    """Remove only verified owned files, preserving recordings and other additions."""
    prefix = checked_path(prefix)
    if not apply:
        return _uninstall(prefix, False)
    with retention._lock(prefix.parent, 30, "." + prefix.name + ".management.lock"):
        _restore_uninstall(prefix)
        return _uninstall(prefix, True)


def _uninstall(prefix, apply=False):
    prefix = checked_path(prefix)
    receipt = verify_installation(prefix)
    # Check every boundary before the first write. Unknown files are not touched.
    for relative in receipt["directories"]:
        checked_path(prefix / relative)
    report = {
        "status": "planned",
        "prefix": str(prefix),
        "remove_files": len(receipt["files"]),
        "service_changed": False,
        "unknown_files_preserved": True,
    }
    if not apply:
        return report
    transaction, entries = _prepare_uninstall(prefix, receipt)
    try:
        for entry in entries:
            _move_owned_file(prefix, transaction, entry)
        for relative in sorted(
            receipt["directories"],
            key=lambda name: len(PurePosixPath(name).parts),
            reverse=True,
        ):
            directory = prefix / relative
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()
        previous = receipt.get("preinstall", {})
        if previous.get("prefix_existed"):
            if os.name != "nt" and isinstance(previous.get("prefix_mode"), int):
                os.chmod(prefix, previous["prefix_mode"])
        elif not any(prefix.iterdir()):
            prefix.rmdir()
        retention._fsync_dir(prefix.parent)
    except BaseException:
        _restore_uninstall(prefix)
        raise
    _remove_transaction(transaction, entries)
    report["status"] = "uninstalled"
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    stage = subparsers.add_parser(
        "install", help="Plan or stage an isolated installation"
    )
    stage.add_argument("--upstream", required=True, type=Path)
    stage.add_argument("--prefix", required=True, type=Path)
    stage.add_argument("--config", required=True, type=Path)
    stage.add_argument("--apply", action="store_true")
    remove = subparsers.add_parser(
        "uninstall", help="Plan or remove unchanged installed files"
    )
    remove.add_argument("--prefix", required=True, type=Path)
    remove.add_argument("--apply", action="store_true")
    verify = subparsers.add_parser(
        "verify", help="Read-only installed-file integrity check"
    )
    verify.add_argument("--prefix", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.action == "install":
            package = Path(__file__).resolve().parent
            result = install(
                package,
                args.upstream,
                args.prefix,
                args.config,
                load_contract(package),
                apply=args.apply,
            )
        elif args.action == "uninstall":
            result = uninstall(args.prefix, apply=args.apply)
        else:
            receipt = verify_installation(args.prefix)
            result = {
                "status": "integrity_verified",
                "files": len(receipt["files"]),
                "upstream_commit": receipt["upstream_commit"],
                "device_validation": "not_performed",
                "service_changed": False,
            }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (
        InstallError,
        retention.RetentionError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
    ) as error:
        print("Management operation failed: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
