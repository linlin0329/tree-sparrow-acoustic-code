# SPDX-License-Identifier: MIT
"""Filesystem and real temporary-Git tests; never touch services or devices."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import manage as installer


@unittest.skipUnless(shutil.which("git"), "Git is required for checkout fixture tests")
class InstallTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="retention install ")
        self.root = Path(self.temp.name)
        self.upstream = self.root / "clean upstream"
        self.upstream.mkdir()
        self.git("init")
        self.git("config", "user.email", "fixture@example.invalid")
        self.git("config", "user.name", "Offline Test Fixture")
        self.git("config", "core.autocrlf", "false")
        (self.upstream / "server.py").write_bytes(b"fixture = True\n")
        self.git("add", "server.py")
        self.git("commit", "-m", "fixture")
        self.commit = self.git("rev-parse", "HEAD").strip()
        self.contract = installer.InstallerContract(
            self.commit,
            {"server.py": installer.digest((self.upstream / "server.py").read_bytes())},
        )
        self.package = self.root / "package"
        self.package.mkdir()
        for relative in (
            "pipeline.py",
            "model_bridge.py",
            "retention.py",
            "manage.py",
            "vendor/server.py",
            "vendor/LICENSE",
            "LICENSE",
            "LICENSE-MIT",
            "upstream_contract.json",
        ):
            target = self.package / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"offline fixture\n")
        shutil.copyfile(
            Path(__file__).resolve().parents[1] / "vendor/labels.txt",
            self.package / "vendor/labels.txt",
        )
        self.config = self.root / "input config.json"
        self.config.write_text(
            json.dumps(
                {
                    "runtime_user": "birdrecorder",
                    "runtime_python": "/home/birdrecorder/runtime with spaces/bin/python3",
                    "pending_dir": str(self.root / "pending"),
                    "archive_dir": str(self.root / "archive"),
                }
            ),
            encoding="utf-8",
        )
        self.prefix = self.root / "installed prefix"

    def tearDown(self):
        self.temp.cleanup()

    def git(self, *args):
        return subprocess.run(
            ["git", "-C", str(self.upstream), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    def install(self, **kwargs):
        options = dict(
            package=self.package,
            upstream=self.upstream,
            prefix=self.prefix,
            config_path=self.config,
            contract=self.contract,
        )
        options.update(kwargs)
        return installer.install(**options)

    def snapshot(self):
        return {
            p.relative_to(self.root).as_posix(): p.read_bytes()
            for p in self.root.rglob("*")
            if p.is_file() and not p.name.endswith(".management.lock")
        }

    def test_dry_run_writes_nothing(self):
        before = self.snapshot()
        result = self.install()
        self.assertEqual(result["status"], "planned")
        self.assertFalse(self.prefix.exists())
        self.assertEqual(before, self.snapshot())

    def test_install_twice_is_idempotent_and_unit_is_only_proposed(self):
        self.assertEqual(self.install(apply=True)["status"], "staged")
        before = self.snapshot()
        self.assertEqual(self.install(apply=True)["status"], "already_installed")
        self.assertEqual(before, self.snapshot())
        receipt = installer.verify_installation(self.prefix)
        self.assertEqual(receipt["upstream_commit"], self.commit)
        self.assertEqual(
            (self.prefix / "config.json").read_bytes(), self.config.read_bytes()
        )
        unit = (
            self.prefix / "systemd" / "birdnet-retention.service.proposed"
        ).read_text()
        self.assertIn(' --config "', unit)
        self.assertIn("User=birdrecorder", unit)
        self.assertIn(
            'ExecStart="/home/birdrecorder/runtime with spaces/bin/python3" ', unit
        )
        self.assertFalse(
            (self.prefix / "systemd" / "birdnet-retention.service").exists()
        )

    def test_installed_label_table_matches_runtime_contract(self):
        self.install(apply=True)
        installed = self.prefix / "vendor/labels.txt"
        self.assertEqual(
            installer.digest(installed.read_bytes()), installer.pipeline.LABELS_SHA256
        )
        self.assertEqual(
            installed.read_text(encoding="utf-8")
            .splitlines()
            .count("Passer montanus_麻雀"),
            1,
        )

    def test_changed_packaged_labels_are_refused_before_installation(self):
        (self.package / "vendor/labels.txt").write_text(
            "Passer montanus_Tree Sparrow\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(installer.InstallError, "Chinese label SHA-256"):
            self.install(apply=True)
        self.assertFalse(self.prefix.exists())

    def test_wrong_commit_is_refused(self):
        bad = installer.InstallerContract("0" * 40, self.contract.files)
        with self.assertRaisesRegex(installer.InstallError, "Wrong upstream commit"):
            self.install(contract=bad, apply=True)
        self.assertFalse(self.prefix.exists())

    def test_wrong_hash_is_refused(self):
        bad = installer.InstallerContract(self.commit, {"server.py": "0" * 64})
        with self.assertRaisesRegex(installer.InstallError, "SHA-256 mismatch"):
            self.install(contract=bad, apply=True)
        self.assertFalse(self.prefix.exists())

    def test_dirty_tracked_file_is_refused(self):
        (self.upstream / "server.py").write_text("changed\n")
        with self.assertRaisesRegex(installer.InstallError, "tracked changes"):
            self.install(apply=True)
        self.assertFalse(self.prefix.exists())

    def test_existing_nonempty_directory_is_preserved(self):
        self.prefix.mkdir()
        original = self.prefix / "recording.wav"
        original.write_bytes(b"valuable sound")
        before = self.snapshot()
        with self.assertRaisesRegex(installer.InstallError, "no installation receipt"):
            self.install(apply=True)
        self.assertEqual(before, self.snapshot())

    def test_empty_directory_state_is_backed_up_and_restored(self):
        self.prefix.mkdir(mode=0o750)
        mode = stat.S_IMODE(self.prefix.stat().st_mode)
        self.install(apply=True)
        backup = json.loads((self.prefix / "backup" / "preinstall.json").read_text())
        self.assertTrue(backup["prefix_existed"])
        self.assertEqual(backup["prefix_mode"], mode)
        installer.uninstall(self.prefix, apply=True)
        self.assertTrue(self.prefix.is_dir())
        self.assertEqual(list(self.prefix.iterdir()), [])
        if os.name != "nt":
            self.assertEqual(stat.S_IMODE(self.prefix.stat().st_mode), mode)

    def test_uninstall_preserves_generated_files_and_neighbours(self):
        neighbour = self.root / "other recorder.wav"
        neighbour.write_bytes(b"untouched")
        self.install(apply=True)
        generated = self.prefix / "audio" / "new.wav"
        generated.parent.mkdir()
        generated.write_bytes(b"keep")
        nested_unknown = self.prefix / "vendor" / "user-note.txt"
        nested_unknown.write_text("keep")
        before = self.snapshot()
        self.assertEqual(installer.uninstall(self.prefix)["status"], "planned")
        self.assertEqual(before, self.snapshot())
        self.assertEqual(
            installer.uninstall(self.prefix, apply=True)["status"], "uninstalled"
        )
        self.assertEqual(generated.read_bytes(), b"keep")
        self.assertEqual(neighbour.read_bytes(), b"untouched")
        self.assertEqual(nested_unknown.read_text(), "keep")
        self.assertFalse((self.prefix / "config.json").exists())

    def test_modified_file_aborts_rollback_before_any_removal(self):
        self.install(apply=True)
        (self.prefix / "pipeline.py").write_text("user changes\n")
        before = self.snapshot()
        with self.assertRaisesRegex(installer.InstallError, "content changed"):
            installer.uninstall(self.prefix, apply=True)
        self.assertEqual(before, self.snapshot())

    def test_missing_file_aborts_rollback_before_any_removal(self):
        self.install(apply=True)
        (self.prefix / "pipeline.py").unlink()
        before = self.snapshot()
        with self.assertRaisesRegex(installer.InstallError, "missing"):
            installer.uninstall(self.prefix, apply=True)
        self.assertEqual(before, self.snapshot())

    @unittest.skipIf(os.name == "nt", "POSIX file modes")
    def test_modified_permissions_abort_rollback(self):
        self.install(apply=True)
        os.chmod(self.prefix / "config.json", 0o644)
        with self.assertRaisesRegex(installer.InstallError, "permissions changed"):
            installer.uninstall(self.prefix, apply=True)
        self.assertTrue((self.prefix / "pipeline.py").exists())

    def test_failed_staging_leaves_no_prefix_or_temporary_payload(self):
        original = installer.write_file
        calls = []

        def fail_second(*args):
            calls.append(args[0])
            if len(calls) == 2:
                raise OSError("simulated full storage")
            original(*args)

        before = self.snapshot()
        with mock.patch.object(installer, "write_file", side_effect=fail_second):
            with self.assertRaisesRegex(OSError, "simulated full storage"):
                self.install(apply=True)
        self.assertEqual(before, self.snapshot())
        self.assertFalse(self.prefix.exists())
        self.assertFalse(list(self.root.glob(".*.staging-*")))

    def test_changed_configuration_requires_explicit_uninstall(self):
        self.install(apply=True)
        config = json.loads(self.config.read_text())
        config["runtime_user"] = "anotheruser"
        self.config.write_text(json.dumps(config))
        before = self.snapshot()
        with self.assertRaisesRegex(installer.InstallError, "different installation"):
            self.install(apply=True)
        self.assertEqual(before, self.snapshot())

    def test_prefix_inside_upstream_or_package_is_refused(self):
        for parent in (self.upstream, self.package):
            with self.subTest(parent=parent):
                with self.assertRaisesRegex(installer.InstallError, "must be separate"):
                    self.install(prefix=parent / "new-install", apply=True)

    def test_symlink_boundary_is_refused(self):
        actual = self.root / "actual"
        actual.mkdir()
        link = self.root / "alias"
        try:
            link.symlink_to(actual, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("Host does not allow creating symlinks")
        with self.assertRaisesRegex(installer.InstallError, "Symlink/junction"):
            self.install(prefix=link / "new-install", apply=True)
        self.assertEqual(list(actual.iterdir()), [])

    def test_path_traversal_in_receipt_is_refused(self):
        self.install(apply=True)
        receipt_path = self.prefix / installer.RECEIPT
        receipt = json.loads(receipt_path.read_text())
        receipt["files"][0]["path"] = "../outside"
        changed = json.dumps(receipt).encode("utf-8")
        receipt_path.write_bytes(changed)
        (self.prefix / installer.RECEIPT_HASH).write_text(
            installer.digest(changed) + "\n"
        )
        with self.assertRaisesRegex(installer.InstallError, "Unsafe relative path"):
            installer.uninstall(self.prefix, apply=True)
        self.assertTrue((self.prefix / "config.json").exists())

    def test_modified_receipt_aborts_rollback(self):
        self.install(apply=True)
        receipt_path = self.prefix / installer.RECEIPT
        receipt_path.write_bytes(receipt_path.read_bytes() + b" ")
        with self.assertRaisesRegex(installer.InstallError, "Receipt integrity"):
            installer.uninstall(self.prefix, apply=True)
        self.assertTrue((self.prefix / "config.json").exists())

    def test_runtime_python_must_be_explicit_absolute_path(self):
        for value in (None, "python3", "/usr/bin/python3\nUser=root"):
            config = {"runtime_user": "birdrecorder", "runtime_python": value}
            self.config.write_text(json.dumps(config))
            with self.assertRaisesRegex(installer.InstallError, "runtime_python"):
                self.install(apply=True)
        self.assertFalse(self.prefix.exists())

    def test_configuration_bom_is_refused_before_staging(self):
        self.config.write_bytes(b"\xef\xbb\xbf" + self.config.read_bytes())
        with self.assertRaisesRegex(installer.InstallError, "without a BOM"):
            self.install(apply=True)
        self.assertFalse(self.prefix.exists())

    def test_production_commit_cannot_be_replaced_by_manifest(self):
        (self.package / "upstream_contract.json").write_text(
            json.dumps({"upstream_commit": self.commit, "files": self.contract.files})
        )
        with self.assertRaisesRegex(installer.InstallError, "pinned upstream commit"):
            installer.load_contract(self.package)

    def test_ancestor_prefix_is_refused(self):
        with self.assertRaisesRegex(installer.InstallError, "must be separate"):
            self.install(prefix=self.root, apply=True)

    def test_data_directory_must_not_overlap_installation_or_sources(self):
        original = json.loads(self.config.read_text())
        for root in (self.prefix, self.package, self.upstream):
            with self.subTest(root=root):
                self.config.write_text(
                    json.dumps({**original, "pending_dir": str(root / "data")})
                )
                with self.assertRaisesRegex(installer.InstallError, "must be separate"):
                    self.install(apply=True)
        self.assertFalse(self.prefix.exists())

    def test_interrupted_uninstall_recovers_before_retry(self):
        self.install(apply=True)
        receipt = installer.verify_installation(self.prefix)
        installer._prepare_uninstall(self.prefix, receipt)
        (self.prefix / "pipeline.py").unlink()
        with self.assertRaisesRegex(installer.InstallError, "Interrupted uninstall"):
            self.install(apply=True)
        result = installer.uninstall(self.prefix, apply=True)
        self.assertEqual(result["status"], "uninstalled")
        self.assertFalse(self.prefix.exists())
        self.assertFalse(installer._transaction_path(self.prefix).exists())

    def test_failed_uninstall_restores_all_installed_files(self):
        self.install(apply=True)
        before = self.snapshot()
        rename = installer.os.rename
        failed = []

        def fail_once(path, *args, **kwargs):
            path = Path(path)
            if path.name == "pipeline.py" and path.parent == self.prefix and not failed:
                failed.append(True)
                raise OSError("simulated interruption during removal")
            return rename(path, *args, **kwargs)

        with mock.patch.object(installer.os, "rename", fail_once):
            with self.assertRaisesRegex(OSError, "simulated interruption"):
                installer.uninstall(self.prefix, apply=True)
        self.assertEqual(before, self.snapshot())
        installer.verify_installation(self.prefix)

    def test_operator_edit_after_uninstall_backup_survives(self):
        self.install(apply=True)
        prepare = installer._prepare_uninstall
        changed = self.prefix / "pipeline.py"

        def edit_after_backup(prefix, receipt):
            transaction, entries = prepare(prefix, receipt)
            changed.write_bytes(b"operator edit after backup")
            return transaction, entries

        with mock.patch.object(installer, "_prepare_uninstall", edit_after_backup):
            with self.assertRaisesRegex(
                installer.InstallError, "changed during recovery"
            ):
                installer.uninstall(self.prefix, apply=True)
        self.assertEqual(changed.read_bytes(), b"operator edit after backup")
        transaction = installer._transaction_path(self.prefix)
        self.assertTrue(transaction.exists())
        self.assertEqual(
            (transaction / "removed/pipeline.py").read_bytes(), changed.read_bytes()
        )

    def test_edit_at_last_moment_before_atomic_move_survives(self):
        self.install(apply=True)
        rename = installer.os.rename
        changed = self.prefix / "pipeline.py"

        def edit_then_rename(source, destination):
            if Path(source) == changed:
                changed.write_bytes(b"last-moment operator edit")
            return rename(source, destination)

        with mock.patch.object(installer.os, "rename", edit_then_rename):
            with self.assertRaisesRegex(
                installer.InstallError, "changed during recovery"
            ):
                installer.uninstall(self.prefix, apply=True)
        self.assertEqual(changed.read_bytes(), b"last-moment operator edit")
        self.assertTrue(installer._transaction_path(self.prefix).exists())

    def test_original_path_replacement_and_changed_detached_file_both_survive(self):
        self.install(apply=True)
        rename = installer.os.rename
        changed = self.prefix / "pipeline.py"

        def replace_after_move(source, destination):
            rename(source, destination)
            if Path(source) == changed:
                Path(destination).write_bytes(b"edited detached object")
                changed.write_bytes(b"replacement at original path")

        with mock.patch.object(installer.os, "rename", replace_after_move):
            with self.assertRaisesRegex(
                installer.InstallError, "both versions retained"
            ):
                installer.uninstall(self.prefix, apply=True)
        self.assertEqual(changed.read_bytes(), b"replacement at original path")
        transaction = installer._transaction_path(self.prefix)
        self.assertEqual(
            (transaction / "removed/pipeline.py").read_bytes(),
            b"edited detached object",
        )

    def test_new_replacement_after_move_is_not_deleted(self):
        self.install(apply=True)
        rename = installer.os.rename
        changed = self.prefix / "pipeline.py"

        def replace_after_move(source, destination):
            rename(source, destination)
            if Path(source) == changed:
                changed.write_bytes(b"new replacement at original path")

        with mock.patch.object(installer.os, "rename", replace_after_move):
            result = installer.uninstall(self.prefix, apply=True)
        self.assertEqual(result["status"], "uninstalled")
        self.assertEqual(changed.read_bytes(), b"new replacement at original path")

    def test_recovery_never_overwrites_changed_surviving_file(self):
        self.install(apply=True)
        receipt = installer.verify_installation(self.prefix)
        installer._prepare_uninstall(self.prefix, receipt)
        changed = self.prefix / "pipeline.py"
        changed.write_bytes(b"new user content")
        with self.assertRaisesRegex(installer.InstallError, "changed during recovery"):
            installer.uninstall(self.prefix, apply=True)
        self.assertEqual(changed.read_bytes(), b"new user content")
        self.assertTrue(installer._transaction_path(self.prefix).exists())

    def test_systemd_environment_expansion_is_escaped(self):
        self.assertEqual(installer.systemd_quote("/srv/$DATA"), '"/srv/$$DATA"')

    def test_service_specifier_is_escaped(self):
        self.assertEqual(
            installer.systemd_quote('/home/user/100% "space"'),
            '"/home/user/100%% \\"space\\""',
        )


if __name__ == "__main__":
    unittest.main()
