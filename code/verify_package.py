#!/usr/bin/env python3
"""Verify package bytes and membership against both distributed manifests.

Run before modifying/building the source tree. Generated caches are intentionally
not part of the immutable distribution and will be reported as extra files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def verify(root):
    root = root.resolve()
    with (root / 'manifest.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    expected = {}
    for row in rows:
        relative = Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts or row['path'] in expected:
            raise ValueError('Unsafe or duplicate manifest path: ' + row['path'])
        target = root / relative
        if target.is_symlink() or not target.resolve().is_relative_to(root):
            raise ValueError('Manifest path is a link or escapes the package')
        if not target.is_file() or target.stat().st_size != int(row['bytes']):
            raise ValueError('Missing or size-mismatched file: ' + row['path'])
        value = sha256(target)
        if value != row['sha256']:
            raise ValueError('Content hash mismatch: ' + row['path'])
        expected[row['path']] = value
    actual = set()
    for target in root.rglob('*'):
        if target.is_symlink():
            raise ValueError('Unexpected symbolic link: ' + str(target))
        if target.is_file():
            actual.add(target.relative_to(root).as_posix())
    allowed = set(expected) | {'manifest.csv', 'SHA256SUMS'}
    if actual != allowed:
        raise ValueError('Unlisted/missing package members: ' + repr(sorted(actual ^ allowed)))
    checks = {}
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        value, relative = line.split('  ', 1)
        if relative in checks:
            raise ValueError('Duplicate SHA256SUMS entry')
        checks[relative] = value
    if set(checks) != set(expected) | {'manifest.csv'}:
        raise ValueError('SHA256SUMS membership mismatch')
    for relative, value in checks.items():
        if value != (sha256(root / relative) if relative == 'manifest.csv' else expected[relative]):
            raise ValueError('SHA256SUMS mismatch: ' + relative)
    return {'status': 'passed', 'content_files': len(expected),
            'total_files': len(actual), 'bytes': sum((root / p).stat().st_size for p in actual)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.root), indent=2))
