# SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""Pinned historical 6K score core, without importing the legacy server.

The SHA-256 guarded AST allowlist compiles eight historical function definitions.
Only loadModel's modelpath/labelspath assignments and predict's HUMAN.txt path
are redirected to explicit configuration. Text open defaults to UTF-8. Socket,
threads, database, notification, configuration parsing and server.start are never
executed. The historical score core is retained; the surrounding runtime is new.
No model binary is distributed and offline mocks do not validate inference.
"""

import ast
import builtins
import datetime
import importlib
import math
import operator
from pathlib import Path
import time

from retention import sha256

SOURCE_SHA256 = "a108bde1ccb0bffb61711ec9321160254f48792dbd637a6227d1f66f48532050"
MODEL_SHA256 = "f073d0aa626b4a5bb0705982d06de74c562f041ba76ebada18bc3e29126f6755"
LABELS_SHA256 = "8008d3d16e8f0fad06895ce4cea88d5a90aeaecad9325353c34ada27f331ed8d"
FUNCTIONS = (
    "loadModel",
    "splitSignal",
    "readAudioData",
    "convertMetadata",
    "custom_sigmoid",
    "predict",
    "analyzeAudioData",
    "writeResultsToFile",
)
SOURCE = Path(__file__).resolve().parent / "vendor/server.py"


def _utf8_open(file, mode="r", *args, **kwargs):
    if "b" not in mode:
        kwargs.setdefault("encoding", "utf-8")
    return builtins.open(file, mode, *args, **kwargs)


def core_ast(source=SOURCE):
    """Verify the entire pinned source before extracting or adapting definitions."""
    source = Path(source)
    if sha256(source) != SOURCE_SHA256:
        raise ValueError("Unsupported historical server.py SHA-256")
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS
    ]
    if {node.name for node in selected} != set(FUNCTIONS) or len(selected) != len(
        FUNCTIONS
    ):
        raise ValueError("Historical function allowlist mismatch")
    changed = {"modelpath": 0, "labelspath": 0, "human_log": 0}
    human_expr = ast.dump(
        ast.parse("userDir + '/BirdNET-Pi/HUMAN.txt'", mode="eval").body
    )
    for definition in selected:
        for node in ast.walk(definition):
            if definition.name == "loadModel" and isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in (
                        "modelpath",
                        "labelspath",
                    ):
                        node.value = ast.Name(
                            id="configured_" + target.id, ctx=ast.Load()
                        )
                        changed[target.id] += 1
            if (
                definition.name == "predict"
                and isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "open"
                and node.args
                and ast.dump(node.args[0]) == human_expr
            ):
                node.args[0] = ast.Name(id="configured_human_log", ctx=ast.Load())
                changed["human_log"] += 1
    if changed != {"modelpath": 1, "labelspath": 1, "human_log": 1}:
        raise ValueError("Historical path adaptation mismatch")
    return ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))


def compile_core(config, dependencies=None):
    """Prepare functions only. Optional fake dependencies are for offline tests."""
    tree = core_ast()
    if dependencies is None:
        dependencies = {
            "np": importlib.import_module("numpy"),
            "librosa": importlib.import_module("librosa"),
            "tflite": importlib.import_module("tflite_runtime.interpreter"),
        }
    namespace = dict(dependencies)
    namespace.update(
        {
            "math": math,
            "operator": operator,
            "time": time,
            "datetime": datetime,
            "open": _utf8_open,
            "model": "BirdNET_6K_GLOBAL_MODEL",
            "priv_thresh": 0.0,
            "configured_modelpath": str(config["model_path"]),
            "configured_labelspath": str(config["labels_path"]),
            "configured_human_log": str(
                Path(config["pending_dir"]) / "human_detections.log"
            ),
            "PREDICTED_SPECIES_LIST": [],
            "INCLUDE_LIST": [config["scientific_name"] + "_" + config["common_name"]],
            "EXCLUDE_LIST": [],
        }
    )
    exec(compile(tree, str(SOURCE), "exec"), namespace)
    return namespace


class ModelBridge:
    """One serial consumer only; historical interpreter state is not thread safe."""

    def __init__(self, config):
        self.config = dict(config)
        if self.config.get("privacy_threshold", 0) != 0:
            raise ValueError(
                "Only PRIVACY_THRESHOLD=0 is supported"
            )
        if (
            self.config.get("model", "BirdNET_6K_GLOBAL_MODEL")
            != "BirdNET_6K_GLOBAL_MODEL"
        ):
            raise ValueError("Only the pinned historical 6K model is supported")
        for name, expected in (
            ("model_path", MODEL_SHA256),
            ("labels_path", LABELS_SHA256),
        ):
            if sha256(config[name]) != expected:
                raise ValueError("Unsupported " + name + " SHA-256")
        target = config["scientific_name"] + "_" + config["common_name"]
        labels = Path(config["labels_path"]).read_text(encoding="utf-8").splitlines()
        if labels.count(target) != 1:
            raise ValueError(
                "The exact target label must occur once in the pinned label file"
            )
        self.core = compile_core(config)
        self.core["INTERPRETER"] = self.core["loadModel"]()

    def analyze(self, wav_path, csv_path, recorded_at):
        stamp = datetime.datetime.fromisoformat(recorded_at)
        if stamp.tzinfo is None or stamp.utcoffset() is None:
            raise ValueError("Recording timestamp must contain an explicit UTC offset")
        # The legacy handler derives inference metadata from recording date,
        # not file mtime or request-processing time.
        week = max(1, min(stamp.isocalendar()[1], 48))
        sensitivity = max(0.5, min(2.0 - self.config["sensitivity"], 1.5))
        chunks = self.core["readAudioData"](str(wav_path), self.config["overlap"])
        detections = self.core["analyzeAudioData"](
            chunks,
            self.config["latitude"],
            self.config["longitude"],
            week,
            sensitivity,
            self.config["overlap"],
        )
        self.core["writeResultsToFile"](
            detections, self.config["threshold"], str(csv_path)
        )
        return {
            "inference_week": week,
            "sigmoid_sensitivity": sensitivity,
            "windows": len(chunks),
            "model_sha256": MODEL_SHA256,
            "labels_sha256": LABELS_SHA256,
            "source_sha256": SOURCE_SHA256,
        }
