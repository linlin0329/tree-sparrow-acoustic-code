"""Component resource locations; no research-workspace paths."""

from pathlib import Path

COMPONENT = Path(__file__).resolve().parents[1]
ASSETS = COMPONENT / "assets"
