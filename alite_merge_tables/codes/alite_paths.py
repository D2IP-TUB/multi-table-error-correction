"""ALITE result paths (see blend_merge_tables.config)."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_BLEND = _REPO / "blend_merge_tables"
if str(_BLEND) not in sys.path:
    sys.path.insert(0, str(_BLEND))

import config  # noqa: E402

CORPORA = ("open_data_uk", "mit_dwh")


def get_paths(corpus: str) -> dict[str, Path]:
    return config.get_alite_paths(corpus)


def isolated_dir(corpus: str) -> Path:
    return config._ISOLATED_DIRS[corpus]
