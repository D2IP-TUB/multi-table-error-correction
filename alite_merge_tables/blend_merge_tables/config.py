import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Error detection mode (used by indexing):
#   'GT'       - compare dirty.csv vs clean.csv (default)
#   'DETECTED' - read per-table detected_errors CSV with columns row,column
# Affects index_tables: cell_idx.is_clean
# discover_clusters always uses dirty vs clean (not ERROR_MODE).
ERROR_MODE = 'GT'
DETECTED_ERRORS_FILENAME = 'detected_errors.csv'
# If set, detection files are read from DETECTED_ERRORS_DIR / <table_name> / DETECTED_ERRORS_FILENAME
# instead of DIR_PATH / <table_name> / DETECTED_ERRORS_FILENAME.
DETECTED_ERRORS_DIR = None

# Active corpus: 'open_data_uk' | 'mit_dwh'
CORPUS = 'open_data_uk'

# Offline phase (create BLEND index)
BATCH_SIZE = 10_000  # Number of tuples to store into the BLEND index at a time
TAB_LIMIT = -1  # Number of tables to index (only active if integer >= 0)
TOKENIZE = False

# -------------------------------------------------- #

TOP_SEARCH = 100  # Top-k overlapping columns to retrieve through the single-column seeker

# Join discovery
JOIN = True
JOIN_NUMERIC = True
TOP_JOIN = 10  # Top-k joinable columns to retrieve for every primary key
JOIN_THRESHOLD = 0.5  # Minimum ratio of joined tuples over the length of the joined table
JOIN_ROWS = 0.1  # Minimum ratio of tuples joined for each table

# Union discovery
UNION = True
TOP_UNION = 10  # Top-k unionable tables to retrieve for every table
UNION_THRESHOLD = 0.5  # Minimum ratio of matching tuples over the length of the joined table
UNION_COLS = 0.5  # Minimum ratio of matching columns over the total of both tables

# -------------------------------------------------- #

# Paper package root (parent of alite_merge_tables/) holds shared datasets/
_PAPER_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT = Path(__file__).resolve().parent.parent  # alite_merge_tables/
_DATASETS = _PAPER_ROOT / 'datasets'

_ISOLATED_DIRS = {
    'open_data_uk': _DATASETS / 'real_lakes' / 'open_data_uk',
    'mit_dwh':      _DATASETS / 'real_lakes' / 'mit_dwh',
}


def alite_results_dir(corpus: str | None = None) -> Path:
    return _REPO_ROOT / 'results' / 'alite' / (corpus or CORPUS)


def get_alite_paths(corpus: str) -> dict[str, Path]:
    base = alite_results_dir(corpus)
    return {
        'clusters': base / 'clusters_for_alite',
        'fd_output': base / 'fd_output',
        'stats': base / 'fd_statistics.csv',
    }


DIR_PATH = _ISOLATED_DIRS[CORPUS]
ALITE_RESULTS = alite_results_dir()
CLUSTERS_PATH = ALITE_RESULTS / 'clusters_for_alite'
FD_OUTPUT_PATH = ALITE_RESULTS / 'fd_output'
FD_STATS_PATH = ALITE_RESULTS / 'fd_statistics.csv'
DB_PATH = _REPO_ROOT / 'results' / 'alite' / f'{CORPUS}_blend_index.duckdb'

_CONFIG_KEYS = (
    'ERROR_MODE',
    'DETECTED_ERRORS_FILENAME',
    'DETECTED_ERRORS_DIR',
    'BATCH_SIZE',
    'TAB_LIMIT',
    'TOKENIZE',
    'TOP_SEARCH',
    'JOIN',
    'JOIN_NUMERIC',
    'TOP_JOIN',
    'JOIN_THRESHOLD',
    'JOIN_ROWS',
    'UNION',
    'TOP_UNION',
    'UNION_THRESHOLD',
    'UNION_COLS',
    'CORPUS',
    'DIR_PATH',
    'ALITE_RESULTS',
    'CLUSTERS_PATH',
    'FD_OUTPUT_PATH',
    'FD_STATS_PATH',
    'DB_PATH',
)


def experiment_config_dict(**extra: Any) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for key in _CONFIG_KEYS:
        value = globals()[key]
        if isinstance(value, Path):
            value = str(value)
        data[key] = value
    data.update(extra)
    data['saved_at'] = datetime.now(timezone.utc).isoformat()
    return data


def save_experiment_config(result_dir: Path | None = None, **extra: Any) -> Path:
    """Write the current settings to ``result_dir/experiment_config.json``."""
    if result_dir is None:
        result_dir = DB_PATH.parent
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    path = result_dir / 'experiment_config.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(experiment_config_dict(**extra), f, indent=2, sort_keys=True)
        f.write('\n')
    return path
