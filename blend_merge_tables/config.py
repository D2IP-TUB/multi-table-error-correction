import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Error detection mode (used by indexing, discovery, AND merging):
#   'GT'       - compare dirty.csv vs clean.csv (default)
#   'DETECTED' - read per-table detected_errors CSV with columns row,column
ERROR_MODE = 'GT'
DETECTED_ERRORS_FILENAME = 'detected_errors.csv'
# If set, detection files are read from DETECTED_ERRORS_DIR / <table_name> / DETECTED_ERRORS_FILENAME
# instead of DIR_PATH / <table_name> / DETECTED_ERRORS_FILENAME.
DETECTED_ERRORS_DIR = None

# Offline phase (create BLEND index)
BATCH_SIZE = 10_000  # Number of tuples to store into the BLEND index at a time
TAB_LIMIT = -1  # Number of tables to index (only active if integer >= 0)
TOKENIZE = False

# -------------------------------------------------- #

TOP_SEARCH = 100  # Top-k overlapping columns to retrieve through the single-column seeker

# Join
JOIN = True
JOIN_NUMERIC = True
# Optional post-discovery merge validation (paper §5.9.3/5.9.4: off by default).
MERGE_VALIDATION = False
# When MERGE_VALIDATION is True, which operations to score:
#   'all'   - validate joins and unions
#   'join'  - validate joins only (unions accepted without scoring)
#   'union' - validate unions only
VALIDATION_SCOPE = 'all'
# Validation strategy for merge candidate selection:
#   'fd'                  - FD correctability only
#   'distribution'        - TVD (categorical) / KS (numeric) gate only
#   'fd_and_distribution' - FD score with distribution hard gate
VALIDATION_STRATEGY = 'fd'
DIST_TVD_THRESHOLD = 1.0  # Max TVD for categorical columns (1.0 = only reject total mismatch)
DIST_KS_THRESHOLD = 1.0  # Max KS for numeric columns (1.0 = only reject total mismatch)
TOP_JOIN = 10  # Top-k joinable columns to retrieve for every primary key
JOIN_THRESHOLD = 0.5  # Minimum ratio of joined tuples over the length of the joined table
JOIN_ROWS = 0.1  # Minimum ratio of tuples joined for each table

# Union
UNION = True
TOP_UNION = 10  # Top-k unionable tables to retrieve for every table
UNION_THRESHOLD = 0.5  # Minimum ratio of matching tuples over the length of the joined table
UNION_COLS = 0.5  # Minimum ratio of matching columns over the total of both tables

# Merge candidate ordering during lake construction:
#   'union' - compare top union against competing joins (default)
#   'join'  - compare top join against competing unions
MERGE_PRIORITY = 'union'

# When set, string-recreation outputs are written under RESULTS_ROOT/ instead of cwd.
RESULTS_ROOT = None

# -------------------------------------------------- #

# Paths (repo-relative; override as needed)
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATASETS = _REPO_ROOT / 'datasets'
_BLEND_RESULTS = _REPO_ROOT / 'results' / 'blend'

# 'open_data_uk' (bundled) | 'mit_dwh' (proprietary — place under datasets/real_lakes/mit_dwh/)
# Legacy alias 'uk_open_data' maps to open_data_uk.
CORPUS = 'open_data_uk'

_CORPUS_ISOLATED = {
    'open_data_uk': _DATASETS / 'real_lakes' / 'open_data_uk',
    'uk_open_data': _DATASETS / 'real_lakes' / 'open_data_uk',
    'mit_dwh': _DATASETS / 'real_lakes' / 'mit_dwh',
}

DIR_PATH = _CORPUS_ISOLATED[CORPUS]  # isolated corpus tables
MERGED_PATH = _BLEND_RESULTS / CORPUS / 'merged'
DB_PATH = _BLEND_RESULTS / CORPUS / f'{CORPUS}_blend_index.duckdb'
TRACKER_PATH = MERGED_PATH / 'tracker.json'

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
    'MERGE_VALIDATION',
    'VALIDATION_SCOPE',
    'VALIDATION_STRATEGY',
    'DIST_TVD_THRESHOLD',
    'DIST_KS_THRESHOLD',
    'TOP_JOIN',
    'JOIN_THRESHOLD',
    'JOIN_ROWS',
    'UNION',
    'TOP_UNION',
    'UNION_THRESHOLD',
    'UNION_COLS',
    'MERGE_PRIORITY',
    'RESULTS_ROOT',
    'CORPUS',
    'DIR_PATH',
    'MERGED_PATH',
    'DB_PATH',
    'TRACKER_PATH',
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

