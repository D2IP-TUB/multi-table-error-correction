from pathlib import Path


# Offline phase (create BLEND index)
BATCH_SIZE = 10_000  # Number of tuples to store into the BLEND index at a time
TAB_LIMIT = -1  # Number of tables to index (only active if integer >= 0)
TOKENIZE = False

# -------------------------------------------------- #

TOP_SEARCH = 100  # Top-k overlapping columns to retrieve through the single-column seeker

# Join
JOIN = True
JOIN_NUMERIC = True
JOIN_VALIDATION = False  # Validate joins (True) or always accept them (False)
TOP_JOIN = 10  # Top-k joinable columns to retrieve for every primary key
JOIN_THRESHOLD = 0.5  # Minimum ratio of joined tuples over the length of the joined table
JOIN_ROWS = 0.1  # Minimum ratio of tuples joined for each table

# Union
UNION = True
TOP_UNION = 10  # Top-k unionable tables to retrieve for every table
UNION_THRESHOLD = 0.5  # Minimum ratio of matching tuples over the length of the joined table
UNION_COLS = 0.5  # Minimum ratio of matching columns over the total of both tables

# -------------------------------------------------- #

# Paths
CORPUS = 'mit_dwh'  # 'uk_open_data' | 'mit_dwh'
DIR_PATH = Path('/home/ahmadi/Blend_X/tables') / CORPUS / 'isolated'  # directory storing the original corpus
MERGED_PATH = Path('/home/ahmadi/Blend_X/tables') / CORPUS / 'merged'  # directory storing the merged tables
DB_PATH = Path('/home/ahmadi/Blend_X/indices') / f'{CORPUS}_blend_index.duckdb'
TRACKER_PATH = MERGED_PATH / ('tracker.json')

