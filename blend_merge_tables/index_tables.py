import csv
import duckdb
from pathlib import Path
from tqdm import tqdm

import config
from utils import tokenize_text


def setup_database(db_conn):
    """
    Initialize a DuckDB database with three tables to index cells, columns, and tables, respectively.
    :param db_conn: connection to the DuckDB database used to store the BLEND index.
    """

    # Cell index
    cell_idx_schema = """
        tab_id INTEGER,
        col_id INTEGER,
        row_id INTEGER,
        value VARCHAR,
        tokenized VARCHAR,
        is_clean BOOLEAN
    """
    db_conn.execute(f"CREATE TABLE cell_idx ({cell_idx_schema});")

    # Column index
    col_idx_schema = """
        tab_id INTEGER,
        col_id INTEGER,
        header VARCHAR,
        is_numeric BOOLEAN
    """
    db_conn.execute(f"CREATE TABLE col_idx ({col_idx_schema});")

    # Table index
    tab_idx_schema = """
        tab_id INTEGER,
        name VARCHAR
    """
    db_conn.execute(f"CREATE TABLE tab_idx ({tab_idx_schema});")


def update_database(db_conn, tuples, type):
    """
    Initialize a DuckDB database with three tables to index cells, columns, and tables, respectively.
    :param db_conn: connection to the DuckDB database used to store the BLEND index.
    :param tuples: tuples to be inserted into the BLEND index.
    :param type: name of the index into which tuples have to be inserted ('cells', 'columns', 'tables').
    """

    # Table index
    if type == 'tables':
        db_conn.executemany("""
            INSERT INTO tab_idx (tab_id, name)
            VALUES (?, ?)
        """, tuples)

    # Column index
    elif type == 'columns':
        db_conn.executemany("""
            INSERT INTO col_idx (tab_id, col_id, header, is_numeric)
            VALUES (?, ?, ?, ?)
        """, tuples)

    # Cell index
    else:
        db_conn.executemany("""
            INSERT INTO cell_idx (tab_id, col_id, row_id, value, tokenized, is_clean)
            VALUES (?, ?, ?, ?, ?, ?)
        """, tuples)


def index_tables(dir_path, db_path, batch_size, tab_limit=None):
    """
    Create BLEND index for the tables in the corpus.
    :param dir_path: path (pathlib.Path) to the directory containing the tables (each as a folder, with a dirty.csv file inside).
    :param db_path: path to the DuckDB database used to store the BLEND index.
    :param batch_size: number of tables to store into the BLEND index at a time.
    :param tab_limit: number of tables to index (only active if integer >= 0).
    """

    try:
        tab_limit = int(tab_limit)
        if tab_limit <= 0:
            tab_limit = None
    except Exception:
        tab_limit = None

    if db_path.exists():
        db_path.unlink()

    db_conn = duckdb.connect(db_path)

    setup_database(db_conn)

    tab_paths = sorted([p for p in dir_path.iterdir() if p.is_dir()])

    if tab_limit:
        tab_paths = tab_paths[:min(tab_limit, len(tab_paths))]

    tab_id = 0

    tab_idx_tups = list()
    col_idx_tups = list()
    cell_idx_tups = list()

    db_conn.begin()

    for tab_path in tqdm(tab_paths):

        tab_name = tab_path.stem
        tab_idx_tups.append((tab_id, tab_name))

        with open(tab_path / 'clean.csv', encoding='latin1') as file:
            csv_reader = csv.reader(file)
            clean_header = next(csv_reader)
            clean_data = [list(col) for col in zip(*list(csv_reader))]

        with open(tab_path / 'dirty.csv', encoding='latin1') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)
            data = [list(col) for col in zip(*list(csv_reader))]

        for col_id in range(len(header)):

            domain = set()

            for row_id in range(len(data[col_id])):
                cell_value = data[col_id][row_id]
                is_clean = cell_value == clean_data[col_id][row_id]
                if is_clean:
                    domain.add(cell_value)
                cell_idx_tups.append((tab_id, col_id, row_id, cell_value, tokenize_text(cell_value), is_clean))

                if len(cell_idx_tups) >= batch_size:
                    update_database(db_conn, cell_idx_tups, 'cells')
                    cell_idx_tups.clear()

            try:
                [float(v) for v in list(domain.difference({''}))]
                is_numeric = True
            except ValueError:
                is_numeric = False
            col_idx_tups.append((tab_id, col_id, header[col_id], is_numeric))

            if len(col_idx_tups) >= batch_size:
                update_database(db_conn, col_idx_tups, 'columns')
                col_idx_tups.clear()

        if len(tab_idx_tups) >= batch_size:
            update_database(db_conn, tab_idx_tups, 'tables')
            tab_idx_tups.clear()

        tab_id += 1

    if len(cell_idx_tups):
        update_database(db_conn, cell_idx_tups, 'cells')
        cell_idx_tups.clear()

    if len(col_idx_tups):
        update_database(db_conn, col_idx_tups, 'columns')
        col_idx_tups.clear()
    
    if len(tab_idx_tups):
        update_database(db_conn, tab_idx_tups, 'tables')
        tab_idx_tups.clear()

    db_conn.commit()


def main():
    index_tables(config.DIR_PATH, config.DB_PATH, config.BATCH_SIZE, config.TAB_LIMIT)


if __name__ == '__main__':
    main()

