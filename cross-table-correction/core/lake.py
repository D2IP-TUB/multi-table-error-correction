from dataclasses import dataclass, field
from typing import Dict

from core.table import Table


@dataclass
class Lake:
    tables: Dict[str, "Table"] = field(default_factory=dict)
    n_cells: int = 0
    n_tables: int = 0
    n_columns: int = 0
    n_errors: int = 0

    def add_table(self, table_id, table):
        self.tables[table_id] = table

    def get_all_cells(self):
        cells = []
        for table in self.tables.values():
            cells.extend(table.get_all_cells())
        return cells
