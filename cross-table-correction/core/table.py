from core.cell import Cell
from core.column import Column


class Table:
    def __init__(self, table_name, table_id, dataframe, clean_dataframe=None):
        self.table_name = table_name
        self.table_id = table_id
        self.dataframe = dataframe
        self.clean_dataframe = clean_dataframe
        self.columns = {}  # col_idx -> Column
        self._initialize_columns()
        self.samples = []
        self.candidate_models_initialized = False

    def _initialize_columns(self):
        for col_idx in range(self.dataframe.shape[1]):
            column = Column(self.table_id, col_idx)
            self.columns[col_idx] = column

    def get_all_cells(self):
        cells = []
        for column in self.columns.values():
            cells.extend(column.cells.values())
        return cells

    def get_error_cells(self):
        """Get all error cells in the table"""
        return [cell for cell in self.get_all_cells() if cell.is_error]

    def add_sample(self, cell):
        """Add a sample."""
        self.samples.append(cell)

    def get_samples(self):
        """Get all samples from the table."""
        return self.samples

    def generate_candidates_for_errors(self, candidate_generators):
        """Generate candidates for all error cells using provided generators"""
        error_cells = self.get_error_cells()

        for cell in error_cells:
            # Clear existing candidates
            cell.clear_candidates()

            # Generate candidates using each generator
            for generator in candidate_generators:
                candidates = generator.generate_candidates(cell, self)
                cell.add_candidates(candidates)

    def get_cell(self, row_idx: int, col_idx: int) -> Cell:
        """Get a specific cell by coordinates"""
        if col_idx in self.columns and row_idx in self.columns[col_idx].cells:
            return self.columns[col_idx].cells[row_idx]
        return None
