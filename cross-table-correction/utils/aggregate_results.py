param = -1
results_path = str(Path(__file__).resolve().parents[2] / "results" / "cross-table" / "output")
dataset_name = "Quintet_2"
# labeling_budgets = [28, 53, 76, 122, 186, 226]
# labeling_budgets = [50, 100, 300, 683, 1255, 1767, 2685, 3937, 4712]
labeling_budgets = [5, 10, 21, 40, 58, 91, 136]
# labeling_budgets = [1, 2, 3, 5, 8, 10]
execution_times = 1
n_errors = 16728  # Total number of errors in the dataset
from pathlib import Path
i