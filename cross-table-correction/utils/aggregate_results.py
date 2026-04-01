param = -1
results_path = "/home/fatemeh/ECS-1iter/EC-at-Scale/exp-07-08-Quintet-no-pruning"
dataset_name = "Quintet_2"
# labeling_budgets = [28, 53, 76, 122, 186, 226]
# labeling_budgets = [50, 100, 300, 683, 1255, 1767, 2685, 3937, 4712]
labeling_budgets = [5, 10, 21, 40, 58, 91, 136]
# labeling_budgets = [1, 2, 3, 5, 8, 10]
execution_times = 1
n_errors = 16728  # Total number of errors in the dataset
import os

import pandas as pd

results_dict = {}
for exec in range(1, execution_times + 1):
    for labeling_budget in labeling_budgets:
        results_dict[labeling_budget] = {
            "precision": 0,
            "recall": 0,
            "f1_score": 0,
            # "f1_score_std": 0,
            "ec_tpfp": 0,
            "ec_tpfn": 0,
            "tp": 0,
            # "execution_time": 0,
        }

for exec in range(1, execution_times + 1):
    for labeling_budget in labeling_budgets:
        output_dir = (
            f"{results_path}/output_{dataset_name}_{exec}_{labeling_budget}_{param}"
        )
        if not os.path.exists(output_dir):
            print(f"Output directory does not exist: {output_dir}")
            continue

        with open(f"{output_dir}/evaluation_results.pickle", "rb") as f:
            results = pd.read_pickle(f)

        precision = results["tp"] / results["tp + fp"] if results["tp + fp"] > 0 else 0
        recall = results["tp"] / n_errors
        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        results_dict[labeling_budget]["tp"] += results["tp"]
        results_dict[labeling_budget]["ec_tpfp"] += results["tp + fp"]
        results_dict[labeling_budget]["ec_tpfn"] += results["total_error_cells"]
        results_dict[labeling_budget]["precision"] += precision
        results_dict[labeling_budget]["recall"] += recall
        results_dict[labeling_budget]["f1_score"] += f1_score
        # results_dict[labeling_budget]["execution_time"] += results["execution_times"][
        #     "total_time"
        # ]

for labeling_budget in labeling_budgets:
    results_dict[labeling_budget]["precision"] = (
        results_dict[labeling_budget]["precision"] / execution_times
    )
    results_dict[labeling_budget]["recall"] = (
        results_dict[labeling_budget]["recall"] / execution_times
    )
    results_dict[labeling_budget]["f1_score"] = (
        results_dict[labeling_budget]["f1_score"] / execution_times
    )
    results_dict[labeling_budget]["ec_tpfp"] = (
        results_dict[labeling_budget]["ec_tpfp"] / execution_times
    )
    results_dict[labeling_budget]["ec_tpfn"] = (
        results_dict[labeling_budget]["ec_tpfn"] / execution_times
    )
    results_dict[labeling_budget]["tp"] = (
        results_dict[labeling_budget]["tp"] / execution_times
    )
    # results_dict[labeling_budget]["execution_time"] = (
    #     results_dict[labeling_budget]["execution_time"] / execution_times
    # )

results_df = pd.DataFrame.from_dict(results_dict, orient="index")
results_df.index.name = "labeling_budget"
results_df.reset_index(inplace=True)
results_df.to_csv(f"{results_path}/aggregated_results.csv", index=False)
print("Aggregated results saved to aggregated_results.csv")
