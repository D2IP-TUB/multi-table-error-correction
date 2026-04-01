import configparser
import os
import random
import subprocess


def read_config(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    return config


def update_config(config, section, parameter, new_value):
    config.set(section, parameter, new_value)


def save_config(config, file_path):
    with open(file_path, "w") as configfile:
        config.write(configfile)


def run_exp(
    execution_times,
    dataset_budgets,
    vicinity_confidence_thresholds,
    output_path,
):
    config_file_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config", "config.ini"
)
    config = read_config(config_file_path)

    random_states = random.sample(range(1000), execution_times)
    for exec in range(3, execution_times + 1):
        random_state = random_states[exec - 1]
        for dataset_name, labeling_budgets in dataset_budgets.items():
            update_config(config, "DIRECTORIES", "tables_dir", dataset_name)
            for labeling_budget in labeling_budgets:
                for vicinity_confidence_threshold in vicinity_confidence_thresholds:
                    update_config(
                        config,
                        "PRUNING",
                        "vicinity_confidence_threshold",
                        str(vicinity_confidence_threshold),
                    )
                    update_config(
                        config, "EXPERIMENT", "random_state", str(random_state)
                    )
                    update_config(
                        config,
                        "DIRECTORIES",
                        "output_dir",
                        f"{output_path}/output_{dataset_name}_{exec}_{labeling_budget}_{vicinity_confidence_threshold}",
                    )
                    update_config(
                        config,
                        "DIRECTORIES",
                        "logs_dir",
                        f"{output_path}/logs_{dataset_name}_{exec}_{labeling_budget}_{vicinity_confidence_threshold}",
                    )
                    update_config(
                        config, "LABELING", "labeling_budget", str(labeling_budget)
                    )
                    save_config(config, config_file_path)
                    subprocess.run(
                        [
                            "python",
                            "-m",
                            "main",
                        ],  # assumes your main.py has `if __name__ == "__main__": main()`
                        cwd="/home/fatemeh/GuidedLakeCorrection",
                        check=True,
                    )

    print("✅ ESC execution finished!")


execution_times = 3
dataset_budgets = {
    
    # "open_data_uk_filtered": [838, 1520, 2081, 3097, 4523, 5379],
    # "DGov_NTR": [1259, 1767, 2677, 3951, 4734],
    # "Quintet_3": [10, 22, 42, 61, 96, 144, 173]
    # "flattened_partitioned_base": [917, 5167 ]
    "open_data_uk_93": [627, 4026, 2317]
}
output_path = "/home/fatemeh/GuidedLakeCorrection/results_clustering_based_fixed_sampling_open_data"
vicinity_confidence_thresholds = [-1]
if not os.path.exists(output_path):
    os.makedirs(output_path)
run_exp(
    execution_times,
    dataset_budgets,
    vicinity_confidence_thresholds,
    output_path,
)
