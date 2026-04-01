import os
import subprocess
import sys
from typing import Any, Dict

import yaml
from raha.send_log import send_log


def run_experiment_not_enough_labels_lpc():
    print("Running experiment...")
    result = subprocess.run([
        sys.executable, 
        "raha/ecs_run_experiments/baran_not_enough_labels_column_wise.py"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Experiment completed!")
        print(result.stdout)
    else:
        print("Experiment failed!")
        print(result.stderr)

    print("Experiment completed!")

class LakeCorrectionExperimentRunner:
    def __init__(self, yaml_file_path: str):
        self.yaml_file_path = yaml_file_path
    
    def update_shared_config(self, updates: Dict[str, Any]) -> None:
        """
        Update the shared section of the YAML file
        
        Args:
            updates: Dictionary containing the keys and values to update in shared section
        """
        if os.path.exists(self.yaml_file_path):
            with open(self.yaml_file_path, 'r') as file:
                data = yaml.safe_load(file) or {}
        else:
            data = {}
        
        if 'shared' not in data:
            data['shared'] = {}
        
        data['shared'].update(updates)
        
        with open(self.yaml_file_path, 'w') as file:
            yaml.dump(data, file, default_flow_style=False, indent=2)
        
        print(f"Updated shared config in {self.yaml_file_path}")
    
    def update_results_config(self, updates: Dict[str, Any]) -> None:
        """
        Update the results section of the YAML file
        
        Args:
            updates: Dictionary containing the keys and values to update in results section
        """
        if os.path.exists(self.yaml_file_path):
            with open(self.yaml_file_path, 'r') as file:
                data = yaml.safe_load(file) or {}
        else:
            data = {}
        
        if 'results' not in data:
            data['results'] = {}
        
        data['results'].update(updates)
        
        with open(self.yaml_file_path, 'w') as file:
            yaml.dump(data, file, default_flow_style=False, indent=2)
        
        print(f"Updated results config in {self.yaml_file_path}")
    
    def update_full_config(self, config_data: Dict[str, Any]) -> None:
        """
        Update the entire YAML file with new configuration
        
        Args:
            config_data: Full configuration dictionary
        """
        with open(self.yaml_file_path, 'w') as file:
            yaml.dump(config_data, file, default_flow_style=False, indent=2)
        
        print(f"Updated full config in {self.yaml_file_path}")
    
    def get_current_config(self) -> Dict[str, Any]:
        """
        Load and return the current configuration
        """
        with open(self.yaml_file_path, 'r') as file:
            return yaml.safe_load(file) or {}
    
    def run_experiment(self) -> None:
        """
        Run the experiment with current configuration
        """
        config = self.get_current_config()
        shared = config.get('shared', {})
        
        print("Running experiment with shared config:")
        print(f"  Sandbox path: {shared.get('sandbox_path')}")
        print(f"  Results path: {shared.get('results_path')}")
        print(f"  Dirty file: {shared.get('dirty_file_name')}")
        print(f"  Clean file: {shared.get('clean_file_name')}")
        print(f"  Repetitions: {shared.get('repetitions')}")
        
        run_experiment_not_enough_labels_lpc()


if __name__ == "__main__":
    dataset_names =  ["Quintet_3"]
    datasets_path = "/home/fatemeh/LakeCorrectionBench/datasets"
    
    shared_runner = LakeCorrectionExperimentRunner("/home/fatemeh/LakeCorrectionBench/Baran/Error-Correction-at-Scale/benchmarks/Baran_Experiments/raha/raha/ecs_run_experiments/hydra_configs/shared.yaml")
    results_runner = LakeCorrectionExperimentRunner("/home/fatemeh/LakeCorrectionBench/Baran/Error-Correction-at-Scale/benchmarks/Baran_Experiments/raha/raha/ecs_run_experiments/hydra_configs/results.yaml")
    
    for dataset_name in dataset_names:
        # send_log(f"Running experiment for dataset: {dataset_name}")
        full_shared_config = {
            "shared": {
                "sandbox_path": f"{datasets_path}/{dataset_name}",
                "results_path": "/home/fatemeh/LakeCorrectionBench/results_baran_lpc_10_2025", 
                "dirty_file_name": "dirty.csv",
                "clean_file_name": "clean.csv",
                "repetitions": 5
            }
        }
        shared_runner.update_full_config(full_shared_config)
        
        # full_results_config = {
        #     "defaults": ["*self*", "shared", "standard"],
        #     "results": {
        #         "path_to_benchmark_dataframe": f"/home/fatemeh/LakeCorrectionBench/Results-10-03/results-no-rs/{dataset_name}/exp_raha-enough-labels/baran_standard_onebyone_results.csv",
        #         "path_to_experiment_results_folder": f"/home/fatemeh/LakeCorrectionBench/Results-10-03/results-no-rs/{dataset_name}/exp_raha-enough-labels",
        #         "labeling_budget": [1, 2, 3, 5, 8, 10],
        #         "variant": "standard"
        #     }
        # }
        # results_runner.update_full_config(full_results_config)
        
        shared_runner.run_experiment()