import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def extract_labeling_budget_and_execution(dir_name):
    """Extract labeling budget and execution number from directory name"""
    # Pattern: output_DATASET_EXEC_BUDGET (e.g., output_QRM_1_30, output_QRM_2_30)
    match = re.search(r"output_(.+)_(\d+)_(\d+)$", dir_name)
    if match:
        dataset, execution, budget = match.groups()
        return int(execution), int(budget)

    # Fallback pattern: output_DATASET_BUDGET (e.g., output_QRM_30)
    match = re.search(r"output_(.+)_(\d+)$", dir_name)
    if match:
        dataset, budget = match.groups()
        return 1, int(budget)  # Default execution = 1

    return None, None


def read_results_file(results_path):
    """Read and parse the results JSON file"""
    try:
        with open(results_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading {results_path}: {e}")
        return None


def read_baseline_results(baseline_file_path):
    """Read baseline results from CSV file"""
    try:
        baseline_df = pd.read_csv(baseline_file_path)
        # Rename columns to match our format
        if "f_score" in baseline_df.columns:
            baseline_df = baseline_df.rename(columns={"f_score": "f1_score"})
        if "execution_time" in baseline_df.columns:
            baseline_df = baseline_df.rename(columns={"execution_time": "total_time"})
        return baseline_df
    except FileNotFoundError:
        print(f"Baseline file not found: {baseline_file_path}")
        return None
    except Exception as e:
        print(f"Error reading baseline file: {e}")
        return None


def collect_all_results(base_directory, dataset_name):
    """Collect results from all dataset output directories and average across executions"""
    execution_results = {}  # {budget: [list of results for each execution]}
    base_path = Path(base_directory)

    if not base_path.exists():
        print(f"   ❌ Directory {base_path} does not exist")
        return []

    # Find all dataset output directories
    pattern = f"output_{dataset_name}_"
    all_dirs = [d.name for d in base_path.iterdir() if d.is_dir()]
    dataset_dirs = [
        d for d in base_path.iterdir() if d.is_dir() and d.name.startswith(pattern)
    ]

    print(f"   🔍 Looking for pattern: {pattern}*")
    print(f"   📁 All directories found: {all_dirs}")
    print(f"   ✅ Matching directories: {[d.name for d in dataset_dirs]}")

    if not dataset_dirs:
        print(f"   ❌ No directories found matching pattern {pattern}*")
        return []

    for dataset_dir in dataset_dirs:
        execution, labeling_budget = extract_labeling_budget_and_execution(
            dataset_dir.name
        )

        if execution is None or labeling_budget is None:
            print(f"Could not parse directory name: {dataset_dir.name}")
            continue

        # Look for results file directly in the dataset directory
        results_file = dataset_dir / "test_ec_evaluation_results.json"

        if results_file.exists():
            data = read_results_file(results_file)
            if data:
                # Extract key metrics
                pipeline_metrics = data.get("pipeline_metrics", {})
                result_entry = {
                    "execution": execution,
                    "labeling_budget": labeling_budget,
                    "precision": pipeline_metrics.get("overall_precision", 0),
                    "recall": pipeline_metrics.get("overall_recall", 0),
                    "f1_score": pipeline_metrics.get("overall_f1", 0),
                    "total_error_cells": pipeline_metrics.get("total_error_cells", 0),
                    "total_corrections": pipeline_metrics.get("total_corrections", 0),
                    "total_correct_corrections": pipeline_metrics.get(
                        "total_correct_corrections", 0
                    ),
                    "total_candidates_generated": pipeline_metrics.get(
                        "total_candidates_generated", 0
                    ),
                    "candidate_generation_time": pipeline_metrics.get(
                        "candidate_generation_time", 0
                    ),
                    "label_propagation_time": pipeline_metrics.get(
                        "label_propagation_time", 0
                    ),
                    "training_and_prediction": pipeline_metrics.get(
                        "training_and_prediction", 0
                    ),
                    "total_time": pipeline_metrics.get("total_time", 0),
                }

                # Group results by labeling budget
                if labeling_budget not in execution_results:
                    execution_results[labeling_budget] = []
                execution_results[labeling_budget].append(result_entry)

                print(
                    f"Loaded {dataset_name} execution {execution}, budget {labeling_budget}: "
                    f"P={result_entry['precision']:.3f}, R={result_entry['recall']:.3f}, F1={result_entry['f1_score']:.3f}"
                )
        else:
            print(f"Results file not found: {results_file}")

    if not execution_results:
        print(f"No valid results found for dataset {dataset_name}")
        return []

    # Average results across executions for each labeling budget
    averaged_results = []
    for budget, budget_results in execution_results.items():
        num_executions = len(budget_results)

        # Calculate averages and standard deviations
        avg_result = {
            "labeling_budget": budget,
            "num_executions": num_executions,
            "precision": np.mean([r["precision"] for r in budget_results]),
            "precision_std": np.std([r["precision"] for r in budget_results])
            if num_executions > 1
            else 0,
            "recall": np.mean([r["recall"] for r in budget_results]),
            "recall_std": np.std([r["recall"] for r in budget_results])
            if num_executions > 1
            else 0,
            "f1_score": np.mean([r["f1_score"] for r in budget_results]),
            "f1_score_std": np.std([r["f1_score"] for r in budget_results])
            if num_executions > 1
            else 0,
            "total_error_cells": int(
                np.mean([r["total_error_cells"] for r in budget_results])
            ),
            "total_corrections": np.mean(
                [r["total_corrections"] for r in budget_results]
            ),
            "total_correct_corrections": np.mean(
                [r["total_correct_corrections"] for r in budget_results]
            ),
            "total_candidates_generated": np.mean(
                [r["total_candidates_generated"] for r in budget_results]
            ),
            "candidate_generation_time": np.mean(
                [r["candidate_generation_time"] for r in budget_results]
            ),
            "label_propagation_time": np.mean(
                [r["label_propagation_time"] for r in budget_results]
            ),
            "training_and_prediction": np.mean(
                [r["training_and_prediction"] for r in budget_results]
            ),
            "total_time": np.mean([r["total_time"] for r in budget_results]),
            "total_time_std": np.std([r["total_time"] for r in budget_results])
            if num_executions > 1
            else 0,
        }

        averaged_results.append(avg_result)

        if num_executions > 1:
            print(
                f"Averaged {dataset_name} over {num_executions} executions for budget {budget}: "
                f"P={avg_result['precision']:.3f}±{avg_result['precision_std']:.3f}, "
                f"R={avg_result['recall']:.3f}±{avg_result['recall_std']:.3f}, "
                f"F1={avg_result['f1_score']:.3f}±{avg_result['f1_score_std']:.3f}"
            )

    return sorted(averaged_results, key=lambda x: x["labeling_budget"])


def create_plots(
    our_data, baseline_data=None, old_data=None, dataset_name="Dataset", output_path="."
):
    """Create comparison plots for all methods"""

    # Set the style for better looking plots
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 16,
        }
    )

    # Convert to DataFrames
    our_df = pd.DataFrame(our_data)

    # Individual Precision, Recall, F1, and Execution Time plots
    metrics = ["precision", "recall", "f1_score", "total_time"]
    metric_names = ["Precision", "Recall", "F1-Score", "Execution Time (s)"]

    # Colors and markers for different methods
    our_color = "#2E8B57"  # Sea Green
    old_color = "#DC143C"  # Crimson Red
    baseline_color = "#4169E1"  # Royal Blue

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Dynamic title
    title_parts = ["Our Method"]
    if old_data is not None and len(old_data) > 0:
        title_parts.append("Old Version")
    if baseline_data is not None and len(baseline_data) > 0:
        title_parts.append("Baran Baseline")

    fig.suptitle(
        f"{dataset_name} Dataset: {' vs '.join(title_parts)}",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    # Collect all budget values for consistent x-axis
    all_budgets = sorted(list(our_df["labeling_budget"]))
    if baseline_data is not None and len(baseline_data) > 0:
        all_budgets = sorted(
            list(set(all_budgets + list(baseline_data["labeling_budget"])))
        )
    if old_data is not None and len(old_data) > 0:
        old_df = pd.DataFrame(old_data)
        all_budgets = sorted(list(set(all_budgets + list(old_df["labeling_budget"]))))

    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i]

        # Plot our method (green circles)
        ax.plot(
            our_df["labeling_budget"],
            our_df[metric],
            "o-",
            linewidth=3,
            markersize=8,
            color=our_color,
            markerfacecolor=our_color,
            markeredgecolor="white",
            markeredgewidth=1.5,
            alpha=0.9,
            label="Our Method",
        )

        # Add confidence bands for our method
        if our_df["num_executions"].iloc[0] > 1:
            std_col = f"{metric}_std" if metric != "total_time" else "total_time_std"
            if std_col in our_df.columns:
                ax.fill_between(
                    our_df["labeling_budget"],
                    our_df[metric] - our_df[std_col],
                    our_df[metric] + our_df[std_col],
                    alpha=0.2,
                    color=our_color,
                )

        # Plot old version (red triangles)
        if old_data is not None and len(old_data) > 0:
            old_df = pd.DataFrame(old_data)
            if metric in old_df.columns:
                ax.plot(
                    old_df["labeling_budget"],
                    old_df[metric],
                    "^-",
                    linewidth=3,
                    markersize=8,
                    color=old_color,
                    markerfacecolor=old_color,
                    markeredgecolor="white",
                    markeredgewidth=1.5,
                    alpha=0.9,
                    label="Old Version",
                )

                # Add confidence bands for old version
                if (
                    "num_executions" in old_df.columns
                    and old_df["num_executions"].iloc[0] > 1
                ):
                    std_col = (
                        f"{metric}_std" if metric != "total_time" else "total_time_std"
                    )
                    if std_col in old_df.columns:
                        ax.fill_between(
                            old_df["labeling_budget"],
                            old_df[metric] - old_df[std_col],
                            old_df[metric] + old_df[std_col],
                            alpha=0.2,
                            color=old_color,
                        )

        # Plot baseline (blue squares)
        if baseline_data is not None and len(baseline_data) > 0:
            if metric in baseline_data.columns:
                ax.plot(
                    baseline_data["labeling_budget"],
                    baseline_data[metric],
                    "s-",
                    linewidth=3,
                    markersize=8,
                    color=baseline_color,
                    markerfacecolor=baseline_color,
                    markeredgecolor="white",
                    markeredgewidth=1.5,
                    alpha=0.9,
                    label="Baran Baseline",
                )

                # Add confidence bands for baseline if available
                baseline_std_col = f"{metric}_std"
                if baseline_std_col in baseline_data.columns:
                    ax.fill_between(
                        baseline_data["labeling_budget"],
                        baseline_data[metric] - baseline_data[baseline_std_col],
                        baseline_data[metric] + baseline_data[baseline_std_col],
                        alpha=0.2,
                        color=baseline_color,
                    )

        # Customize plot
        ax.set_xlabel("Labeling Budget", fontweight="bold")
        ax.set_ylabel(name, fontweight="bold")
        ax.set_title(f"{name} vs Labeling Budget", fontweight="bold", pad=15)
        ax.grid(True, alpha=0.3, linestyle="--")

        # Set y-limits
        if metric == "total_time":
            y_max = our_df[metric].max()
            if old_data is not None and len(old_data) > 0:
                y_max = max(y_max, pd.DataFrame(old_data)[metric].max())
            if baseline_data is not None and len(baseline_data) > 0:
                y_max = max(y_max, baseline_data[metric].max())
            ax.set_ylim(0, y_max * 1.1)
        else:
            ax.set_ylim(0, 1)

        # Set x-ticks
        ax.set_xticks(all_budgets)
        ax.set_xticklabels(all_budgets, rotation=45 if len(all_budgets) > 6 else 0)

        # Add annotations for our method
        for j, (budget, value) in enumerate(
            zip(our_df["labeling_budget"], our_df[metric])
        ):
            if metric == "total_time":
                text = f"{value:.1f}s"
            else:
                text = f"{value:.3f}"

            ax.annotate(
                text,
                (budget, value),
                textcoords="offset points",
                xytext=(0, 15),
                ha="center",
                fontweight="bold",
                fontsize=8,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    edgecolor=our_color,
                    alpha=0.8,
                ),
            )

        # Style improvements
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1)
        ax.spines["bottom"].set_linewidth(1)
        ax.set_facecolor("#fafafa")

        # Add legend to first plot
        if i == 0:
            ax.legend(loc="best", frameon=True, shadow=True, fancybox=True)

    plt.tight_layout()

    # Save plot
    output_file = Path(output_path) / f"{dataset_name.lower()}_comparison_metrics.png"
    plt.savefig(
        output_file, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    plt.show()

    return str(output_file)


def print_summary_stats(
    our_data, baseline_data=None, old_data=None, dataset_name="Dataset"
):
    """Print comprehensive summary statistics"""

    our_df = pd.DataFrame(our_data)

    print("\n" + "=" * 70)
    print(f"{dataset_name.upper()} DATASET SUMMARY STATISTICS")
    print("=" * 70)

    # Our Method Statistics
    print("\n📊 OUR METHOD:")
    print(
        f"   Labeling Budget Range: {our_df['labeling_budget'].min()} - {our_df['labeling_budget'].max()}"
    )
    print(f"   Number of Budget Points: {len(our_df)}")
    if "num_executions" in our_df.columns:
        total_experiments = our_df["num_executions"].sum()
        print(f"   Executions per Budget: {our_df['num_executions'].iloc[0]}")
        print(f"   Total Experiments: {total_experiments}")

    # Old Version Statistics
    if old_data is not None and len(old_data) > 0:
        old_df = pd.DataFrame(old_data)
        print("\n🔄 OLD VERSION:")
        print(
            f"   Labeling Budget Range: {old_df['labeling_budget'].min()} - {old_df['labeling_budget'].max()}"
        )
        print(f"   Number of Budget Points: {len(old_df)}")
        if "num_executions" in old_df.columns:
            print(f"   Executions per Budget: {old_df['num_executions'].iloc[0]}")
            print(f"   Total Experiments: {old_df['num_executions'].sum()}")

    # Baseline Statistics
    if baseline_data is not None and len(baseline_data) > 0:
        print("\n🎯 BARAN BASELINE:")
        print(
            f"   Labeling Budget Range: {baseline_data['labeling_budget'].min()} - {baseline_data['labeling_budget'].max()}"
        )
        print(f"   Number of Experiments: {len(baseline_data)}")

    # Performance Comparison
    print("\n🏆 BEST PERFORMANCE COMPARISON:")

    # Our method best performance
    our_best_f1 = our_df.loc[our_df["f1_score"].idxmax()]
    print("\n   Our Method Best:")
    print(
        f"     F1-Score: {our_best_f1['f1_score']:.4f} (Budget: {our_best_f1['labeling_budget']})"
    )
    print(
        f"     Precision: {our_df.loc[our_df['precision'].idxmax(), 'precision']:.4f}"
    )
    print(f"     Recall: {our_df.loc[our_df['recall'].idxmax(), 'recall']:.4f}")

    # Old version best performance
    if old_data is not None and len(old_data) > 0:
        old_df = pd.DataFrame(old_data)
        old_best_f1 = old_df.loc[old_df["f1_score"].idxmax()]
        print("\n   Old Version Best:")
        print(
            f"     F1-Score: {old_best_f1['f1_score']:.4f} (Budget: {old_best_f1['labeling_budget']})"
        )
        print(
            f"     Precision: {old_df.loc[old_df['precision'].idxmax(), 'precision']:.4f}"
        )
        print(f"     Recall: {old_df.loc[old_df['recall'].idxmax(), 'recall']:.4f}")

    # Baseline best performance
    if baseline_data is not None and len(baseline_data) > 0:
        baseline_best_f1 = baseline_data.loc[baseline_data["f1_score"].idxmax()]
        print("\n   Baran Baseline Best:")
        print(
            f"     F1-Score: {baseline_best_f1['f1_score']:.4f} (Budget: {baseline_best_f1['labeling_budget']})"
        )
        print(
            f"     Precision: {baseline_data.loc[baseline_data['precision'].idxmax(), 'precision']:.4f}"
        )
        print(
            f"     Recall: {baseline_data.loc[baseline_data['recall'].idxmax(), 'recall']:.4f}"
        )

    # Efficiency Analysis
    print("\n⚡ EFFICIENCY ANALYSIS (F1-Score / Time):")
    our_df["efficiency"] = our_df["f1_score"] / our_df["total_time"]
    our_best_eff = our_df.loc[our_df["efficiency"].idxmax()]
    print("\n   Our Method Most Efficient:")
    print(
        f"     Efficiency: {our_best_eff['efficiency']:.4f} (F1={our_best_eff['f1_score']:.3f}, Time={our_best_eff['total_time']:.1f}s)"
    )
    print(f"     Budget: {our_best_eff['labeling_budget']}")

    if old_data is not None and len(old_data) > 0:
        old_df = pd.DataFrame(old_data)
        old_df["efficiency"] = old_df["f1_score"] / old_df["total_time"]
        old_best_eff = old_df.loc[old_df["efficiency"].idxmax()]
        print("\n   Old Version Most Efficient:")
        print(
            f"     Efficiency: {old_best_eff['efficiency']:.4f} (F1={old_best_eff['f1_score']:.3f}, Time={old_best_eff['total_time']:.1f}s)"
        )
        print(f"     Budget: {old_best_eff['labeling_budget']}")


def save_results_to_csv(our_data, dataset_name, output_path):
    """Save results to CSV with proper column organization"""

    df = pd.DataFrame(our_data)

    # Define column order
    csv_columns = [
        "labeling_budget",
        "num_executions",
        "precision",
        "precision_std",
        "recall",
        "recall_std",
        "f1_score",
        "f1_score_std",
        "total_time",
        "total_time_std",
        "total_error_cells",
        "total_corrections",
        "total_correct_corrections",
        "total_candidates_generated",
        "candidate_generation_time",
        "label_propagation_time",
        "training_and_prediction",
    ]

    # Only include columns that exist
    available_columns = [col for col in csv_columns if col in df.columns]
    csv_df = df[available_columns]

    # Save to CSV
    csv_file = Path(output_path) / f"{dataset_name.lower()}_results_summary.csv"
    csv_df.to_csv(csv_file, index=False)

    return str(csv_file)


def main():
    """Main function to run the analysis"""

    # ======================== CONFIGURATION ========================
    # Modify these paths and settings according to your setup

    INPUT_PATH = "/home/fatemeh/ECS-Dev-Local/EC-at-Scale/exp-0406-brute_force"  # Path to current method results
    DATASET_NAME = "NTR_RS"  # Dataset name (QRM, HOSPITAL, etc.)
    BASELINE_PATH = "/home/fatemeh/ECS-1iter/EC-at-Scale/exp-0106/results/NTR_RS/baran.csv"  # Path to baseline CSV (or None)
    OLD_PATH = "/home/fatemeh/ECS-1iter/EC-at-Scale/exp-0306"  # Path to old version results (or None)
    OUTPUT_PATH = "/home/fatemeh/ECS-Dev-Local/EC-at-Scale/exp-0406-brute_force"  # Output directory

    # ================================================================

    print(f"\n🔬 {DATASET_NAME} Dataset Results Analysis")
    print("=" * 60)
    print(f"📁 Current Method Path: {INPUT_PATH}")
    print(f"📁 Old Version Path: {OLD_PATH if OLD_PATH else 'None'}")
    print(f"📁 Baseline Path: {BASELINE_PATH if BASELINE_PATH else 'None'}")
    print(f"📁 Output Path: {OUTPUT_PATH}")
    print(f"🗂️  Dataset: {DATASET_NAME}")
    print("=" * 60)

    # Create output directory
    output_path = Path(OUTPUT_PATH)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect our method results
    print("\n📊 Loading current method results...")
    our_results = collect_all_results(INPUT_PATH, DATASET_NAME)

    if not our_results:
        print("❌ No results found for current method!")
        print(f"   Check that directory exists: {INPUT_PATH}")
        print(f"   Check directory pattern: output_{DATASET_NAME}_X_Y")
        print("   Check results files exist: test_ec_evaluation_results.json")
        return

    print(f"✅ Loaded {len(our_results)} budget points for current method")

    # Collect old version results
    old_results = None
    if OLD_PATH:
        print("\n📊 Loading old version results...")
        # Use the same dataset name pattern for old version
        old_results = collect_all_results(OLD_PATH, DATASET_NAME)

        if old_results:
            print(f"✅ Loaded {len(old_results)} budget points for old version")
        else:
            print(f"⚠️  No old version results found at {OLD_PATH}")
            print(f"   Looking for pattern: output_{DATASET_NAME}_*")

    # Load baseline results
    baseline_results = None
    if BASELINE_PATH:
        print("\n📊 Loading baseline results...")
        baseline_results = read_baseline_results(BASELINE_PATH)
        if baseline_results is not None:
            print(f"✅ Loaded {len(baseline_results)} experiments for baseline")
        else:
            print(f"⚠️  Failed to load baseline from {BASELINE_PATH}")

    # Create visualizations
    print("\n🎨 Creating visualizations...")
    plot_file = create_plots(
        our_results, baseline_results, old_results, DATASET_NAME, output_path
    )

    # Save results to CSV
    print("\n💾 Saving results...")
    csv_file = save_results_to_csv(our_results, DATASET_NAME, output_path)

    # Print summary statistics
    print_summary_stats(our_results, baseline_results, old_results, DATASET_NAME)

    # Final summary
    print("\n✅ ANALYSIS COMPLETE!")
    print(f"📊 Plot saved: {plot_file}")
    print(f"📄 CSV saved: {csv_file}")
    print(f"🎯 All files saved to: {output_path}")


if __name__ == "__main__":
    main()
