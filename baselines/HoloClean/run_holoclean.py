
import os
import pickle
import time

import holoclean
import pandas as pd
from holoclean.detect.groundtruthdetector import GroundTruthDetector
from holoclean.repair.featurize.constraintfeat import ConstraintFeaturizer
from holoclean.repair.featurize.freqfeat import FreqFeaturizer
from holoclean.repair.featurize.initattrfeat import InitAttrFeaturizer
from holoclean.repair.featurize.occurattrfeat import OccurAttrFeaturizer
from utils import (evaluate, evaluate_by_error_type, get_cleaner_directory,
                   store_cleaned_data)


def dcHoloCleaner(dataset_name, dirty_df, clean_df, dataset_path, detections, method, seed=45, iteration=0, threads=1):
    """
    This method repairs errors detected with denial constraints,

    Arguments:
    detections -- dictionary - keys represent i,j of dirty cells & values are constant string "JUST A DUUMY VALUE"
    dirtyDF -- dataframe of shape n_R (# of records) x n_A (# of attributes) - containing a dirty version of a dataset
    with_init -- boolean - if true InitAttributeFeaturizer is used as feature
    path_to_constraints -- String - Path to the txt-file, containing the constraints with which the errors where detected
    seed -- int - random seed for reproducibility
    iteration -- int - iteration number for file naming
    threads -- int - number of threads for HoloClean to use

    Returns:
    repairedDF -- dataframe of shape n_R x n_A - containing a cleaned version of the dirty dataset 
    results (dict) -- dictionary with results from evaluation
    """

    if len(detections) == 0:
        return dirty_df, {"Problem": "No Errors to be cleaned"}

    # Extract the necessary parameters
    #with_init = configs["with_init"]
    with_init = True if method == "with_init" else False
    path_to_constraints = os.path.abspath(dataset_path)
    #path_to_constraints = configs["path_to_constraints"]

    # 1. Setup a HoloClean session.
    hc = holoclean.HoloClean(
        db_name="holo",
        domain_thresh_1=0,
        domain_thresh_2=0,
        weak_label_thresh=0.90,
        max_domain=10000,
        cor_strength=0.6,
        nb_cor_strength=0.8,
        epochs=10,
        weight_decay=0.01,
        learning_rate=0.001,
        threads=threads,
        batch_size=1,
        seed=seed,
        verbose=True,
        timeout=360000,
        feature_norm=False,
        weight_norm=False,
        print_fw=True,
    ).session

    # load the dirty data in holoclean
    hc.load_data(dataset_name, fpath=dataset_path, df=dirty_df)

    # load the constraints from dataset constraint directory
    hc.load_dcs(os.path.join(path_to_constraints, "holo_constraints.txt"))

    # set the constraints in holoclean
    hc.ds.set_constraints(hc.get_dcs())

    detectors = [GroundTruthDetector(dirty_df=dirty_df, gt_df=clean_df)]
    hc.detect_errors(detectors)

    # 4. Repair errors utilizing the defined features.
    hc.setup_domain()
    if with_init == True:
        featurizers = [
            InitAttrFeaturizer(),
            OccurAttrFeaturizer(),
            FreqFeaturizer(),
            ConstraintFeaturizer(),
        ]
    else:
        featurizers = [
            OccurAttrFeaturizer(),
            FreqFeaturizer(),
            ConstraintFeaturizer(),
        ]
    
    start_time = time.time()
    # repair errors and get repaired dataframe
    _, repaired_df = hc.repair_errors(featurizers)

    end_time = time.time()
    # repaired_df = repaired_df.drop("_tid_", 1)

    # if dataset contained index column, insert it again into df
    # if index_col_pos != -1:
    #     cleanedDF.insert(index_col_pos, "index", index_col_values)

    cleaner_directory = get_cleaner_directory(
        "dcHoloCleaner-{specification}".format(
            specification="with_init" if with_init == True else "without_init"), cleaner_name="HoloClean"
        
    )
    store_cleaned_data(
        repaired_df, os.path.join(cleaner_directory, f"repaired_holoclean_{dataset_name}_seed{seed}_iter{iteration}.csv")
    )

    # Use custom evaluate function
    custom_results = evaluate(detections, dirty_df, clean_df, repaired_df)
    custom_results["cleaning_runtime"] = end_time - start_time

    # Check if provenance file exists for error type evaluation
    provenance_file = os.path.join(os.path.dirname(dataset_path), dataset_name, "clean_changes_provenance.csv")
    if not os.path.exists(provenance_file):
        # Try alternative provenance file name
        provenance_file = os.path.join(os.path.dirname(dataset_path), dataset_name, "source_mapping.json")
    
    error_type_results = None
    if os.path.exists(provenance_file):
        error_type_results = evaluate_by_error_type(dirty_df, clean_df, repaired_df, provenance_file)

    # Also get HoloClean's built-in evaluation
    report = hc.evaluate(
        fpath= os.path.join(dataset_path, "all_cells.csv"),
        # os.path.join(os.path.dirname(dataset_path), dataset_name, "clean.csv"),
                    tid_col='tid',
                    attr_col='attribute',
                    val_col='correct_val')
    report_dict = report._asdict()
    
    # Combine both evaluations
    report_dict["cleaning_runtime"] = end_time - start_time
    report_dict["actual_errors_expected"] = len(detections)
    report_dict["seed"] = seed
    report_dict["iteration"] = iteration
    report_dict["custom_evaluation"] = custom_results
    if error_type_results:
        report_dict["error_type_evaluation"] = error_type_results
    
    with open(os.path.join(cleaner_directory, f"results_holoclean_{dataset_name}_seed{seed}_iter{iteration}.pkl"), "wb") as f:
        pickle.dump(report_dict, f)
    return repaired_df, report_dict