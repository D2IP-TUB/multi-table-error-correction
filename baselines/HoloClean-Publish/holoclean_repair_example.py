import sys

import pandas as pd
# sys.path.append('../')
from holoclean import holoclean
from holoclean.detect.nulldetector import NullDetector
from holoclean.detect.violationdetector import ViolationDetector
from holoclean.repair.featurize import *


# 1. Setup a HoloClean session.
hc = holoclean.HoloClean(
    db_name='holo',
    domain_thresh_1=0,
    domain_thresh_2=0,
    weak_label_thresh=0.99,
    max_domain=10000,
    cor_strength=0.6,
    nb_cor_strength=0.8,
    epochs=10,
    weight_decay=0.01,
    learning_rate=0.001,
    threads=1,
    batch_size=1,
    verbose=True,
    timeout=3*60000,
    feature_norm=False,
    weight_norm=False,
    print_fw=True
).session

# 2. Load training data and denial constraints.
hc.load_data('hospital', '/home/fatemeh/LakeCorrectionBench/datasets/Quintet/hospital/dirty.csv')
hc.load_dcs('/home/fatemeh/LakeCorrectionBench/holoclean/testdata/hospital_constraints.txt')
hc.ds.set_constraints(hc.get_dcs())

# # 3. Detect erroneous cells using these two detectors.
# detectors = [NullDetector(), ViolationDetector()]
# hc.detect_errors(detectors)
holoclean_error_df = pd.read_csv("/home/fatemeh/LakeCorrectionBench/datasets/Quintet/hospital/corrected_cells.csv")
holoclean_error_df.drop_duplicates()
holoclean_error_df["_cid_"] = holoclean_error_df.apply(
    lambda x: hc.ds.get_cell_id(x["_tid_"], x["attribute"]), axis=1
)
hc.detect_engine.store_detected_errors(holoclean_error_df)

# 4. Repair errors utilizing the defined features.
hc.setup_domain()
featurizers = [
    InitAttrFeaturizer(),
    OccurAttrFeaturizer(),
    FreqFeaturizer(),
    ConstraintFeaturizer(),
]

hc.repair_errors(featurizers)

# 5. Evaluate the correctness of the results.
hc.evaluate(fpath='/home/fatemeh/LakeCorrectionBench/datasets/Quintet/hospital/corrected_cells.csv',
            tid_col='tid',
            attr_col='attribute',
            val_col='correct_val')
