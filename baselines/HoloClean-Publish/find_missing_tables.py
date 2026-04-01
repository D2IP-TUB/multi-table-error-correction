#!/usr/bin/env python3
"""
Reads open_data_uk_name_map.pkl and prints the original dataset names
that are NOT present in datasets/open_data_uk_93.
"""
import os
import pickle

PICKLE_PATH = "/home/fatemeh/LakeCorrectionBench/HoloClean/datasets-holo/open_data_uk_name_map.pkl"
DATASETS_DIR = "/home/fatemeh/LakeCorrectionBench/datasets/open_data_uk_93"

with open(PICKLE_PATH, "rb") as f:
    name_map = pickle.load(f)

# The keys are full paths; the basename is the original dataset name (UK_CSV...)
in_lake = set(os.listdir(DATASETS_DIR))

missing = []
for src_path in name_map:
    name = os.path.basename(src_path)
    if name not in in_lake:
        missing.append(name)

missing.sort()
print(f"{len(missing)} table(s) from the name map are NOT in {DATASETS_DIR}:\n")
for name in missing:
    print(f"  {name}")
