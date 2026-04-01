import os
import pickle

results_path = (
    "/home/fatemeh/ECS-1iter/EC-at-Scale/exp-0907/DGov/output_DGov_NTR_pm_rm_1_683"
)

with open(os.path.join(results_path, "table_id_to_names.pickle"), "rb") as file:
    table_id_to_names = pickle.load(file)

with open(
    os.path.join(results_path, "correction_dict_all_zones.pickle"),
    "rb",
) as file:
    correction_dict_all_zones = pickle.load(file)

with open(os.path.join(results_path, "selected_cells_scores.pickle"), "rb") as file:
    selected_cells_scores = pickle.load(file)

results_per_table = {}

for table_id, table in table_id_to_names.items():
    results_per_table[table_id] = {}
    results_per_table[table_id]["table_name"] = table["table_name"]
    results_per_table[table_id]["columns"] = {
        column["column_index"]: {"column_name": column["column_name"]}
        for column in table["columns"]
    }

for zone in correction_dict_all_zones:
    for cell in correction_dict_all_zones[zone]:
        table_id, column_index, row_index = cell
        if zone not in results_per_table[table_id]["columns"][column_index]:
            results_per_table[table_id]["columns"][column_index][zone] = {
                "tp_corrected_cells": [],
                "fp_corrected_cells": [],
                "not_corrected_cells": [],
                "selected_cells": [],
            }

        if len(correction_dict_all_zones[zone][cell]["predicted_corrections"]) == 0:
            results_per_table[table_id]["columns"][column_index][zone][
                "not_corrected_cells"
            ].append(correction_dict_all_zones[zone][cell])
            continue
        if (
            correction_dict_all_zones[zone][cell]["predicted_corrections"][0][
                "candidate"
            ]
            == correction_dict_all_zones[zone][cell]["ground_truth"]
        ):
            results_per_table[table_id]["columns"][column_index][zone][
                "tp_corrected_cells"
            ].append(correction_dict_all_zones[zone][cell])
        else:
            results_per_table[table_id]["columns"][column_index][zone][
                "fp_corrected_cells"
            ].append(correction_dict_all_zones[zone][cell])

with open(
    os.path.join(results_path, "results_per_table.pickle"),
    "wb",
) as file:
    pickle.dump(results_per_table, file)
