import os
import shutil
import pickle 

def rename_and_copy_dirs(input_path, output_path):
    # Ensure output_path exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Get a list of directories in the input path
    dirs = [d for d in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, d))]
    res_dict = {}
    for index, dir_name in enumerate(dirs):
        # Create new directory name
        new_dir_name = f"table_{index + 1}"
        
        # Define full paths
        original_dir_path = os.path.join(input_path, dir_name)
        new_dir_path = os.path.join(output_path, new_dir_name)
        res_dict[original_dir_path] = new_dir_path
        # Copy the directory to the new location with the new name
        shutil.copytree(original_dir_path, new_dir_path)
        print(f"Copied {original_dir_path} to {new_dir_path}")
    return res_dict
# Example usage
input_path = "/home/fatemeh/LakeCorrectionBench/datasets/open_data_uk_93"
output_path = "/home/fatemeh/LakeCorrectionBench/HoloClean/datasets-holo/open_data_uk_93"
res_dict = rename_and_copy_dirs(input_path, output_path)
with open("open_data_93_res_dict_no.pkl", "wb") as f:
    pickle.dump(res_dict, f)