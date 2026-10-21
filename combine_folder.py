import shutil
from pathlib import Path

def combine_and_rename(folder1, folder2, output_folder):
    # Convert string paths to Path objects
    path1 = Path(folder1)
    path2 = Path(folder2)
    out_path = Path(output_folder)
    
    # Create the output folder if it doesn't exist
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Gather all .npy files from both folders and sort them
    # Sorting ensures a consistent order when renaming
    files_folder1 = sorted(path1.glob('*.npy'))
    files_folder2 = sorted(path2.glob('*.npy'))
    
    # Combine the lists of files
    all_files = files_folder1 + files_folder2
    
    # Check if the total matches your expectation of 200 files
    total_files = len(all_files)
    if total_files != 200:
        print(f"Warning: Found {total_files} .npy files, but expected 200 based on 'calib_0199'.")
    
    # Iterate through the combined list and copy/rename them
    for index, file_path in enumerate(all_files):
        # Format the index to have 4 digits, padding with zeros (e.g., 0000, 0001)
        new_filename = f"calib_{index:04d}.npy"
        destination_path = out_path / new_filename
        
        # shutil.copy2 copies the file and attempts to preserve metadata (like creation time)
        shutil.copy2(file_path, destination_path)
        print(f"Copied: {file_path.name} -> {new_filename}")
        
    print(f"\nSuccessfully combined {total_files} files into '{output_folder}'.")

# --- Example Usage ---
# Replace these strings with the actual paths on your system
folder1_path = 'D:\ZCU104_deploy - Copy\complex_yolov4\calibration_data'
folder2_path = 'D:\ZCU104_deploy - Copy\complex_yolov4\calibration_data_1'
output_path = 'D:\ZCU104_deploy - Copy\complex_yolov4\calibration_data_200'

combine_and_rename(folder1_path, folder2_path, output_path)