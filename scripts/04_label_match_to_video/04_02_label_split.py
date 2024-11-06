import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# Define directories
roi_face_dir = '/storage/group/klk37/default/homebytes/video/fbs/FRCNN+yolo'
label_split_dir = '/storage/group/klk37/default/homebytes/video/fbs/labels'

# Ensure the label_split_dir directory exists
os.makedirs(label_split_dir, exist_ok=True)

# Define the train, val, test directories within ROI_face
splits = ['train', 'val', 'test']
split_dirs = {split: os.path.join(roi_face_dir, split) for split in splits}

def process_split(split):
    # Get the subfolder names for the current split
    split_subfolders = set(os.listdir(split_dirs[split]))
    
    # List all Excel files in the label_split_dir
    excel_files = [os.path.join(root, file) for root, _, files in os.walk(label_split_dir) for file in files if file.endswith('.xlsx')]
    
    for file_path in excel_files:
        # Get the Excel file name without the extension
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Check if there's a matching folder with _fixed in the current split
        matching_folder = f"{file_name}_fixed"
        
        if matching_folder in split_subfolders:
            # Load the Excel file
            df = pd.read_excel(file_path)
            
            # Save the file in the corresponding split directory
            split_output_dir = os.path.join(label_split_dir, split)
            os.makedirs(split_output_dir, exist_ok=True)
            output_filename = os.path.join(split_output_dir, os.path.basename(file_path))
            
            # Save the DataFrame as an Excel file in the split directory
            df.to_excel(output_filename, index=False)
            print(f"Saved {file_name} to {output_filename} under {split} split.")

def main():
    # Use ThreadPoolExecutor to process each split in parallel
    with ThreadPoolExecutor() as executor:
        executor.map(process_split, splits)

    print("All files processed and saved.")

if __name__ == '__main__':
    main()
