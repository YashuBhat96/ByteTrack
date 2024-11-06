import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    filename='/storage/home/ybr5070/group/homebytes/code/scripts/logs/file_matching.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'  # Overwrite log file on each run
)

def process_single_excel_file(excel_file, image_folder, split_output_dir):
    # Dummy function to simulate processing
    logging.info(f"Processing Excel file: {excel_file} with image folder: {image_folder}")
    # Add actual processing code here

# Function to filter and update Excel files based on image folders
def filter_and_update_excel_files(split, base_excel_dir, base_image_dir, base_output_dir='/output_dir', max_workers=4):
    excel_split_dir = os.path.join(base_excel_dir, split)
    image_split_dir = os.path.join(base_image_dir, split)

    logging.info(f"Starting to process {split} split")

    # Lists to store Excel files and image folders
    excel_files = []
    image_folders = []

    # Find all Excel files in the split folder
    logging.info(f"Searching for Excel files in {excel_split_dir}")
    for root, _, files in os.walk(excel_split_dir):
        for file in files:
            if file.endswith('.xlsx'):
                excel_files.append(os.path.join(root, file))
    logging.info(f"Found {len(excel_files)} Excel files in {excel_split_dir}")

    # Find all image folders in the corresponding split folder
    logging.info(f"Searching for image folders in {image_split_dir}")
    for root_img, dirs, _ in os.walk(image_split_dir):
        for dir_name in dirs:
            if dir_name.endswith('_fixed'):
                image_folders.append(os.path.join(root_img, dir_name))
    logging.info(f"Found {len(image_folders)} image folders in {image_split_dir}")

    # Track matched and unmatched files
    matched = []
    unmatched_excel = []
    unmatched_image = []

    tasks = []

    # Match Excel files with corresponding image folders
    logging.info(f"Matching Excel files with image folders for {split} split")
    for excel_file in excel_files:
        subject_info = os.path.basename(excel_file).replace('.xlsx', '')
        match_found = False
        for image_folder in image_folders:
            if subject_info in os.path.basename(image_folder):
                matched.append((excel_file, image_folder))
                match_found = True
                # Ensure the subfolder for the current split exists within the output directory
                split_output_dir = os.path.join(base_output_dir, split)
                tasks.append((excel_file, image_folder, split_output_dir))
                logging.info(f"Matched Excel file: {excel_file} with image folder: {image_folder}")
                break
        if not match_found:
            unmatched_excel.append(excel_file)
            logging.warning(f"Unmatched Excel file: {excel_file}")

    # Find unmatched image folders
    for image_folder in image_folders:
        folder_name = os.path.basename(image_folder).replace('_fixed', '')
        if not any(folder_name in os.path.basename(excel_file) for excel_file in excel_files):
            unmatched_image.append(image_folder)
            logging.warning(f"Unmatched image folder: {image_folder}")

    # Log summary of matches and mismatches
    logging.info(f"Matched {len(matched)} files for {split} split.")
    logging.info(f"Unmatched Excel files: {len(unmatched_excel)}")
    logging.info(f"Unmatched image folders: {len(unmatched_image)}")

    # Process Excel files concurrently using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_excel_file, *task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing Excel Files ({split})"):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Task failed with exception: {e}")

    logging.info(f"All Excel files processed and updated successfully for the {split} split.")


# Specify the base paths
base_excel_dir = '/storage/group/klk37/default/homebytes/video/fbs/labels'
base_image_dir = '/storage/group/klk37/default/homebytes/video/fbs/FRCNN+yolo'

# Run the function for the train, test, and val folders
filter_and_update_excel_files('test', base_excel_dir, base_image_dir)
filter_and_update_excel_files('train', base_excel_dir, base_image_dir)
filter_and_update_excel_files('val', base_excel_dir, base_image_dir)
