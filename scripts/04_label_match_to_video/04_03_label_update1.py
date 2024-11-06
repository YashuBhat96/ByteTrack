import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(filename='/storage/home/ybr5070/group/homebytes/code/scripts/logs/lab_upd1_new_misc.log', 
                    level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

def round_up_timestamps(df):
    logging.debug("Rounding up timestamps.")
    
    # Rename 'Time_Relative_sf' to 'Time_point_s' if 'Time_point_s' does not exist
    if 'Time_point_s' not in df.columns and 'Time_Relative_sf' in df.columns:
        df.rename(columns={'Time_Relative_sf': 'Time_point_s'}, inplace=True)
        logging.info("Renamed 'Time_Relative_sf' to 'Time_point_s'")
    
    if 'Time_point_s' not in df.columns:
        logging.error("Neither 'Time_point_s' nor 'Time_Relative_sf' found in the columns.")
        return df  # Return the original DataFrame if the column isn't found
    
    # Apply rounding to the 'Time_point_s' column
    df['Time_point_s'] = df['Time_point_s'].apply(lambda x: round(x * 30) / 30 if pd.notnull(x) else x)  # Round to nearest 0.03 seconds
    df['Frame Name'] = df['Time_point_s'].apply(lambda x: f"tracked_{x:.2f}.png" if pd.notnull(x) else x)
    return df

def get_sorted_image_names(file_names):
    logging.debug("Sorting image names.")
    # Sort numerically by the timestamp portion of the file name
    sorted_files = sorted(file_names, key=lambda x: float(os.path.basename(x).split('_')[1].replace('.png', '')))
    return sorted_files

def generate_imputed_rows(last_frame_time):
    """
    Generate rows with incremented timestamps starting from 0 to the last frame time.
    Each timestamp will be incremented by 1/30 seconds and rounded to two decimal places.
    """
    logging.debug(f"Generating imputed rows up to {last_frame_time:.2f} seconds.")
    
    frame_duration = 1 / 30  # Exact frame duration for 30 FPS
    imputed_rows = []
    current_frame = 0  # Start from frame 0
    current_time = 0.00
    
    while current_time <= last_frame_time:
        frame_name = f"tracked_{current_time:.2f}.png"
        imputed_rows.append({'Frame Name': frame_name, 'Time_point_s': round(current_time, 2), 'Behavior': ''})
        
        # Increment by the frame duration
        current_frame += 1
        current_time = round(current_frame * frame_duration, 2)  # Calculate the next timestamp
        
    return imputed_rows

def process_single_excel_file(excel_file, image_folder, output_dir):
    try:
        logging.info(f"Processing file: {excel_file}")
        df = pd.read_excel(excel_file)
        df = round_up_timestamps(df)

        # Get all sorted image names (looking for .png files now)
        image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png')]
        sorted_image_names = get_sorted_image_names(image_files)

        # Determine the last frame time from the sorted images
        if sorted_image_names:
            last_image = sorted_image_names[-1]
            last_frame_time = float(os.path.basename(last_image).split('_')[1].replace('.png', ''))
        else:
            logging.warning(f"No images found in {image_folder}.")
            return

        # Impute rows with incremented timestamps
        imputed_rows = generate_imputed_rows(last_frame_time)
        imputed_df = pd.DataFrame(imputed_rows)

        # Merge imputed rows with the existing DataFrame, without overwriting existing rows
        df = pd.concat([df, imputed_df[~imputed_df['Frame Name'].isin(df['Frame Name'])]], ignore_index=True)

        # Log the DataFrame shape to ensure it’s not empty
        logging.info(f"DataFrame shape before saving: {df.shape}")

        # Sort by the 'Time_point_s' column after merging
        df = df.sort_values(by=['Time_point_s']).reset_index(drop=True)

        # Ensure output directory (including split subfolder) exists
        logging.info(f"Ensuring output directory exists: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        # Save the updated DataFrame to the output directory
        output_excel_path = os.path.join(output_dir, os.path.basename(excel_file))
        logging.info(f"Attempting to save Excel file to: {output_excel_path}")
        df.to_excel(output_excel_path, index=False)
        logging.info(f"Successfully saved updated Excel file to {output_excel_path}")
    
    except Exception as e:
        logging.error(f"Error processing {excel_file}: {e}")

def filter_and_update_excel_files(split, base_excel_dir, base_output_dir, base_image_dir):
    os.makedirs(base_output_dir, exist_ok=True)
    max_workers = max(1, os.cpu_count() - 2)
    tasks = []

    logging.info(f"Walking through the directory: {base_excel_dir}")

    # Process the specified split (train, test, etc.)
    logging.info(f"Processing split: {split}")

    excel_split_dir = os.path.join(base_excel_dir, split)  # Excel folder (e.g., labels/{split})
    image_split_dir = os.path.join(base_image_dir, split)  # Image folder (e.g., FRCNN+yolo/{split})

    excel_files = []
    image_folders = []

    # Find all Excel files in the split folder
    for root, _, files in os.walk(excel_split_dir):
        for file in files:
            if file.endswith('.xlsx'):
                excel_files.append(os.path.join(root, file))

    # Find all image folders in the corresponding split folder
    for root_img, dirs, _ in os.walk(image_split_dir):
        for dir_name in dirs:
            if dir_name.endswith('_fixed'):
                image_folders.append(os.path.join(root_img, dir_name))

    # Match Excel files with corresponding image folders
    for excel_file in excel_files:
        subject_info = os.path.basename(excel_file).replace('.xlsx', '')
        for image_folder in image_folders:
            if subject_info in os.path.basename(image_folder):
                logging.info(f"Matched Excel file: {excel_file} with image folder: {image_folder}")

                # Ensure the subfolder for the current split exists within the output directory
                split_output_dir = os.path.join(base_output_dir, split)
                tasks.append((excel_file, image_folder, split_output_dir))

    # Use ThreadPoolExecutor to process files concurrently
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
base_output_dir = '/storage/group/klk37/default/homebytes/video/fbs/labels_1'
base_image_dir = '/storage/group/klk37/default/homebytes/video/fbs/FRCNN+yolo'

# Run the function for the train and test folders
filter_and_update_excel_files('test', base_excel_dir, base_output_dir, base_image_dir)
filter_and_update_excel_files('train', base_excel_dir, base_output_dir, base_image_dir)
filter_and_update_excel_files('val', base_excel_dir, base_output_dir, base_image_dir)
