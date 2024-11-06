import os
import pandas as pd
import random
import shutil
import concurrent.futures
import logging

# Paths to the provided Excel files
train_excel = '/storage/group/klk37/default/homebytes/video/fbs/model1_split/train.xlsx'
val_excel = '/storage/group/klk37/default/homebytes/video/fbs/model1_split/val.xlsx'
test_excel = '/storage/group/klk37/default/homebytes/video/fbs/model1_split/test.xlsx'

# Path to the input folder containing all subfolders
input_folder = '/storage/group/klk37/default/homebytes/video/fbs/FRCNN+yolo'

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Function to read Excel files and return the list of subfolders
def read_excel(file_path):
    df = pd.read_excel(file_path)
    return df['Subject'].tolist()

# Read subfolders from provided Excel files
train_subfolders = read_excel(train_excel)
val_subfolders = read_excel(val_excel)
test_subfolders = read_excel(test_excel)

# Combine all subfolders from Excel files
all_excel_subfolders = set(train_subfolders + val_subfolders + test_subfolders)

# Get all subfolders in the input folder
all_input_subfolders = [f for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))]

# Filter out subfolders that are already in the Excel files
remaining_subfolders = [f for f in all_input_subfolders if f not in all_excel_subfolders]

# Shuffle and split the remaining subfolders
random.shuffle(remaining_subfolders)
total_count = len(remaining_subfolders)
train_count = int(total_count * 0.7)
val_count = int(total_count * 0.15)
test_count = total_count - train_count - val_count

additional_train = remaining_subfolders[:train_count]
additional_val = remaining_subfolders[train_count:train_count + val_count]
additional_test = remaining_subfolders[train_count + val_count:]

# Combine subfolders
final_train = train_subfolders + additional_train
final_val = val_subfolders + additional_val
final_test = test_subfolders + additional_test

# Function to move subfolders to the corresponding output directory
def move_subfolders(subfolder, output_subfolder):
    output_path = os.path.join(input_folder, output_subfolder)
    os.makedirs(output_path, exist_ok=True)
    src = os.path.join(input_folder, subfolder)
    dst = os.path.join(output_path, subfolder)
    shutil.move(src, dst)
    logging.info(f"Moved {subfolder} to {output_subfolder}")

# Move subfolders to train, val, and test directories using parallel processing
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    futures = []
    for subfolder in final_train:
        futures.append(executor.submit(move_subfolders, subfolder, 'train'))
    for subfolder in final_val:
        futures.append(executor.submit(move_subfolders, subfolder, 'val'))
    for subfolder in final_test:
        futures.append(executor.submit(move_subfolders, subfolder, 'test'))
    
    for future in concurrent.futures.as_completed(futures):
        future.result()  # Raise exception if occurred

logging.info("Subfolders have been split and moved successfully.")
print("Subfolders have been split and moved successfully.")
