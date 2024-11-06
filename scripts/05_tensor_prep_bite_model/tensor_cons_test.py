import os
import torch
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Path to the folder containing all .pt files
pt_folder = '/storage/group/klk37/default/homebytes/video/fbs/test_new'

# Function to load a .pt file and return its contents
def load_pt_file(pt_file):
    try:
        data = torch.load(pt_file, map_location=torch.device('cpu'))  # Load to CPU
        logging.info(f"Loaded {pt_file} successfully.")
        return data
    except Exception as e:
        logging.error(f"Error loading {pt_file}: {str(e)}")
        return None

# Function to consolidate the data from all .pt files
def consolidate_pt_files(pt_folder, num_workers=8):
    pt_files = [os.path.join(pt_folder, f) for f in os.listdir(pt_folder) if f.endswith('.pt')]
    
    features_list = []
    labels_list = []
    masks_list = []

    # Use ThreadPoolExecutor to process .pt files in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(load_pt_file, pt_file): pt_file for pt_file in pt_files}
        
        for future in as_completed(futures):
            pt_file = futures[future]
            data = future.result()

            if data is None:
                logging.warning(f"Skipping {pt_file} due to load failure.")
                continue

            # Extract features, labels, and masks
            features_list.append(data['features'])
            labels_list.append(data['labels'])
            masks_list.append(data['masks'])
            
            # Log success
            logging.info(f"Consolidated data from {pt_file}")
            
            # Free memory after each file
            del data
            gc.collect()

    # Concatenate all features, labels, and masks
    consolidated_features = torch.cat(features_list, dim=0)
    consolidated_labels = torch.cat(labels_list, dim=0)
    consolidated_masks = torch.cat(masks_list, dim=0)

    logging.info(f"Consolidation completed: {len(features_list)} files combined.")
    
    # Save the consolidated .pt file
    output_file = os.path.join(pt_folder, 'consolidated_sequences.pt')
    torch.save({
        'features': consolidated_features,
        'labels': consolidated_labels,
        'masks': consolidated_masks
    }, output_file)
    
    logging.info(f"Consolidated .pt file saved at {output_file}")

if __name__ == '__main__':
    consolidate_pt_files(pt_folder, num_workers=8)
v