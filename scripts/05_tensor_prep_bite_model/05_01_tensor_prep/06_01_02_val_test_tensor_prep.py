import os
import pandas as pd
import torch
from torchvision import models, transforms
from PIL import Image
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # For progress tracking
import logging
import gc  # For garbage collection

# Setup global logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Transformation for input images to EfficientNet
image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Helper function to extract numeric part from frame name
def extract_numeric_part(frame_name):
    match = re.search(r'tracked_(\d+\.\d+)\.png', frame_name)
    return float(match.group(1)) if match else None

# Function to match Excel file with folder based on naming
def match_excel_with_folder(excel_file, root_image_dir):
    folder_name = os.path.splitext(os.path.basename(excel_file))[0] + '_fixed'
    folder_path = os.path.join(root_image_dir, folder_name)
    return folder_path if os.path.exists(folder_path) else None

# Function to parse Excel file and get unique sequences
def parse_excel_file(excel_file_path):
    df = pd.read_excel(excel_file_path)
    unique_sequences = df.groupby('Sequence Number')
    return unique_sequences

# EfficientNet feature extraction setup
efficientnet = models.efficientnet_b0(pretrained=True)
efficientnet.classifier = torch.nn.Identity()  # Remove the classification layer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available
efficientnet.to(device)

# Since we're only extracting features for now, keep EfficientNet in eval mode
efficientnet.eval()  # No need for fine-tuning right now

# Handle missing frames with padding and create a mask for bite and non-bite sequences
def handle_missing_frames(images, mask, sequence_length=50, max_padding=5):
    padding_count = sequence_length - len(images)

    if padding_count > max_padding:
        # Discard sequence if more than max_padding frames are missing
        return None, None

    # Pad missing frames with zeros for both bite and non-bite sequences
    images.extend([torch.zeros(3, 224, 224)] * padding_count)
    mask.extend([0.0] * padding_count)  # Pad the mask with 0.0 for the padded frames

    return images, mask

# Process a single sequence, applying padding and masking to ensure the sequence is 50 frames long
def process_sequence(sequence_number, frames, folder_path, label, sequence_length=50):
    images = []
    mask = []
    missing_frame_count = 0

    for idx, row in frames.iterrows():
        frame_name = row['Frame Name']
        frame_path = os.path.join(folder_path, frame_name)
        if os.path.exists(frame_path):
            image = Image.open(frame_path).convert('RGB')
            image_tensor = image_transform(image)  # Apply the transformation
            image.close()  # Close the PIL image to free memory
            images.append(image_tensor)  # Append the tensor
            mask.append(1.0)  # Mark as a valid frame
        else:
            missing_frame_count += 1
            images.append(torch.zeros(3, 224, 224))  # Placeholder for missing frames
            mask.append(0.0)  # Mark as a missing frame

    # Apply padding and create a mask for missing frames
    images, mask = handle_missing_frames(images, mask, sequence_length, max_padding=5)

    if images is None:  # If too many frames are missing, discard the sequence
        return None, label, mask, missing_frame_count

    images_tensor = torch.stack(images[:sequence_length])  # Stack images to form the sequence tensor
    mask_tensor = torch.tensor(mask[:sequence_length])  # Convert mask to tensor

    # Return image tensors and mask tensors
    return images_tensor, label, mask_tensor, missing_frame_count

# EfficientNet feature extraction with masking
def extract_features_with_efficientnet(batch_data, mask):
    with torch.no_grad():
        # Extract features from original frames
        frame_features = efficientnet(batch_data.to(device))
        
        # Apply the mask to zero out features corresponding to padded frames
        masked_frame_features = frame_features * mask.unsqueeze(-1)  # Apply mask to original frames
    
    return masked_frame_features.cpu()

# Extract features for sequences with mask
def extract_features_for_sequences(sequences, masks, batch_size=24):
    feature_list = []
    for images_tensor, mask_tensor in zip(sequences, masks):
        features_list = []
        for i in range(0, len(images_tensor), batch_size):
            batch = images_tensor[i:i + batch_size]
            mask = mask_tensor[i:i + batch_size]
            
            # Extract features from original frames
            features = extract_features_with_efficientnet(batch, mask)
            features_list.append(features)

            # Free up memory incrementally after each batch
            gc.collect()

        feature_list.append(torch.cat(features_list, dim=0))
    return feature_list

# Checkpointing function to save intermediate results and free memory
def save_checkpoint(bite_features, non_bite_features, checkpoint_dir, file_name):
    checkpoint_path = os.path.join(checkpoint_dir, f"{file_name}_checkpoint.pt")
    torch.save({
        'bite_features': bite_features,
        'non_bite_features': non_bite_features,
    }, checkpoint_path)
    
    # After saving, clear the features to free up memory
    del bite_features
    del non_bite_features
    torch.cuda.empty_cache()  # Free GPU memory
    gc.collect()  # Free CPU memory

# Process a single Excel file, but do feature extraction after downsampling
def process_single_excel_file(excel_file, root_image_dir, sequence_length, checkpoint_dir):
    try:
        image_folder = match_excel_with_folder(excel_file, root_image_dir)
        if image_folder is None:
            return None, None

        unique_sequences = parse_excel_file(excel_file)

        bite_sequences = []
        non_bite_sequences = []
        file_missing_frames = 0

        for sequence_number, frames in unique_sequences:
            label = frames['Sequence_Label'].iloc[0]
            images_tensor, label, mask, missing_frame_count = process_sequence(sequence_number, frames, image_folder, label, sequence_length)
            file_missing_frames += missing_frame_count

            if images_tensor is not None:
                sequence_data = {'images': images_tensor, 'mask': mask, 'label': label, 'Frame Name': frames['Frame Name'].iloc[0]}
                if label == 1:
                    bite_sequences.append(sequence_data)
                else:
                    non_bite_sequences.append(sequence_data)

        # Clear memory after processing sequences
        gc.collect()

        logging.info(f"Processed {len(bite_sequences)} bite sequences and {len(non_bite_sequences)} non-bite sequences for {os.path.basename(excel_file)}")

        # Extract features using EfficientNet only for the sequences
        bite_images = [seq['images'] for seq in bite_sequences]
        bite_masks = [seq['mask'] for seq in bite_sequences]

        non_bite_images = [seq['images'] for seq in non_bite_sequences]
        non_bite_masks = [seq['mask'] for seq in non_bite_sequences]

        bite_features = extract_features_for_sequences(bite_images, bite_masks, batch_size=24)
        non_bite_features = extract_features_for_sequences(non_bite_images, non_bite_masks, batch_size=24)

        file_name = os.path.splitext(os.path.basename(excel_file))[0]
        save_checkpoint(bite_features, non_bite_features, checkpoint_dir, file_name)

        # Clear memory after saving checkpoints
        del bite_images, bite_masks, non_bite_images, non_bite_masks
        torch.cuda.empty_cache()  # Free GPU memory
        gc.collect()  # Free CPU memory

        return bite_features, non_bite_features

    except Exception as e:
        logging.error(f"Error processing file {excel_file}: {str(e)}")
        return None, None

# Main function with ThreadPoolExecutor for parallel processing, using 20 workers
def process_all_parallel(root_excel_dir, root_image_dir, output_folder, sequence_length=50, num_workers=12):
    excel_files = [os.path.join(root_excel_dir, f) for f in os.listdir(root_excel_dir) if f.endswith('.xlsx')]
    global_bite_features = []
    global_non_bite_features = []
    checkpoint_dir = output_folder  # Save checkpoints to log directory

    # Progress tracking with tqdm
    with tqdm(total=len(excel_files), desc="Processing Excel files") as pbar:

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_single_excel_file, excel_file, root_image_dir, sequence_length, checkpoint_dir): excel_file for excel_file in excel_files}

            for future in as_completed(futures):
                try:
                    bite_features, non_bite_features = future.result()
                    if bite_features:
                        global_bite_features.extend(bite_features)
                    if non_bite_features:
                        global_non_bite_features.extend(non_bite_features)
                    pbar.update(1)

                    # Save checkpoints and free memory after processing every file
                    file_counter = pbar.n  # Use the progress bar counter as the file number
                    logging.info(f"Checkpointing after file {file_counter}")
                    save_checkpoint(global_bite_features, global_non_bite_features, checkpoint_dir, f"checkpoint_{file_counter}")
                    global_bite_features.clear()
                    global_non_bite_features.clear()
                    gc.collect()
                    torch.cuda.empty_cache()

                except Exception as e:
                    logging.error(f"Error processing file: {futures[future]} - {str(e)}")

    # Log final global sequence counts
    logging.info(f"Total bite sequences processed: {len(global_bite_features)}")
    logging.info(f"Total non-bite sequences processed: {len(global_non_bite_features)}")

    # Save global tensors at the end if any remaining features
    if global_bite_features:
        global_bite_tensor = torch.cat(global_bite_features)
        torch.save(global_bite_tensor, os.path.join(output_folder, 'all_bite_sequences.pt'))
        logging.info(f'Saved global bite sequences at {os.path.join(output_folder, "all_bite_sequences.pt")}')

    if global_non_bite_features:
        global_non_bite_tensor = torch.cat(global_non_bite_features)
        torch.save(global_non_bite_tensor, os.path.join(output_folder, 'all_non_bite_sequences.pt'))
        logging.info(f'Saved global non-bite sequences at {os.path.join(output_folder, "all_non_bite_sequences.pt")}')
        
if __name__ == '__main__':
    root_val_dir = '/storage/group/klk37/default/homebytes/video/fbs/labels_4/test'
    val_output_folder = '/storage/group/klk37/default/homebytes/video/fbs/tensors_test_lstm_new'
    # Process validation set separately
    process_all_parallel(root_val_dir, '/storage/group/klk37/default/homebytes/video/fbs/FRCNN+yolo/test', val_output_folder, sequence_length=50, num_workers=12)

    # Process test set separately
