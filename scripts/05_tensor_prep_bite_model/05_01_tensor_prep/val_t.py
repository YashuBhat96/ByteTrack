import os
import pandas as pd
import torch
from torchvision import models, transforms
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging
import gc

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Image transformation for input images to EfficientNet (without resizing)
image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# EfficientNet model setup
efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
efficientnet.classifier = torch.nn.Identity()  # Remove the classification layer for feature extraction
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
efficientnet.to(device)
efficientnet.eval()

# Function to match Excel file to its corresponding image folder
def match_excel_with_folder(excel_file, root_image_dir):
    folder_name = os.path.splitext(os.path.basename(excel_file))[0] + '_fixed'
    folder_path = os.path.join(root_image_dir, folder_name)
    return folder_path if os.path.exists(folder_path) else None

# Handle missing frames with padding and create a mask for bite and non-bite sequences
def handle_missing_frames(images, mask, sequence_length=50, max_padding=5):
    padding_count = sequence_length - len(images)

    # If more than 5 frames are missing, discard the sequence
    if padding_count > max_padding:
        return None, None  # Return None to signal the sequence should be discarded

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

    # If too many frames are missing, discard the sequence
    if images is None:
        return None, label, mask, missing_frame_count

    images_tensor = torch.stack(images[:sequence_length])  # Stack images to form the sequence tensor
    mask_tensor = torch.tensor(mask[:sequence_length])  # Convert mask to tensor

    # Return image tensors and mask tensors
    return images_tensor, label, mask_tensor, missing_frame_count

# Extract features using EfficientNet and return mask for LSTM
def extract_features_with_efficientnet(images_tensor, mask_tensor, batch_size=24):
    features_list = []
    with torch.no_grad():
        for i in range(0, len(images_tensor), batch_size):
            batch_images = images_tensor[i:i + batch_size].to(device)
            batch_mask = mask_tensor[i:i + batch_size].to(device)

            # Ensure the mask and images batch size match
            if batch_images.size(0) != batch_mask.size(0):
                raise RuntimeError(f"Batch size mismatch: images {batch_images.size(0)}, mask {batch_mask.size(0)}")

            # Pass through EfficientNet to get features
            features = efficientnet(batch_images)  # Shape: (batch_size, num_features)

            # Expand the mask to match the feature size
            expanded_mask = batch_mask.unsqueeze(-1).expand_as(features)  # Expand to match feature size
            masked_features = features * expanded_mask  # Zero out the features where the mask is 0

            features_list.append(masked_features.cpu())

    # Concatenate all features after processing in batches
    concatenated_features = torch.cat(features_list)

    return concatenated_features, mask_tensor  # Returning both features and the mask for LSTM

# Process a single Excel file and log bite/non-bite sequence statistics
def process_excel_file(excel_file, root_image_dir, sequence_length=50, batch_size=24, num_workers=8):
    logging.info(f"Loading {excel_file} for processing...")
    df = pd.read_excel(excel_file)
    
    folder_path = match_excel_with_folder(excel_file, root_image_dir)
    if not folder_path:
        logging.warning(f"Image folder not found for {excel_file}")
        return None

    features_list = []
    labels_list = []  # Store corresponding labels for each sequence (0 for non-bite, 1 for bite)
    masks_list = []   # Store masks for LSTM usage

    bite_seq_count = 0
    non_bite_seq_count = 0

    # Group by 'Sequence_Label' instead of 'Sequence Number'
    unique_sequences = df.groupby('Sequence_Label')

    # Parallel processing within the file
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for sequence_label, frames in unique_sequences:
            # Debugging: Print out the sequence labels to check why non-bite might not be processed
            logging.debug(f"Processing sequence label: {sequence_label}")

            # Check for bite sequence (label 1) and non-bite sequence (label 0)
            if sequence_label.startswith("bite_seq_"):
                label = 1  # Bite sequence
                bite_seq_count += 1
            elif sequence_label.startswith("non_bite_seq_"):
                label = 0  # Non-bite sequence
                non_bite_seq_count += 1
            else:
                # Logging unknown sequence labels for further investigation
                logging.warning(f"Unknown sequence label {sequence_label}. Skipping sequence.")
                continue

            # Process each sequence (either bite or non-bite) as a distinct unit
            futures.append(executor.submit(process_sequence, sequence_label, frames, folder_path, label, sequence_length))

        for future in as_completed(futures):
            images_tensor, label, mask_tensor, _ = future.result()
            if images_tensor is None:
                continue  # Skip if too many frames are missing
            features, final_mask = extract_features_with_efficientnet(images_tensor, mask_tensor, batch_size=batch_size)
            features_list.append(features)
            labels_list.append(label)  # Append single label for each sequence
            masks_list.append(final_mask)  # Store the mask for LSTM input

    # Log how many viable bite and non-bite sequences were processed
    logging.info(f"Processed {excel_file}: {bite_seq_count} bite sequences, {non_bite_seq_count} non-bite sequences.")

    # Save features, labels, and masks for this Excel file
    output_file = f"/storage/group/klk37/default/homebytes/video/fbs/val_new/{os.path.splitext(os.path.basename(excel_file))[0]}_sequences.pt"
    torch.save({
        'features': torch.cat(features_list),
        'labels': torch.tensor(labels_list),
        'masks': torch.cat(masks_list)
    }, output_file)

    logging.info(f"Saved features, labels, and masks to {output_file}")

    # Free memory after processing this Excel file
    del features_list, labels_list, masks_list, df
    gc.collect()

# Main function for processing all Excel files sequentially with global tqdm
def process_all_excel_files(root_excel_dir, root_image_dir, sequence_length=50, num_workers=8, batch_size=24):
    excel_files = [os.path.join(root_excel_dir, f) for f in os.listdir(root_excel_dir) if f.endswith('.xlsx')]
    
    # Overall progress tracking with global tqdm
    with tqdm(total=len(excel_files), desc="Processing all Excel files") as pbar:
        for excel_file in excel_files:
            process_excel_file(excel_file, root_image_dir, sequence_length=sequence_length, batch_size=batch_size, num_workers=num_workers)
            pbar.update(1)

if __name__ == '__main__':
    root_excel_dir = '/storage/group/klk37/default/homebytes/video/fbs/labels_3/val'
    root_image_dir = '/storage/group/klk37/default/homebytes/video/fbs/FRCNN+yolo/val'

    # Process all Excel files one by one, save .pt for each
    process_all_excel_files(root_excel_dir, root_image_dir, sequence_length=50, num_workers=8, batch_size=24)
