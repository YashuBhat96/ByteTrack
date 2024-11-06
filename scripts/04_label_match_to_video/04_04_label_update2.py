import os
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Function to label bites, handle overlap, and create sequences (including non-bite sequences)
def label_bites_and_non_bites_in_file(file_path, save_dir):
    try:
        # Load the Excel file
        df = pd.read_excel(file_path)

        # Convert 'Time_relative_sf' to 'Time_point_s' if necessary
        if 'Time_relative_sf' in df.columns and 'Time_point_s' not in df.columns:
            df['Time_point_s'] = df['Time_relative_sf']
        
        # Create or modify a 'Label' column, defaulting to 0 (non-bite)
        df['Label'] = 0
        df['Sequence_Label'] = ""  # New column for sequence labeling
        df['Overlap'] = ""  # Track overlap frames

        # Define a window of 50 frames around each peak:bite (25 before and 25 after)
        frame_window = 25

        # Find the indices where 'Behavior' is 'peak:bite'
        peak_bite_indices = df[df['Behavior'] == 'peak:bite'].index

        # Initialize a set to track the indices that should be labeled as 'bite'
        bite_indices = set()

        # Store start-end ranges for each peak:bite window to check for overlap later
        ranges = []

        # Label 25 frames before and 25 frames after (total 50 frames) for each peak:bite
        for index in peak_bite_indices:
            start = max(index - frame_window, 0)  # Ensure we don't go below 0
            end = min(index + frame_window, len(df) - 1)  # Ensure we don't exceed the dataframe length
            ranges.append((start, end))  # Track the range for overlap checking
            bite_indices.update(range(start, end + 1))

        # Set the 'Label' column to 1 for all indices marked as bite
        df.loc[list(bite_indices), 'Label'] = 1

        # Initialize counters for bite sequences
        bite_seq_counter = 0

        # Track start and end of bite sequences
        current_bite_seq_start = None

        # Iterate through the DataFrame to group bite frames into sequences
        for i in range(len(df)):
            if df.loc[i, 'Label'] == 1:  # If it's a bite frame
                if current_bite_seq_start is None:
                    current_bite_seq_start = i  # Start of a new bite sequence
            else:
                if current_bite_seq_start is not None:
                    # End of the current bite sequence
                    current_bite_seq_end = i - 1
                    sequence_label = f"bite_seq_{bite_seq_counter}"
                    df.loc[current_bite_seq_start:current_bite_seq_end, 'Sequence_Label'] = sequence_label
                    bite_seq_counter += 1
                    current_bite_seq_start = None  # Reset for the next sequence

        # If the last frames are part of a bite sequence, close it off
        if current_bite_seq_start is not None:
            current_bite_seq_end = len(df) - 1
            sequence_label = f"bite_seq_{bite_seq_counter}"
            df.loc[current_bite_seq_start:current_bite_seq_end, 'Sequence_Label'] = sequence_label
            bite_seq_counter += 1

        # Handle overlap - duplicate frames for overlapping bite sequences
        for i in range(1, len(ranges)):
            current_range = ranges[i - 1]
            next_range = ranges[i]
            if current_range[1] >= next_range[0]:  # Overlap detected
                overlap_start = next_range[0]
                overlap_end = current_range[1]
                current_seq_label = f"bite_seq_{i - 1}"
                next_seq_label = f"bite_seq_{i}"
                # Assign overlapping frames to both sequences
                df.loc[overlap_start:overlap_end, 'Sequence_Label'] += f" | {next_seq_label}"

        # Now, create non-bite sequences:
        # Apply a 20-frame buffer around each bite sequence
        buffer_size = 20
        bite_ranges_with_buffer = set()
        
        # Extend each bite sequence by 20 frames before and after
        for start, end in ranges:
            extended_start = max(0, start - buffer_size)
            extended_end = min(len(df) - 1, end + buffer_size)
            bite_ranges_with_buffer.update(range(extended_start, extended_end + 1))
        
        # Initialize counter for non-bite sequences
        non_bite_seq_counter = 0
        
        # Track start and end of non-bite sequences
        current_non_bite_seq_start = None
        
        for i in range(len(df)):
            if df.loc[i, 'Label'] == 0 and i not in bite_ranges_with_buffer:  # Non-bite frame not in buffer zone
                if current_non_bite_seq_start is None:
                    current_non_bite_seq_start = i  # Start of a new non-bite sequence
            else:
                if current_non_bite_seq_start is not None:
                    # End of the current non-bite sequence
                    current_non_bite_seq_end = i - 1
                    # Group into consecutive 50-frame sequences
                    while current_non_bite_seq_start <= current_non_bite_seq_end:
                        non_bite_seq_end = min(current_non_bite_seq_start + 49, current_non_bite_seq_end)
                        sequence_label = f"non_bite_seq_{non_bite_seq_counter}"
                        df.loc[current_non_bite_seq_start:non_bite_seq_end, 'Sequence_Label'] = sequence_label
                        non_bite_seq_counter += 1
                        current_non_bite_seq_start += 50  # Move to the next block of 50 frames
                    current_non_bite_seq_start = None  # Reset for the next sequence

        # Ensure the save directory exists, create it if not
        os.makedirs(save_dir, exist_ok=True)

        # Create output path using the original file name in the specified save directory
        file_name = os.path.basename(file_path)  # Keep the same name as the original file
        output_file_path = os.path.join(save_dir, file_name)
        df.to_excel(output_file_path, index=False)

        logger.info(f"Processed and labeled file: {file_path}")
        return output_file_path

    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return None

# Function to process files in a split directory (train, val, test) using concurrent futures
def process_split_concurrently(split_dir, save_dir):
    labeled_files = []
    excel_files = [os.path.join(root, file)
                   for root, dirs, files in os.walk(split_dir)
                   for file in files if file.endswith('.xlsx')]

    with ThreadPoolExecutor() as executor:
        # Submit tasks to executor
        future_to_file = {executor.submit(label_bites_and_non_bites_in_file, file, save_dir): file for file in excel_files}

        # Collect the results as they complete
        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                result = future.result()
                if result:
                    labeled_files.append(result)
            except Exception as e:
                logger.error(f"Error processing file {file}: {e}")
    
    return labeled_files

# Main function to process the entire root directory concurrently
def process_root_directory_concurrently(root_dir, save_dir):
    splits = ['train', 'val', 'test']
    for split in splits:
        split_dir = os.path.join(root_dir, split)
        save_split_dir = os.path.join(save_dir, split)  # Ensure subdirectories match the input
        if os.path.exists(split_dir):
            logger.info(f"Processing {split} split concurrently...")
            
            # Ensure the subdirectory for the current split exists
            os.makedirs(save_split_dir, exist_ok=True)

            labeled_files = process_split_concurrently(split_dir, save_split_dir)
            logger.info(f"Finished processing {split} split. Labeled files: {labeled_files}")
        else:
            logger.warning(f"{split} directory does not exist in {root_dir}")

# Example usage
root_directory = '/storage/group/klk37/default/homebytes/video/fbs/labels_1'  # Set your root directory path here
save_directory = '/storage/group/klk37/default/homebytes/video/fbs/labels_3'
process_root_directory_concurrently(root_directory, save_directory)
