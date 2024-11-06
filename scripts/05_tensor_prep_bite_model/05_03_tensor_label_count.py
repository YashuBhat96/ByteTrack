import torch
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to count bite vs non-bite sequences in a single tensor file
def count_bite_vs_non_bite(tensor_file):
    try:
        # Load the .pt file
        data = torch.load(tensor_file)
        features, labels = data['features'], data['labels']
        
        # Convert labels to a list (labels are already a tensor)
        labels = list(labels.numpy())  # Convert tensor to a list
        
        # Count bite and non-bite sequences
        bite_count = labels.count(1)
        non_bite_count = labels.count(0)
        
        logging.info(f"File '{tensor_file}' - Bite Sequences: {bite_count}, Non-Bite Sequences: {non_bite_count}")
        return bite_count, non_bite_count
    except Exception as e:
        logging.error(f"Error processing file '{tensor_file}': {str(e)}")
        return 0, 0

# Function to process multiple files in parallel
def process_files_in_parallel(tensor_files, set_name):
    bite_total = 0
    non_bite_total = 0
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(count_bite_vs_non_bite, file) for file in tensor_files]
        
        # Collect results as they complete
        for future in as_completed(futures):
            try:
                bite_count, non_bite_count = future.result()
                bite_total += bite_count
                non_bite_total += non_bite_count
            except Exception as e:
                logging.error(f"Error in parallel processing: {str(e)}")
    
    logging.info(f"{set_name} Set - Total Bite Sequences: {bite_total}")
    logging.info(f"{set_name} Set - Total Non-Bite Sequences: {non_bite_total}")

# Example usage for train, validation, and test sets
if __name__ == "__main__":
    # Tensor files for training set
    train_tensor_files = [
        '/storage/group/klk37/default/homebytes/video/fbs/tensors_train_lstm_new_1/consolidated_bite_sequences.pt',
        '/storage/group/klk37/default/homebytes/video/fbs/tensors_train_lstm_new_1/consolidated_non_bite_sequences.pt'
    ]
    
    # Tensor files for validation set
    val_tensor_files = [
        '/storage/group/klk37/default/homebytes/video/fbs/tensors_val_lstm_new_1/consolidated_bite_sequences.pt',
        '/storage/group/klk37/default/homebytes/video/fbs/tensors_val_lstm_new_!/consolidated_non_bite_sequences.pt'
    ]
    
 
    # Process train set
    logging.info("Processing Train Set")
    process_files_in_parallel(train_tensor_files, "Train")
    
    # Process validation set
    logging.info("Processing Validation Set")
    process_files_in_parallel(val_tensor_files, "Validation")

