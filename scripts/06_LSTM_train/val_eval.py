# Validation function with binary classification metrics
def validate_model(model, val_loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    running_val_loss = 0.0
    val_predictions, val_labels_list = [], []
    val_probabilities = []

    with torch.no_grad():
        for val_features, val_labels_batch in val_loader:
            val_features, val_labels_batch = val_features.to(device), val_labels_batch.to(device)
            
            # Forward pass
            outputs = model(val_features)
            loss = criterion(outputs, val_labels_batch)
            running_val_loss += loss.item()
            
            # Get predicted classes and probabilities
            _, preds = torch.max(outputs, 1)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]  # Probability for class 1
            
            val_predictions.extend(preds.cpu().numpy())
            val_labels_list.extend(val_labels_batch.cpu().numpy())
            val_probabilities.extend(probabilities.cpu().numpy())

    # Convert predictions and labels to numpy arrays
    val_predictions = np.array(val_predictions)
    val_labels_list = np.array(val_labels_list)

    # Confusion Matrix
    conf_matrix = confusion_matrix(val_labels_list, val_predictions)
    TP = conf_matrix[1, 1]  # True positives
    TN = conf_matrix[0, 0]  # True negatives
    FP = conf_matrix[0, 1]  # False positives
    FN = conf_matrix[1, 0]  # False negatives

    # Accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    # Precision, Recall, and F1 Score for class 1
    precision = precision_score(val_labels_list, val_predictions, pos_label=1)
    recall = recall_score(val_labels_list, val_predictions, pos_label=1)
    f1 = f1_score(val_labels_list, val_predictions, pos_label=1)

    # AUC-PR score
    auc_pr = average_precision_score(val_labels_list, val_probabilities)
    
    # AUC-ROC score
    auc_roc = roc_auc_score(val_labels_list, val_probabilities)

    return {
        "val_loss": running_val_loss / len(val_loader),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc_pr": auc_pr,
        "auc_roc": auc_roc,
        "conf_matrix": conf_matrix,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN
    }

# Main function for validation
def main():
    # Paths for validation data
    val_bite_tensor_path = '/storage/group/klk37/default/homebytes/video/fbs/tensors_val_lstm_new/consolidated_bite_sequences.pt'
    val_non_bite_tensor_path = '/storage/group/klk37/default/homebytes/video/fbs/tensors_val_lstm_new/consolidated_non_bite_sequences.pt'
    
    # Path for the best saved model
    model_path = '/path/to/best_model.pth'

    # Load the bite and non-bite data
    logging.info("Loading validation datasets...")
    val_bite_data = torch.load(val_bite_tensor_path)
    val_non_bite_data = torch.load(val_non_bite_tensor_path)

    # Extract features and labels
    val_bite_features, val_bite_labels = val_bite_data['features'], val_bite_data['labels']
    val_non_bite_features, val_non_bite_labels = val_non_bite_data['features'], val_non_bite_data['labels']

    # Reshape features to match sequence length
    sequence_length = 50
    val_bite_features = reshape_features(val_bite_features, sequence_length)
    val_non_bite_features = reshape_features(val_non_bite_features, sequence_length)

    # Ensure labels match the number of sequences
    val_bite_labels = val_bite_labels[:len(val_bite_features)]
    val_non_bite_labels = val_non_bite_labels[:len(val_non_bite_features)]

    # Concatenate features and labels for validation set
    val_features = torch.cat((val_bite_features, val_non_bite_features), dim=0)
    val_labels = torch.cat((val_bite_labels, val_non_bite_labels), dim=0)

    # Set device to CPU (since GPU is not available)
    device = torch.device('cpu')

    # Load the saved model
    model = LSTMModel(
        input_size=val_features.size(2),
        hidden_size=128,  # Fixed hyperparameter
        num_layers=2,  # Fixed hyperparameter
        num_classes=2,  # Binary classification
        dropout_rate=0.3,  # Fixed hyperparameter
        bidirectional=True,  # Fixed hyperparameter
        device=device
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))

    # Define the validation dataset and data loader
    val_dataset = CombinedDataset(val_features, val_labels)
    
    # Use DataLoader with multiprocessing for parallel processing
    num_workers = multiprocessing.cpu_count()  # Use all available CPU cores
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=num_workers)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Validate the model and compute metrics
    metrics = validate_model(model, val_loader, criterion, device)

    # Log the results
    logging.info(f'Validation Loss: {metrics["val_loss"]:.4f}')
    logging.info(f'Accuracy: {metrics["accuracy"]:.4f}')
    logging.info(f'Precision: {metrics["precision"]:.4f}')
    logging.info(f'Recall: {metrics["recall"]:.4f}')
    logging.info(f'F1 Score: {metrics["f1_score"]:.4f}')
    logging.info(f'AUC-PR: {metrics["auc_pr"]:.4f}')
    logging.info(f'AUC-ROC: {metrics["auc_roc"]:.4f}')
    
    # Log the confusion matrix and related metrics
    logging.info(f'Confusion Matrix:\n{metrics["conf_matrix"]}')
    logging.info(f'True Positives (TP): {metrics["TP"]}')
    logging.info(f'True Negatives (TN): {metrics["TN"]}')
    logging.info(f'False Positives (FP): {metrics["FP"]}')
    logging.info(f'False Negatives (FN): {metrics["FN"]}')


if __name__ == '__main__':
    main()
