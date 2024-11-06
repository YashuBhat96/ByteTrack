import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, precision_recall_fscore_support, confusion_matrix
from imblearn.over_sampling import SMOTE  
import numpy as np
from itertools import product
import logging
import os
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# LSTM model without attention and with bidirectionality
class LSTMWithoutAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes=1, bidirectional=True, dropout=0.5):
        super(LSTMWithoutAttention, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            bidirectional=bidirectional, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        h_lstm, _ = self.lstm(x)  # LSTM output: [batch_size, sequence_length, hidden_size]
    
        # Check if the mask is provided
        if mask is not None:
            # If mask has only 1 dimension (batch_size), unsqueeze to add sequence length
            if len(mask.shape) == 1:  # Mask is of shape [batch_size]
                # Assuming a fixed sequence length (e.g., 50)
                sequence_length = h_lstm.shape[1]  # Get sequence length from LSTM output
                mask = mask.unsqueeze(1).expand(-1, sequence_length)  # Shape it to [batch_size, sequence_length]
            
            # Ensure mask now has 2 dimensions [batch_size, sequence_length]
            if len(mask.shape) == 2:
                assert mask.shape[0] == h_lstm.shape[0], f"Mask batch size {mask.shape[0]} does not match LSTM output batch size {h_lstm.shape[0]}"
                assert mask.shape[1] == h_lstm.shape[1], f"Mask sequence length {mask.shape[1]} does not match LSTM output sequence length {h_lstm.shape[1]}"
                
                # Expand mask to [batch_size, sequence_length, 1] for broadcasting
                mask = mask.unsqueeze(-1)  # Now [batch_size, sequence_length, 1]
                h_lstm = h_lstm * mask  # Apply mask to zero out padded frames
            else:
                raise ValueError(f"Expected mask to have 2 dimensions, but got {mask.shape}")
        
        h_lstm = self.dropout(h_lstm)  # Apply dropout after masking
        
        # Pooling mechanism to summarize LSTM output (use max/mean/sum pooling instead of attention)
        context = torch.sum(h_lstm, dim=1)  # Sum across the sequence length: [batch_size, hidden_size]
        # Alternative: context = torch.mean(h_lstm, dim=1)  # Use mean pooling

        output = self.fc(context)  # Final output: [batch_size, num_classes]
        
        return output

# Reshape tensors based on sequence length
def reshape_tensors(data, sequence_length=50):
    features = data['features']
    labels = data['labels']
    masks = data['masks']
    
    num_sequences = features.size(0) // sequence_length
    feature_dim = features.size(1)
    features = features.view(num_sequences, sequence_length, feature_dim)
    labels = labels[:num_sequences]
    masks = masks.view(num_sequences, sequence_length)
    
    return {'features': features, 'labels': labels, 'masks': masks}

# Optimized SMOTE in batches
def smote_in_batches(features, labels, batch_size=10000):
    smote = SMOTE(sampling_strategy=0.7, n_jobs=-1)
    num_samples, feature_dim = features.shape
    
    resampled_features = []
    resampled_labels = []
    
    # Process data in batches to save memory
    for i in range(0, num_samples, batch_size):
        end = min(i + batch_size, num_samples)
        batch_features = features[i:end]
        batch_labels = labels[i:end]
        
        # Check if the batch contains more than one class
        unique_classes = np.unique(batch_labels)
        if len(unique_classes) > 1:
            batch_features_resampled, batch_labels_resampled = smote.fit_resample(batch_features, batch_labels)
            resampled_features.append(batch_features_resampled)
            resampled_labels.append(batch_labels_resampled)
        else:
            logging.warning(f"Skipping batch {i}-{end} because it contains only one class: {unique_classes}")
    
    # Concatenate the results of all batches
    if resampled_features:
        resampled_features = np.vstack(resampled_features)
        resampled_labels = np.hstack(resampled_labels)
        return resampled_features, resampled_labels
    else:
        raise ValueError("No valid batches with more than one class were found.")

# Load datasets and apply SMOTE to the minority class, with caching and batch processing
def load_data_with_smote(train_path, val_path, batch_size=64, sequence_length=50, num_workers=16):
    smote_train_path = os.path.join(os.path.dirname(train_path), 'smote_train_data.pt')
    
    if os.path.exists(smote_train_path):
        logging.info(f"Loading SMOTE-processed training data from {smote_train_path}")
        train_data = torch.load(smote_train_path)
    else:
        logging.info(f"Applying SMOTE to training data and saving to {smote_train_path}")
        
        # Load and reshape the original training data
        train_data = torch.load(train_path)
        train_data = reshape_tensors(train_data, sequence_length)  # Use existing reshape logic
        
        num_samples, seq_len, feature_dim = train_data['features'].shape
        train_features_flat = train_data['features'].view(num_samples * seq_len, feature_dim)
        train_labels_flat = train_data['labels'].repeat_interleave(seq_len)

        start_time = time.time()

        # Apply SMOTE in batches
        train_features_resampled, train_labels_resampled = smote_in_batches(
            train_features_flat.cpu().numpy(), train_labels_flat.cpu().numpy(), batch_size=100000
        )
        
        logging.info(f"SMOTE completed in {time.time() - start_time:.2f} seconds")

        # Calculate the exact number of samples needed to fit a multiple of sequence_length
        total_resampled_samples = train_features_resampled.shape[0]
        num_required_samples = (total_resampled_samples // sequence_length) * sequence_length

        # Truncate the resampled features and labels to match the exact required number of samples
        train_features_resampled = train_features_resampled[:num_required_samples]
        train_labels_resampled = train_labels_resampled[:num_required_samples]

        # Reshape the resampled features and labels using the existing reshape_tensors function
        resampled_data = {
            'features': torch.tensor(train_features_resampled, dtype=torch.float32),
            'labels': torch.tensor(train_labels_resampled, dtype=torch.long),
            'masks': torch.ones(num_required_samples // sequence_length * sequence_length)  # Create a new mask
        }

        # Apply reshape_tensors to reshape the resampled data to [num_sequences, seq_len, feature_dim]
        resampled_data = reshape_tensors(resampled_data, sequence_length)

        # Save the SMOTE-processed data for future reuse
        torch.save(resampled_data, smote_train_path)
        logging.info(f"SMOTE-processed training data saved to {smote_train_path}")
    
    # Load and reshape validation data (without SMOTE)
    val_data = torch.load(val_path)
    val_data = reshape_tensors(val_data, sequence_length)

    # Create TensorDatasets
    train_dataset = TensorDataset(train_data['features'], train_data['labels'], train_data['masks'])
    val_dataset = TensorDataset(val_data['features'], val_data['labels'], val_data['masks'])

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader




# Function to create the loss function
def get_loss_function(device):
    pos_weight = torch.tensor([3.0], device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Optimizer and scheduler with weight decay
def get_optimizer_and_scheduler(model, learning_rate, weight_decay=1e-4):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1, verbose=True)  # Mode 'min' since we want to reduce when val_loss stops improving
    return optimizer, scheduler


# Train and evaluate functions with confusion matrix logging
def train_model(model, train_loader, val_loader, epochs, patience, learning_rate, dropout, threshold, device):
    criterion = get_loss_function(device)  # Pass device here
    optimizer, scheduler = get_optimizer_and_scheduler(model, learning_rate)
    
    best_val_f1 = 0.0
    early_stopping_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        all_preds = []
        all_labels = []
        
        for inputs, labels, masks in train_loader:
            inputs, labels, masks = inputs.to(device), labels.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, masks).squeeze()
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = (torch.sigmoid(outputs) > threshold).long().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        # Training metrics
        train_f1 = f1_score(all_labels, all_preds)
        train_precision, train_recall, _, _ = precision_recall_fscore_support(all_labels, all_preds, pos_label=1, average='binary')

        # Evaluate on validation data
        val_loss, val_f1, val_precision, val_recall, val_auc_roc, val_auc_pr, val_cm = evaluate_model(model, val_loader, criterion, threshold, device)

        # Log train and validation metrics
        logging.info(f"Epoch {epoch + 1}/{epochs}: Train Loss: {train_loss/len(train_loader):.4f}, "
                     f"Train F1: {train_f1:.4f}, Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, "
                     f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}, Val Precision: {val_precision:.4f}, "
                     f"Val Recall: {val_recall:.4f}, Val AUC-ROC: {val_auc_roc:.4f}, Val AUC-PR: {val_auc_pr:.4f}")

        # Log confusion matrix for the validation data
        logging.info(f"Validation Confusion Matrix for Epoch {epoch + 1}:\n{val_cm}")

        scheduler.step(val_f1)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            early_stopping_counter = 0
            torch.save(model.state_dict(), f'best_model_dropout_{dropout}_lr_{learning_rate}_threshold_{threshold}.pt')
        else:
            early_stopping_counter += 1
        
        if early_stopping_counter >= patience:
            logging.info("Early stopping triggered.")
            break

    return best_val_f1

def evaluate_model(model, loader, criterion, threshold, device):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels, masks in loader:
            inputs, labels, masks = inputs.to(device), labels.to(device), masks.to(device)
            outputs = model(inputs, masks).squeeze()
            loss = criterion(outputs, labels.float())
            val_loss += loss.item()
            
            preds = (torch.sigmoid(outputs) > threshold).long().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    val_f1 = f1_score(all_labels, all_preds)
    val_precision, val_recall, _, _ = precision_recall_fscore_support(all_labels, all_preds, pos_label=1, average='binary')
    
    val_auc_roc = roc_auc_score(all_labels, all_preds)
    val_auc_pr = average_precision_score(all_labels, all_preds)
    val_cm = confusion_matrix(all_labels, all_preds)

    return val_loss / len(loader), val_f1, val_precision, val_recall, val_auc_roc, val_auc_pr, val_cm

# Grid search for hyperparameters without multiprocessing
dropouts = [0.3, 0.5,0.7]
learning_rates = [0.0001]
thresholds = [0.5, 0.55, 0.6]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

best_hyperparams = None
best_val_f1 = 0.0

for dropout, learning_rate, threshold in product(dropouts, learning_rates, thresholds):
    logging.info(f"Training with dropout={dropout}, learning_rate={learning_rate}, threshold={threshold}")
    
    # Correctly pass the grid-search dropout value to the model instantiation
    model = LSTMWithoutAttention(input_size=1280, hidden_size=128, num_layers=2, bidirectional=True, dropout=dropout)
    model.to(device)
    
    train_loader, val_loader = load_data_with_smote(
        train_path='/storage/group/klk37/default/homebytes/video/fbs/train_new/consolidated_sequences.pt',
        val_path='/storage/group/klk37/default/homebytes/video/fbs/val_new/consolidated_sequences.pt',
        batch_size=32,
        sequence_length=50,
        num_workers=8
    )
    
    val_f1 = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=30,
        patience=3,
        learning_rate=learning_rate,
        dropout=dropout,  # Pass dropout for logging purposes, though it's already set in the model.
        threshold=threshold,
        device=device
    )
    
    logging.info(f"Validation F1 Score with dropout={dropout}, learning_rate={learning_rate}, threshold={threshold}: {val_f1:.4f}")
    
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_hyperparams = (dropout, learning_rate, threshold)

logging.info(f"Best hyperparameters: {best_hyperparams} with F1 score: {best_val_f1:.4f}")
