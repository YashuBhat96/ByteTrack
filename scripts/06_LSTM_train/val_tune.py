import torch
import torch.nn as nn
import logging
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.utils import resample
from sklearn.model_selection import ParameterGrid
import os

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to load and reshape data
def load_data(path, sequence_length=50):
    logging.info(f"Loading data from {path}")
    data = torch.load(path)  # Load the data from a .pt file
    return reshape_tensors(data, sequence_length)

# Function to reshape tensors to the appropriate shape for sequence modeling
def reshape_tensors(data, sequence_length=50):
    logging.info(f"Reshaping data tensors to sequence length {sequence_length}")
    features = data['features']
    labels = data['labels']
    masks = data['masks']
    
    feature_dim = 1280  # Assuming the feature dimension is fixed at 1280
    num_sequences = features.size(0) // sequence_length  # Adjust number of sequences

    if features.numel() != num_sequences * sequence_length * feature_dim:
        raise ValueError("Mismatch in the shape of features for the sequence length")

    # Reshape the features into sequences
    features = features.view(num_sequences, sequence_length, feature_dim)
    labels = labels[:num_sequences]  # One label per sequence
    masks = masks.view(num_sequences, sequence_length)

    return {'features': features, 'labels': labels, 'masks': masks}

# Balanced resampling function
def balanced_resampling(features, labels, masks, undersample_ratio=3.0, label_value=1):
    """
    Apply undersampling to the majority class and random oversampling by duplication to the minority class 
    to balance the dataset at the sequence level.
    """
    # Convert tensors to numpy arrays for easier manipulation
    features_np = features.cpu().numpy()
    labels_np = labels.cpu().numpy()
    masks_np = masks.cpu().numpy()

    # Separate the sequences by class
    minority_class_indices = np.where(labels_np == label_value)[0]  # Class 1 (bites)
    majority_class_indices = np.where(labels_np != label_value)[0]  # Class 0 (non-bites)

    logging.info(f"Original Minority class size (bites): {len(minority_class_indices)}, "
                 f"Majority class size (non-bites): {len(majority_class_indices)}")

    # --- Step 1: Undersample the majority class ---
    majority_class_undersampled_indices = resample(
        majority_class_indices,
        replace=False,  # No replacement for undersampling
        n_samples=int(len(minority_class_indices) * undersample_ratio),  # Undersample majority class
        random_state=42
    )

    # Combine the undersampled majority class with the original minority class
    combined_indices = np.hstack((minority_class_indices, majority_class_undersampled_indices))
    np.random.shuffle(combined_indices)

    # Extract the undersampled data
    features_undersampled = features_np[combined_indices]
    labels_undersampled = labels_np[combined_indices]
    masks_undersampled = masks_np[combined_indices]

    logging.info(f"After undersampling, dataset size: {len(labels_undersampled)}")

    # --- Step 2: Oversample the minority class by duplication ---
    oversampled_minority_indices = resample(
        minority_class_indices,
        replace=True,  # With replacement for oversampling
        n_samples=len(majority_class_undersampled_indices),  # Match the size of the majority class
        random_state=42
    )

    # Combine the oversampled minority class with the undersampled majority class
    combined_indices_final = np.hstack((majority_class_undersampled_indices, oversampled_minority_indices))
    np.random.shuffle(combined_indices_final)

    # Extract the final balanced dataset
    final_features = features_np[combined_indices_final]
    final_labels = labels_np[combined_indices_final]
    final_masks = masks_np[combined_indices_final]

    logging.info(f"Final dataset size after random oversampling and undersampling: {len(final_labels)}")

    # Convert back to tensors
    return torch.tensor(final_features, dtype=torch.float32), \
           torch.tensor(final_labels, dtype=torch.long), \
           torch.tensor(final_masks, dtype=torch.float32)

# LSTM model without attention
class LSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes=1, bidirectional=True, dropout=0.4):
        super(LSTMWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=bidirectional, batch_first=True, dropout=dropout)
        self.attention = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, 1)
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        h_lstm, _ = self.lstm(x)
        
        # Attention mechanism
        attn_weights = torch.softmax(self.attention(h_lstm), dim=1)
        h_lstm_attended = torch.sum(attn_weights * h_lstm, dim=1)

        h_lstm_attended = self.dropout(h_lstm_attended)
        output = self.fc(h_lstm_attended)
        return output

class CombinedTemporalFocalBCELoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, temporal_weight=0.8, context_window=5, bce_weight=0.5, focal_weight=0.5):
        super(CombinedTemporalFocalBCELoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.temporal_weight = temporal_weight
        self.context_window = context_window
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, outputs, targets, epoch=None):
        # Adjust alpha and gamma over time
        if epoch:
            self.alpha = min(0.5, 0.25 + (epoch * 0.01))  # Example dynamic alpha adjustment
            self.gamma = min(3.0, 2.0 + (epoch * 0.05))  # Example dynamic gamma adjustment
        
        # Compute BCE loss
        bce_loss = self.bce_loss(outputs, targets.float())
        probs = torch.sigmoid(outputs)
        
        # Compute Focal Loss
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_loss = -self.alpha * (1 - pt) ** self.gamma * torch.log(pt + 1e-8)

        # Combine BCE and Focal Loss with weights
        combined_loss = self.bce_weight * bce_loss + self.focal_weight * focal_loss

        # Temporal penalty
        for i in range(1, len(targets)):
            if targets[i] == 1 and any(targets[max(0, i - self.context_window):i] == 1):
                combined_loss[i] *= self.temporal_weight

        return combined_loss.mean()


# Temporal smoothing function using a rolling window majority voting
def temporal_smoothing(predictions, alpha=0.1):
    """
    Apply exponential smoothing over the predictions to reduce noise.
    
    Args:
        predictions (np.array): Array of predicted labels (0 or 1).
        alpha (float): Smoothing factor (0 < alpha <= 1).
    
    Returns:
        np.array: Smoothed predictions.
    """
    smoothed_predictions = np.zeros_like(predictions, dtype=float)
    smoothed_predictions[0] = predictions[0]  # Initialize with the first prediction
    
    for i in range(1, len(predictions)):
        smoothed_predictions[i] = alpha * predictions[i] + (1 - alpha) * smoothed_predictions[i - 1]
    
    # Apply a threshold to binarize
    return (smoothed_predictions > 0.5).astype(int)


# Function to find the best threshold using F1 score
def find_best_threshold(all_outputs, all_labels):
    thresholds = np.arange(0.3, 0.7, 0.05)  # Range of thresholds to test
    best_f1 = 0
    best_threshold = 0.4
    for threshold in thresholds:
        preds = (all_outputs > threshold).astype(int)
        f1 = f1_score(all_labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
                
    logging.info(f"Best threshold found: {best_threshold:.2f}, Best F1 score: {best_f1:.4f}")  # Log the best threshold
    return best_threshold, best_f1


# Function to evaluate the model and find the best threshold
def evaluate_model_with_threshold_search(model, loader, criterion, device, smoothing_window=10):
    model.eval()
    val_loss = 0.0
    all_outputs = []  # Initialize for storing outputs
    all_labels = []   # Initialize for storing ground truth labels

    with torch.no_grad():
        for inputs, labels, masks in loader:
            inputs, labels, masks = inputs.to(device), labels.to(device), masks.to(device)
            outputs = model(inputs, masks).squeeze()
            loss = criterion(outputs, labels.float())
            val_loss += loss.item()

            # Collect sigmoid outputs and labels
            all_outputs.extend(torch.sigmoid(outputs).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert lists to numpy arrays
    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)

    # Find the best threshold
    best_threshold, best_f1 = find_best_threshold(all_outputs, all_labels)

    # Make predictions based on the best threshold
    preds = (all_outputs > best_threshold).astype(int)
    smoothed_preds = temporal_smoothing(preds, alpha=0.1)  # Adjust alpha as needed

    val_f1 = f1_score(all_labels, smoothed_preds)
    val_precision, val_recall, _, _ = precision_recall_fscore_support(all_labels, smoothed_preds, pos_label=1, average='binary')
    val_auc_roc = roc_auc_score(all_labels, all_outputs)
    val_auc_pr = average_precision_score(all_labels, all_outputs)
    val_cm = confusion_matrix(all_labels, smoothed_preds)

    return val_loss / len(loader), val_f1, val_precision, val_recall, val_auc_roc, val_auc_pr, val_cm

# Optimizer and scheduler setup
def get_optimizer_and_scheduler(model, initial_lr, weight_decay=5e-5, warmup_steps=5, total_steps=40):
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


# Function to perform grid search over hyperparameters
# Function to perform grid search over hyperparameters
def grid_search(train_dataset, val_dataset, device):
    # Define the parameter grid with two options for each
    param_grid = {
    'learning_rate': [5e-5],  # Keep learning rate fixed for now
    'dropout': [0.4],
    'alpha': [0.25, 0.3],  # Reduced options for alpha
    'gamma': [2.0, 2.5],  # Reduced options for gamma
    'temporal_weight': [0.8],
    'epochs': [40],
    'patience': [5],  # Moderate patience for early stopping
    'weight_decay': [1e-4],
    'batch_size': [128],
    'undersample_ratio': [2.0, 2.5, 3.0]  # Different undersampling ratios
    }


    grid = ParameterGrid(param_grid)
    
    best_f1 = 0.0
    best_config = None
    best_model_state = None  # To store the best model state

    for config in grid:
        logging.info(f"Testing configuration: {config}")
        
        # Adjust the balanced resampling function call to include the undersample_ratio
        train_features_resampled, train_labels_resampled, train_masks_resampled = balanced_resampling(
            train_data['features'], train_data['labels'], train_data['masks'], undersample_ratio=config['undersample_ratio']
        )

        # Create the balanced training dataset and dataloader
        train_dataset = TensorDataset(train_features_resampled, train_labels_resampled, train_masks_resampled)
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=12)
        
        # Initialize a new model for each configuration
        model = LSTMWithAttention(input_size=1280, hidden_size=256, num_layers=3, bidirectional=True, dropout=config['dropout']).to(device)

        # Create the DataLoader for validation set
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=12)
        
        # Train model with current configuration
        val_f1 = train_model_grid_search(model, train_loader, val_loader, config, device)
        
        # Check if the validation F1 score is the best seen so far
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_config = config
            
            # Save the model state for the best configuration
            best_model_state = model.state_dict()  # Save the current model's state

    logging.info(f"Best configuration: {best_config}, Best F1 score: {best_f1:.4f}")
    
    # Save the best overall model at the end of the grid search
    if best_model_state is not None:
        model_path = os.path.join("/storage/group/klk37/default/homebytes/code", "best_overall_model.pth")
        torch.save(best_model_state, model_path)
        logging.info(f"Saved the best overall model to {model_path}")

    return best_config, best_f1


def train_model_grid_search(model, train_dataset, val_dataset, config, device):
    learning_rate = config['learning_rate']
    dropout = config['dropout']
    alpha = config['alpha']
    gamma = config['gamma']
    temporal_weight = config['temporal_weight']
    weight_decay = config['weight_decay']
    epochs = config['epochs']

    criterion = CombinedTemporalFocalBCELoss(alpha=alpha, gamma=gamma, temporal_weight=temporal_weight).to(device)
    optimizer, scheduler = get_optimizer_and_scheduler(model, learning_rate, weight_decay)

    best_val_f1 = 0.0
    early_stopping_counter = 0

    # Directory to save the model
    save_dir = "/storage/group/klk37/default/homebytes/code"
    os.makedirs(save_dir, exist_ok=True)

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()

            train_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.3).long().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        train_f1 = f1_score(all_labels, all_preds)

        # Evaluate on validation set
        val_loss, val_f1, val_precision, val_recall, val_auc_roc, val_auc_pr, val_cm = evaluate_model_with_threshold_search(
            model, val_loader, criterion, device
        )

        logging.info(f"Epoch {epoch + 1}/{epochs}: "
                     f"Train Loss: {train_loss / len(train_loader):.4f}, "
                     f"Train F1: {train_f1:.4f}, "
                     f"Val Loss: {val_loss:.4f}, "
                     f"Val F1: {val_f1:.4f}, "
                     f"Val Precision: {val_precision:.4f}, "
                     f"Val Recall: {val_recall:.4f}, "
                     f"Val AUC-ROC: {val_auc_roc:.4f}, "
                     f"Val AUC-PR: {val_auc_pr:.4f}")

        scheduler.step(val_loss)

        # Check if this is the best validation F1 score and save the model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            early_stopping_counter = 0

            # Save the best model as a .pth file
            model_path = os.path.join(save_dir, f"best_model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), model_path)
            logging.info(f"Saved the best model to {model_path}")
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= config['patience']:
            logging.info("Early stopping triggered.")
            break

    return best_val_f1

# Load the training and validation data
train_path = '/storage/group/klk37/default/homebytes/video/fbs/tensors/consolidated_sequences_train.pt'  # Adjust the path
val_path = '/storage/group/klk37/default/homebytes/video/fbs/tensors/consolidated_sequences_val.pt'  # Adjust the path

# Step 1: Load the training and validation data
train_data = load_data(train_path)
val_data = load_data(val_path)

# Step 2: Perform balanced resampling
train_features_resampled, train_labels_resampled, train_masks_resampled = balanced_resampling(
    train_data['features'], train_data['labels'], train_data['masks'], undersample_ratio=3.0
)

# Step 3: Create the balanced training dataset and dataloader
train_dataset = TensorDataset(train_features_resampled, train_labels_resampled, train_masks_resampled)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=12)

# Step 4: Prepare validation data (no resampling)
val_features = val_data['features']
val_labels = val_data['labels']
val_masks = val_data['masks']

# Step 5: Create validation dataset and dataloader
val_dataset = TensorDataset(val_features, val_labels, val_masks)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=12)

# Step 6: Perform grid search
best_config, best_f1 = grid_search(train_loader, val_loader, device)

# Step 7: Log the best configuration and F1 score
logging.info(f"Best configuration: {best_config}, Best F1: {best_f1}")
