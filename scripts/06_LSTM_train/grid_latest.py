import torch
import torch.nn as nn
import logging
import numpy as np
import os
import time
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_fscore_support, roc_auc_score, average_precision_score, confusion_matrix
from itertools import product
from concurrent.futures import ThreadPoolExecutor

# Initialize logging at the start of the script
logging.basicConfig(
    filename='training_log.log',  # Log file to store logs
    level=logging.INFO,  # Log all INFO and above levels
    format='%(asctime)s - %(levelname)s - %(message)s',  # Include timestamp, level, and message
    datefmt='%Y-%m-%d %H:%M:%S'  # Format for timestamp
)

# Start of the program
logging.info("Script started.")


# LSTM with Simple Attention
class LSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes=1, bidirectional=True, dropout=0.5):
        super(LSTMWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            bidirectional=bidirectional, batch_first=True, dropout=dropout)
        
        self.attention_weight = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, 1, bias=False)
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, lstm_output, mask=None):
        attention_scores = self.attention_weight(lstm_output)
        if mask is not None:
            attention_scores = attention_scores.squeeze(-1)
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
            attention_scores = attention_scores.unsqueeze(-1)

        attention_weights = torch.softmax(attention_scores, dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector

    def forward(self, x, mask=None):
        h_lstm, _ = self.lstm(x)
        context = self.attention(h_lstm, mask)
        context = self.dropout(context)
        output = self.fc(context)
        return output


# Reshape tensors based on sequence length
def reshape_tensors(data, sequence_length=50):
    logging.info(f"Reshaping data tensors to sequence length {sequence_length}")
    features = data['features']
    labels = data['labels']
    masks = data['masks']
    
    num_sequences = features.size(0) // sequence_length
    feature_dim = features.size(1)
    features = features.view(num_sequences, sequence_length, feature_dim)
    labels = labels[:num_sequences]
    masks = masks.view(num_sequences, sequence_length)
    
    logging.info(f"Reshaping completed: {features.shape}")
    return {'features': features, 'labels': labels, 'masks': masks}


# Load datasets
def load_data(train_path, batch_size=64, sequence_length=50, num_workers=16):
    logging.info(f"Loading training data from {train_path}")
    train_data = torch.load(train_path)
    train_data = reshape_tensors(train_data, sequence_length)
    return train_data


# Function to create the loss function
def get_loss_function(pos_weight, device):
    logging.info(f"Creating BCEWithLogitsLoss with pos_weight={pos_weight}.")
    pos_weight = torch.tensor([pos_weight], device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


# Optimizer and scheduler with weight decay
def get_optimizer_and_scheduler(model, learning_rate, weight_decay=1e-3):
    logging.info(f"Setting up optimizer with lr={learning_rate}, weight_decay={weight_decay}.")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1, verbose=True)
    return optimizer, scheduler


# Function to tune the threshold dynamically
def find_best_threshold(model, val_loader, criterion, thresholds, device):
    logging.info("Tuning threshold to find the one with best F1 score.")
    best_threshold = 0.5
    best_f1 = 0.0

    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels, masks in val_loader:
            inputs, labels, masks = inputs.to(device), labels.to(device), masks.to(device)
            outputs = model(inputs, masks).squeeze()
            probs = torch.sigmoid(outputs).cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    for threshold in thresholds:
        preds = (np.array(all_probs) > threshold).astype(int)
        f1 = f1_score(all_labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    logging.info(f"Best threshold: {best_threshold} with F1: {best_f1:.4f}")
    return best_threshold

# Focal Loss implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-bce_loss)  # Prevents nans when probability is 0
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Function to dynamically select loss function
def get_loss_function(loss_type, pos_weight=None, alpha=1, gamma=2, device=None):
    if loss_type == "weighted_bce":
        logging.info(f"Creating BCEWithLogitsLoss with pos_weight={pos_weight}.")
        pos_weight = torch.tensor([pos_weight], device=device)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss_type == "focal":
        logging.info(f"Creating Focal Loss with alpha={alpha}, gamma={gamma}.")
        return FocalLoss(alpha=alpha, gamma=gamma)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

# Modify the train_model function to include dynamic loss selection
def train_model(model, train_loader, val_loader, epochs, patience, learning_rate, dropout, pos_weight, device, loss_type, alpha=1, gamma=2):
    logging.info(f"Training model with lr={learning_rate}, dropout={dropout}, pos_weight={pos_weight}, loss_type={loss_type}.")
    criterion = get_loss_function(loss_type=loss_type, pos_weight=pos_weight, alpha=alpha, gamma=gamma, device=device)
    optimizer, scheduler = get_optimizer_and_scheduler(model, learning_rate, weight_decay=1e-3)
    
    best_val_f1 = 0.0
    early_stopping_counter = 0
    best_threshold = 0.5
    thresholds_to_try = np.arange(0.4, 0.7, 0.05)

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
            preds = (torch.sigmoid(outputs) > best_threshold).long().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        # Calculate train F1, Precision, Recall
        train_f1 = f1_score(all_labels, all_preds)
        train_precision = precision_score(all_labels, all_preds)
        train_recall = recall_score(all_labels, all_preds)
        train_conf_matrix = confusion_matrix(all_labels, all_preds)

        # Evaluate on validation data
        val_loss, val_f1, val_precision, val_recall, val_auc_roc, val_auc_pr, val_cm = evaluate_model(
            model, val_loader, criterion, best_threshold, device)

        # Update best threshold based on validation set
        best_threshold = find_best_threshold(model, val_loader, criterion, thresholds_to_try, device)

        # Log epoch details
        logging.info(
            f"Epoch {epoch + 1}/{epochs}: "
            f"Train Loss: {train_loss/len(train_loader):.4f}, "
            f"Train F1: {train_f1:.4f}, Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, "
            f"Train Confusion Matrix: {train_conf_matrix}, "
            f"Val F1: {val_f1:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, "
            f"Val Confusion Matrix: {val_cm}, "
            f"Best Threshold: {best_threshold:.2f}"
        )

        scheduler.step(val_f1)

        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            logging.info("Early stopping triggered.")
            break

    return best_val_f1

# Add loss_type to the grid search
loss_types = ["weighted_bce", "focal"]
alpha_values = [1, 2]  # For focal loss, you can tweak these hyperparameters.
gamma_values = [2]  # Typically 2, but you can add more to experiment.

# Modify grid search loop
logging.info("Starting grid search over hyperparameters.")
for dropout, learning_rate, pos_weight, batch_size, weight_decay, patience, loss_type in product(
        dropouts, learning_rates, pos_weights, batch_sizes, weight_decays, patiences, loss_types):
    logging.info(f"Training with dropout={dropout}, lr={learning_rate}, pos_weight={pos_weight}, batch_size={batch_size}, "
                 f"weight_decay={weight_decay}, patience={patience}, loss_type={loss_type}.")
    
    avg_val_f1 = cross_validate_model_parallel(
        model_class=LSTMWithAttention,
        train_data=train_data,
        fold_indices=fold_indices,
        dropout=dropout,
        learning_rate=learning_rate,
        pos_weight=pos_weight,
        device=device,
        loss_type=loss_type
    )
    
    logging.info(f"Avg Validation F1 for dropout={dropout}, lr={learning_rate}, pos_weight={pos_weight}, loss_type={loss_type}: {avg_val_f1:.4f}")
    
    if avg_val_f1 > best_val_f1:
        best_val_f1 = avg_val_f1
        best_hyperparams = {
            'dropout': dropout,
            'learning_rate': learning_rate,
            'pos_weight': pos_weight,
            'batch_size': batch_size,
            'weight_decay': weight_decay,
            'patience': patience,
            'loss_type': loss_type
        }

logging.info(f"Best hyperparameters: {best_hyperparams}, Best Validation F1: {best_val_f1:.4f}")

# Focal Loss implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-bce_loss)  # Prevents nans when probability is 0
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Function to dynamically select loss function
def get_loss_function(loss_type, pos_weight=None, alpha=1, gamma=2, device=None):
    if loss_type == "weighted_bce":
        logging.info(f"Creating BCEWithLogitsLoss with pos_weight={pos_weight}.")
        pos_weight = torch.tensor([pos_weight], device=device)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss_type == "focal":
        logging.info(f"Creating Focal Loss with alpha={alpha}, gamma={gamma}.")
        return FocalLoss(alpha=alpha, gamma=gamma)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

# Modify the train_model function to include dynamic loss selection
def train_model(model, train_loader, val_loader, epochs, patience, learning_rate, dropout, pos_weight, device, loss_type, alpha=1, gamma=2):
    logging.info(f"Training model with lr={learning_rate}, dropout={dropout}, pos_weight={pos_weight}, loss_type={loss_type}.")
    criterion = get_loss_function(loss_type=loss_type, pos_weight=pos_weight, alpha=alpha, gamma=gamma, device=device)
    optimizer, scheduler = get_optimizer_and_scheduler(model, learning_rate, weight_decay=1e-3)
    
    best_val_f1 = 0.0
    early_stopping_counter = 0
    best_threshold = 0.5
    thresholds_to_try = np.arange(0.4, 0.7, 0.05)

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
            preds = (torch.sigmoid(outputs) > best_threshold).long().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        # Calculate train F1, Precision, Recall
        train_f1 = f1_score(all_labels, all_preds)
        train_precision = precision_score(all_labels, all_preds)
        train_recall = recall_score(all_labels, all_preds)
        train_conf_matrix = confusion_matrix(all_labels, all_preds)

        # Evaluate on validation data
        val_loss, val_f1, val_precision, val_recall, val_auc_roc, val_auc_pr, val_cm = evaluate_model(
            model, val_loader, criterion, best_threshold, device)

        # Update best threshold based on validation set
        best_threshold = find_best_threshold(model, val_loader, criterion, thresholds_to_try, device)

        # Log epoch details
        logging.info(
            f"Epoch {epoch + 1}/{epochs}: "
            f"Train Loss: {train_loss/len(train_loader):.4f}, "
            f"Train F1: {train_f1:.4f}, Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, "
            f"Train Confusion Matrix: {train_conf_matrix}, "
            f"Val F1: {val_f1:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, "
            f"Val Confusion Matrix: {val_cm}, "
            f"Best Threshold: {best_threshold:.2f}"
        )

        scheduler.step(val_f1)

        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            logging.info("Early stopping triggered.")
            break

    return best_val_f1

# Add loss_type to the grid search
loss_types = ["weighted_bce", "focal"]
alpha_values = [1, 2]  # For focal loss, you can tweak these hyperparameters.
gamma_values = [2]  # Typically 2, but you can add more to experiment.

# Modify grid search loop
logging.info("Starting grid search over hyperparameters.")
for dropout, learning_rate, pos_weight, batch_size, weight_decay, patience, loss_type in product(
        dropouts, learning_rates, pos_weights, batch_sizes, weight_decays, patiences, loss_types):
    logging.info(f"Training with dropout={dropout}, lr={learning_rate}, pos_weight={pos_weight}, batch_size={batch_size}, "
                 f"weight_decay={weight_decay}, patience={patience}, loss_type={loss_type}.")
    
    avg_val_f1 = cross_validate_model_parallel(
        model_class=LSTMWithAttention,
        train_data=train_data,
        fold_indices=fold_indices,
        dropout=dropout,
        learning_rate=learning_rate,
        pos_weight=pos_weight,
        device=device,
        loss_type=loss_type
    )
    
    logging.info(f"Avg Validation F1 for dropout={dropout}, lr={learning_rate}, pos_weight={pos_weight}, loss_type={loss_type}: {avg_val_f1:.4f}")
    
    if avg_val_f1 > best_val_f1:
        best_val_f1 = avg_val_f1
        best_hyperparams = {
            'dropout': dropout,
            'learning_rate': learning_rate,
            'pos_weight': pos_weight,
            'batch_size': batch_size,
            'weight_decay': weight_decay,
            'patience': patience,
            'loss_type': loss_type
        }

# Evaluate model function
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

from concurrent.futures import ThreadPoolExecutor
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Cross-validation with parallelism (for CPU)
def cross_validate_model_parallel(model_class, train_data, fold_indices, dropout, learning_rate, pos_weight, device, num_folds=3):
    fold_scores = []
    logging.info(f"Starting parallel cross-validation with {num_folds} folds.")

    def train_fold(fold, train_idx, val_idx):
        logging.info(f"Fold {fold + 1}/{num_folds} starting...")

        train_features = train_data['features'][train_idx]
        train_labels = train_data['labels'][train_idx]
        train_masks = train_data['masks'][train_idx]

        val_features = train_data['features'][val_idx]
        val_labels = train_data['labels'][val_idx]
        val_masks = train_data['masks'][val_idx]

        fold_train_dataset = TensorDataset(train_features, train_labels, train_masks)
        fold_val_dataset = TensorDataset(val_features, val_labels, val_masks)

        train_loader = DataLoader(fold_train_dataset, batch_size=32, shuffle=True, num_workers=12)
        val_loader = DataLoader(fold_val_dataset, batch_size=32, shuffle=False, num_workers=12)

        model = model_class(input_size=1280, hidden_size=128, num_layers=2, bidirectional=True, dropout=dropout)
        model.to(device)

        fold_val_f1 = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=30,
            patience=3,
            learning_rate=learning_rate,
            dropout=dropout,
            pos_weight=pos_weight,
            device=device
        )
        logging.info(f"Fold {fold + 1}/{num_folds} F1 Score: {fold_val_f1:.4f}")
        return fold_val_f1

    # Parallelize cross-validation across folds using CPU
    with ThreadPoolExecutor(max_workers=num_folds) as executor:
        futures = [
            executor.submit(train_fold, fold, train_idx, val_idx)
            for fold, (train_idx, val_idx) in enumerate(fold_indices)
        ]

        # Gather the results as they complete
        for future in futures:
            fold_scores.append(future.result())

    return np.mean(fold_scores)

# Load your training data and prepare cross-validation splits
train_path = '/storage/group/klk37/default/homebytes/video/fbs/train_new/consolidated_sequences.pt'  # Specify the correct path
train_data = load_data(train_path, batch_size=64, sequence_length=50)
logging.info("Train data loaded.")

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
fold_indices = list(skf.split(train_data['features'], train_data['labels']))

# Specify hyperparameters
dropouts = [0.3, 0.2]
learning_rates = [0.0001, 0.00005]
pos_weights = [1.0, 2.0, 3.0]
device = torch.device('cpu')  # Since we're using CPU

# Grid search over hyperparameters with cross-validation
best_hyperparams = None
best_val_f1 = 0.0

logging.info("Starting grid search over hyperparameters.")
for dropout, learning_rate, pos_weight in product(dropouts, learning_rates, pos_weights):
    logging.info(f"Training with dropout={dropout}, learning_rate={learning_rate}, pos_weight={pos_weight}.")
    
    avg_val_f1 = cross_validate_model_parallel(
        model_class=LSTMWithAttention,
        train_data=train_data,
        fold_indices=fold_indices,
        dropout=dropout,
        learning_rate=learning_rate,
        pos_weight=pos_weight,
        device=device
    )
    
    logging.info(f"Avg Validation F1 for dropout={dropout}, lr={learning_rate}, pos_weight={pos_weight}: {avg_val_f1:.4f}")
    
    if avg_val_f1 > best_val_f1:
        best_val_f1 = avg_val_f1
        best_hyperparams = {'dropout': dropout, 'learning_rate': learning_rate, 'pos_weight': pos_weight}

logging.info(f"Best hyperparameters: {best_hyperparams}, Best Validation F1: {best_val_f1:.4f}")
