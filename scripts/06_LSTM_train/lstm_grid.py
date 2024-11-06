import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
import logging
import gc  # For garbage collection
import multiprocessing
from torch.multiprocessing import set_start_method
import matplotlib.pyplot as plt

# Setting multiprocessing start method for compatibility
try:
    set_start_method('spawn')
except RuntimeError:
    pass

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Custom Dataset
class CombinedDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Define LSTM model class with Attention, LayerNorm, and Dropout
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate, bidirectional, device):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.device = device
        self.num_directions = 2 if bidirectional else 1

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, 
                            dropout=dropout_rate, bidirectional=bidirectional)
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_size * self.num_directions)
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        # Fully connected layer
        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)

        # Attention mechanism
        self.attention = Attention(hidden_size * self.num_directions)

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(self.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Apply layer normalization
        out = self.layer_norm(out)

        # Apply attention mechanism
        out, _ = self.attention(out)

        # Apply dropout and fully connected layer
        out = self.dropout(out)
        out = self.fc(out)
        return out

# Attention Mechanism
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, lstm_output):
        # Compute attention weights
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        # Compute context vector as the weighted sum of LSTM outputs
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector, attention_weights

# Focal Loss Definition
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)  # pt is the predicted probability for the true class
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return torch.mean(F_loss)

# Training function with per-class metrics (class 1 metrics only)
from sklearn.metrics import confusion_matrix
import seaborn as sns


# Function to plot confusion matrix
def plot_confusion_matrix(cm, epoch, mode='Train'):
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, square=True)
    plt.title(f'Confusion Matrix (Epoch {epoch+1}) - {mode} Set')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Training function with per-class metrics (class 1 metrics only) and confusion matrix
def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs, patience, device, threshold):
    best_f1 = 0
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch+1}/{num_epochs} started.")
        model.train()

        running_train_loss = 0.0
        total_train_predictions, total_train_labels = [], []

        # Training loop
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            total_train_predictions.extend(preds.cpu().numpy())
            total_train_labels.extend(labels.cpu().numpy())

        # Learning rate scheduling
        scheduler.step()

        # Compute training metrics (class 1 only)
        avg_train_loss = running_train_loss / len(train_loader)
        train_precision = precision_score(total_train_labels, total_train_predictions, average=None)
        train_recall = recall_score(total_train_labels, total_train_predictions, average=None)
        train_f1 = f1_score(total_train_labels, total_train_predictions, average=None)

        # Log metrics only for class 1 (bite)
        logging.info(f"Epoch {epoch+1}/{num_epochs} - Training Metrics (Class 1 - Bite):")
        logging.info(f"Precision: {train_precision[1]:.4f}, Recall: {train_recall[1]:.4f}, F1: {train_f1[1]:.4f}")

        # Confusion matrix for training data
        cm_train = confusion_matrix(total_train_labels, total_train_predictions)
        plot_confusion_matrix(cm_train, epoch, mode='Train')

        # Validation
        val_loss, val_precision, val_recall, val_f1, val_auc_pr, val_auc_roc, cm_val = validate_model(
            model, val_loader, criterion, device, threshold
        )

        logging.info(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                     f'AUC-PR: {val_auc_pr:.4f}, AUC-ROC: {val_auc_roc:.4f}')

        # Log validation metrics only for class 1 (bite)
        logging.info(f"Epoch {epoch+1}/{num_epochs} - Validation Metrics (Class 1 - Bite):")
        logging.info(f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")

        # Plot confusion matrix for validation data
        plot_confusion_matrix(cm_val, epoch, mode='Val')

        # Early stopping based on validation F1 score
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break

    return best_model_state, best_f1



# Validation function to compute class 1 metrics
# Validation function to compute class 1 metrics and confusion matrix
def validate_model(model, val_loader, criterion, device, threshold):
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
            probabilities = torch.softmax(outputs, dim=1)[:, 1]  # Probability for class 1 (bite)
            preds = (probabilities > threshold).int()  # Dynamic threshold
            
            val_predictions.extend(preds.cpu().numpy())
            val_labels_list.extend(val_labels_batch.cpu().numpy())
            val_probabilities.extend(probabilities.cpu().numpy())

    # Calculate class 1 metrics
    weighted_f1 = f1_score(val_labels_list, val_predictions, average=None)[1]
    weighted_precision = precision_score(val_labels_list, val_predictions, average=None)[1]
    weighted_recall = recall_score(val_labels_list, val_predictions, average=None)[1]
    
    # AUC-PR and AUC-ROC scores
    auc_pr = average_precision_score(val_labels_list, val_probabilities)
    auc_roc = roc_auc_score(val_labels_list, val_probabilities)

    # Compute confusion matrix
    cm_val = confusion_matrix(val_labels_list, val_predictions)

    return running_val_loss / len(val_loader), weighted_precision, weighted_recall, weighted_f1, auc_pr, auc_roc, cm_val


# Helper function to reshape features for LSTM
def reshape_features(features, sequence_length):
    num_sequences = features.size(0) // sequence_length
    features = features[:num_sequences * sequence_length]  # Trim to fit exact multiples of sequence_length
    features = features.view(num_sequences, sequence_length, features.size(1))  # Reshape
    return features

# Main function for model training with grid search
def main():
    # Dataset paths
    bite_tensor_path = '/storage/group/klk37/default/homebytes/video/fbs/tensors_train_lstm_new_1/consolidated_bite_sequences.pt'
    non_bite_tensor_path = '/storage/group/klk37/default/homebytes/video/fbs/tensors_train_lstm_new_1/consolidated_non_bite_sequences.pt'
    val_bite_tensor_path = '/storage/group/klk37/default/homebytes/video/fbs/tensors_val_lstm_new_1/consolidated_bite_sequences.pt'
    val_non_bite_tensor_path = '/storage/group/klk37/default/homebytes/video/fbs/tensors_val_lstm_new_1/consolidated_non_bite_sequences.pt'

    # Load the bite and non-bite data
    logging.info("Loading datasets...")
    bite_data = torch.load(bite_tensor_path)
    non_bite_data = torch.load(non_bite_tensor_path)
    val_bite_data = torch.load(val_bite_tensor_path)
    val_non_bite_data = torch.load(val_non_bite_tensor_path)

    # Extract features and labels
    bite_features, bite_labels = bite_data['features'], bite_data['labels']
    non_bite_features, non_bite_labels = non_bite_data['features'], non_bite_data['labels']
    val_bite_features, val_bite_labels = val_bite_data['features'], val_bite_data['labels']
    val_non_bite_features, val_non_bite_labels = val_non_bite_data['features'], val_non_bite_data['labels']

    # Reshape features to match sequence length
    sequence_length = 50
    bite_features = reshape_features(bite_features, sequence_length)
    non_bite_features = reshape_features(non_bite_features, sequence_length)
    val_bite_features = reshape_features(val_bite_features, sequence_length)
    val_non_bite_features = reshape_features(val_non_bite_features, sequence_length)

    # Ensure labels match the number of sequences
    bite_labels = bite_labels[:len(bite_features)]
    non_bite_labels = non_bite_labels[:len(non_bite_features)]
    val_bite_labels = val_bite_labels[:len(val_bite_features)]
    val_non_bite_labels = val_non_bite_labels[:len(val_non_bite_features)]

    # Concatenate features and labels for training and validation sets
    train_features = torch.cat((bite_features, non_bite_features), dim=0)
    train_labels = torch.cat((bite_labels, non_bite_labels), dim=0)
    val_features = torch.cat((val_bite_features, val_non_bite_features), dim=0)
    val_labels = torch.cat((val_bite_labels, val_non_bite_labels), dim=0)

    # Set device to CPU only (since GPU is not available)
    device = torch.device('cpu')

    # Hyperparameter grid
    param_grid = {
        'num_epochs': [30, 40],  # Update epochs
        'patience': [5, 7],  # Update patience
        'class_weights': [[1.0, 15.0]],  # Updated weights
        'dropout_rates': [0.3, 0.5],  # Added grid search for dropout
        'threshold': [0.2],  # Added grid search for thresholds
        'loss_functions': ['weighted_cross_entropy', 'focal_loss']  # Added loss functions
    }

    # Fixed parameters
    batch_size = 64
    lr = 0.001
    hidden_size = 512
    num_layers = 4
    bidirectional = True

    best_f1 = 0
    best_hyperparams = None
    best_model_state = None

    # Perform grid search on epochs, patience, class weights, thresholds, dropout, and loss functions
    for num_epochs in param_grid['num_epochs']:
        for patience in param_grid['patience']:
            for class_weights in param_grid['class_weights']:
                for dropout_rate in param_grid['dropout_rates']:
                    for threshold in param_grid['threshold']:
                        for loss_func in param_grid['loss_functions']:
                            logging.info(f'Testing configuration: Epochs={num_epochs}, Patience={patience}, '
                                         f'Class Weights={class_weights}, Dropout={dropout_rate}, '
                                         f'Loss={loss_func}, Threshold={threshold}')

                            # Define the datasets and data loaders
                            train_dataset = CombinedDataset(train_features, train_labels)
                            val_dataset = CombinedDataset(val_features, val_labels)

                            # Data loaders
                            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

                            # Instantiate the model with current dropout rate
                            model = LSTMModel(
                                input_size=train_features.size(2),
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                num_classes=2,
                                dropout_rate=dropout_rate,  # Use grid-searched dropout rate
                                bidirectional=bidirectional,
                                device=device
                            ).to(device)

                            # Define loss function based on grid search
                            if loss_func == 'weighted_cross_entropy':
                                criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device))
                            elif loss_func == 'focal_loss':
                                criterion = FocalLoss(alpha=class_weights[1], gamma=2.0)  # Can tweak gamma if needed

                            # Define optimizer and scheduler
                            optimizer = optim.Adam(model.parameters(), lr=lr)
                            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

                            # Train the model
                            model_state, f1 = train_model(
                                model, criterion, optimizer, scheduler, train_loader, val_loader,
                                num_epochs, patience, device, threshold
                            )

                            if f1 > best_f1:
                                best_f1 = f1
                                best_hyperparams = {
                                    'num_epochs': num_epochs,
                                    'patience': patience,
                                    'class_weights': class_weights,
                                    'dropout_rate': dropout_rate,
                                    'loss_function': loss_func,
                                    'threshold': threshold
                                }
                                best_model_state = model_state

                            # Run garbage collection after each grid search iteration
                            gc.collect()

    # Save best model state
    torch.save(best_model_state, '/storage/home/ybr5070/group/homebytes/code/scripts/best_lstm_model_f1_latest.pth')
    logging.info(f'Best Hyperparameters: {best_hyperparams}, Best F1 Score: {best_f1}')


if __name__ == '__main__':
    main()
