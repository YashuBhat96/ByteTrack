import os
import torch
import numpy as np
import logging
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from itertools import product

# Setup logging
log_file_path = '/storage/home/ybr5070/group/homebytes/code/scripts/logs/train_cv_latest_1.log'
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Custom Dataset
class CombinedDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# LSTM Model with Dropout
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device, dropout_rate=0.5, bidirectional=False):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # Taking the output of the last time step
        out = self.fc(out)
        return out

# Reshape features to fit the sequence length
def reshape_features(features, sequence_length):
    num_sequences = features.size(0) // sequence_length
    features = features[:num_sequences * sequence_length]  # Trim to fit exact multiples of sequence_length
    features = features.view(num_sequences, sequence_length, features.size(1))  # Reshape
    return features

# Training function
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs, device):
    best_model = None
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_f1s = []
    val_f1s = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0
        all_train_labels = []
        all_train_preds = []

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            all_train_labels.extend(labels.cpu().numpy())
            all_train_preds.extend(predicted.cpu().numpy())

        epoch_train_loss = running_train_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        train_f1 = f1_score(all_train_labels, all_train_preds, average='weighted')

        train_losses.append(epoch_train_loss)
        train_accuracies.append(train_accuracy)
        train_f1s.append(train_f1)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_val_labels = []
        all_val_preds = []

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(predicted.cpu().numpy())

        epoch_val_loss = running_val_loss / len(val_loader)
        val_accuracy = correct_val / total_val
        val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted')

        val_losses.append(epoch_val_loss)
        val_accuracies.append(val_accuracy)
        val_f1s.append(val_f1)

        logging.info(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}, Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}')

        # Save the best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model = model.state_dict()

    return best_model, train_losses, val_losses, train_accuracies, val_accuracies, train_f1s, val_f1s

# Function to plot and save the metrics
def plot_metrics(train_metrics, val_metrics, metric_name, save_path):
    plt.figure()
    plt.plot(train_metrics, label=f'Train {metric_name}')
    plt.plot(val_metrics, label=f'Val {metric_name}')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.title(f'Train vs Val {metric_name}')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

# Main function with Grid Search
def main():
    logging.info('Script started')

    sequence_length = 51  # Define the sequence length

    # Load training datasets
    bite_tensor_path = '/storage/group/klk37/default/homebytes/video/fbs/tensors_train_lstm_new/consolidated_bite_sequences.pt'
    non_bite_tensor_path = '/storage/group/klk37/default/homebytes/video/fbs/tensors_train_lstm_new/consolidated_non_bite_sequences.pt'

    bite_data = torch.load(bite_tensor_path)
    non_bite_data = torch.load(non_bite_tensor_path)

    bite_features, bite_labels = bite_data['features'], bite_data['labels']
    non_bite_features, non_bite_labels = non_bite_data['features'], non_bite_data['labels']

    # Reshape the features to match the sequence length
    bite_features = reshape_features(bite_features, sequence_length)
    non_bite_features = reshape_features(non_bite_features, sequence_length)

    # Ensure that labels match the number of sequences
    bite_labels = bite_labels[:len(bite_features)]
    non_bite_labels = non_bite_labels[:len(non_bite_features)]

    # Concatenate features and labels for training
    features = torch.cat((bite_features, non_bite_features), dim=0)
    labels = torch.cat((bite_labels, non_bite_labels), dim=0)

    # Parameters
    input_size = features.size(2)
    num_classes = 2

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info(f"Using device: {device}")

    param_grid = {
        'batch_size': [32, 64],
        'learning_rate': [0.001],
        'num_epochs': [10, 15],
        'hidden_size': [128, 256, 512],
        'num_layers': [2,3,4],
        'dropout_rate': [0.5],
        'bidirectional': [True]
    }

    best_model = None
    best_val_loss = float('inf')  # Track the best validation loss across all combinations

    # Iterate over all combinations of hyperparameters
    for batch_size, learning_rate, num_epochs, hidden_size, num_layers, dropout_rate, bidirectional in product(
        param_grid['batch_size'],
        param_grid['learning_rate'],
        param_grid['num_epochs'],
        param_grid['hidden_size'],
        param_grid['num_layers'],
        param_grid['dropout_rate'],
        param_grid['bidirectional']
    ):
        logging.info(f"Training with batch_size={batch_size}, lr={learning_rate}, epochs={num_epochs}, hidden_size={hidden_size}, num_layers={num_layers}, dropout_rate={dropout_rate}, bidirectional={bidirectional}")

        fold = 0
        for train_index, val_index in kf.split(features):
            fold += 1
            logging.info(f'Starting fold {fold}')

            # Split the data into training (2 folds) and validation (1 fold)
            train_features, val_features = features[train_index], features[val_index]
            train_labels, val_labels = labels[train_index], labels[val_index]

            train_dataset = CombinedDataset(train_features, train_labels)
            val_dataset = CombinedDataset(val_features, val_labels)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

            model = LSTMModel(input_size, hidden_size, num_layers, num_classes, device, dropout_rate, bidirectional).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            best_fold_model, train_losses, val_losses, train_accuracies, val_accuracies, train_f1s, val_f1s = train_model(
                model, criterion, optimizer, train_loader, val_loader, num_epochs, device)

            # Update the best model if current fold's validation loss is lower
            if min(val_losses) < best_val_loss:
                best_val_loss = min(val_losses)
                best_model = best_fold_model

            # Save model and plot metrics for current combination
            torch.save(best_model, f'/storage/home/ybr5070/group/homebytes/code/scripts/lstm_model_fold_{fold}_bs{batch_size}_lr{learning_rate}_hs{hidden_size}_nl{num_layers}_dr{dropout_rate}_bi{bidirectional}.pth')
            plot_metrics(train_losses, val_losses, 'Loss', f'/storage/home/ybr5070/group/homebytes/code/scripts/loss_fold_{fold}_bs{batch_size}_lr{learning_rate}.png')
            plot_metrics(train_accuracies, val_accuracies, 'Accuracy', f'/storage/home/ybr5070/group/homebytes/code/scripts/accuracy_fold_{fold}_bs{batch_size}_lr{learning_rate}.png')
            plot_metrics(train_f1s, val_f1s, 'F1 Score', f'/storage/home/ybr5070/group/homebytes/code/scripts/f1_fold_{fold}_bs{batch_size}_lr{learning_rate}.png')

    logging.info('Script finished')

if __name__ == '__main__':
    main()
