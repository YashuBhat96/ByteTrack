import os
import torch
import torch.nn.functional as F
import logging
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from torch.multiprocessing import set_start_method

# Setup logging
log_file_path = '/storage/home/ybr5070/group/homebytes/code/scripts/logs/train_lstm_grid_new02.log'
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
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# Reshape features to fit the sequence length
def reshape_features(features, sequence_length):
    num_sequences = features.size(0) // sequence_length
    features = features[:num_sequences * sequence_length]  # Trim to fit exact multiples of sequence_length
    features = features.view(num_sequences, sequence_length, features.size(1))  # Reshape
    return features

# Training function with validation and early stopping
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs, patience, device):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        logging.info(f'Starting Epoch {epoch + 1}/{num_epochs}')
        model.train()
        running_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        logging.info(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logging.info(f'Early stopping triggered after {epoch + 1} epochs.')
            break

    return best_model_state, best_val_loss

# Main function to perform grid search with CPU-only parallel processing
def main():
    logging.info('Script started')

    # Load datasets
    bite_tensor_path = '/storage/group/klk37/default/homebytes/video/fbs/tensors_train_lstm/consolidated_bite_sequences.pt'
    non_bite_tensor_path = '/storage/group/klk37/default/homebytes/video/fbs/tensors_train_lstm/consolidated_non_bite_sequences.pt'
    val_bite_tensor_path = '/storage/group/klk37/default/homebytes/video/fbs/tensors_val_lstm/consolidated_bite_sequences.pt'
    val_non_bite_tensor_path = '/storage/group/klk37/default/homebytes/video/fbs/tensors_val_lstm/consolidated_non_bite_sequences.pt'

    # Load the bite and non-bite data
    bite_data = torch.load(bite_tensor_path)
    non_bite_data = torch.load(non_bite_tensor_path)
    val_bite_data = torch.load(val_bite_tensor_path)
    val_non_bite_data = torch.load(val_non_bite_tensor_path)

    bite_features, bite_labels = bite_data['features'], bite_data['labels']
    non_bite_features, non_bite_labels = non_bite_data['features'], non_bite_data['labels']
    val_bite_features, val_bite_labels = val_bite_data['features'], val_bite_data['labels']
    val_non_bite_features, val_non_bite_labels = val_non_bite_data['features'], val_non_bite_data['labels']

    # Reshape features to match sequence length
    sequence_length = 51
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
    features = torch.cat((bite_features, non_bite_features), dim=0)
    labels = torch.cat((bite_labels, non_bite_labels), dim=0)
    val_features = torch.cat((val_bite_features, val_non_bite_features), dim=0)
    val_labels = torch.cat((val_bite_labels, val_non_bite_labels), dim=0)

    # Set device to CPU
    device = torch.device('cpu')

    # Initialize the grid search parameters
    lr_list = [0.001]
    batch_size_list = [8]
    epochs_list = [30, 25, 15, 20]
    patience_list = [5, 3]
    hidden_size_list = [128, 256]  # Including the suggested 256
    num_layers_list = [1, 2]
    
    best_val_loss = float('inf')
    best_hyperparams = None

    # Perform grid search
    for lr in lr_list:
        for batch_size in batch_size_list:
            for epochs in epochs_list:
                for patience in patience_list:
                    for hidden_size in hidden_size_list:
                        for num_layers in num_layers_list:
                            logging.info(f'Trying configuration: LR={lr}, Batch Size={batch_size}, Epochs={epochs}, Patience={patience}, Hidden Size={hidden_size}, Num Layers={num_layers}')

                            train_dataset = CombinedDataset(features, labels)
                            val_dataset = CombinedDataset(val_features, val_labels)
                            
                            # Use multiple workers for parallel processing on CPU
                            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=15)
                            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=15)

                            model = LSTMModel(features.size(2), hidden_size, num_layers, 2, device).to(device)
                            criterion = nn.CrossEntropyLoss()
                            optimizer = optim.Adam(model.parameters(), lr=lr)

                            best_model_state, val_loss = train_model(model, criterion, optimizer, train_loader, val_loader, epochs, patience, device)

                            logging.info(f'Validation Loss: {val_loss}')

                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                                best_hyperparams = {
                                    'learning_rate': lr,
                                    'batch_size': batch_size,
                                    'epochs': epochs,
                                    'patience': patience,
                                    'hidden_size': hidden_size,
                                    'num_layers': num_layers
                                }
                                torch.save(best_model_state, '/storage/home/ybr5070/group/homebytes/code/scripts/best_lstm_model_grid_new.pth')

    logging.info(f'Best Hyperparameters: {best_hyperparams}, Best Validation Loss: {best_val_loss}')

if __name__ == '__main__':
    # Necessary for multiprocessing on some platforms like Windows
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    main()
