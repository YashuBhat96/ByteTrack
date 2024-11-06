import os
import torch
import torch.nn.functional as F
import logging
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

# Setup logging
log_file_path = '/storage/home/ybr5070/group/homebytes/code/scripts/logs/train_lstm_overlap_final_trial.log'
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
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# Training function with Early Stopping and Validation Feedback
def train_model_with_early_stopping(model, criterion, optimizer, train_loader, val_loader, num_epochs, device, patience=5):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_f1s = []
    val_f1s = []

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        logging.info(f'Starting Epoch {epoch + 1}/{num_epochs}')
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_loss = running_loss / len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds, average='weighted')

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        train_f1s.append(train_f1)

        logging.info(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}, Train Acc: {train_acc}, Train F1: {train_f1}')

        # Validation step
        model.eval()
        val_running_loss = 0.0
        val_all_preds = []
        val_all_labels = []
        with torch.no_grad():
            for val_features, val_labels in val_loader:
                val_features, val_labels = val_features.to(device), val_labels.to(device)
                val_outputs = model(val_features)
                val_loss = criterion(val_outputs, val_labels)
                val_running_loss += val_loss.item()
                val_all_preds.extend(torch.argmax(val_outputs, dim=1).cpu().numpy())
                val_all_labels.extend(val_labels.cpu().numpy())

        val_loss = val_running_loss / len(val_loader)
        val_acc = accuracy_score(val_all_labels, val_all_preds)
        val_f1 = f1_score(val_all_labels, val_all_preds, average='weighted')

        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_f1s.append(val_f1)

        logging.info(f'Epoch {epoch + 1}/{num_epochs}, Val Loss: {val_loss}, Val Acc: {val_acc}, Val F1: {val_f1}')

        # Early stopping check based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the model checkpoint
            torch.save(model.state_dict(), '/storage/home/ybr5070/group/homebytes/code/scripts/lstm_model_best.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logging.info(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # Load the best model before early stopping
    model.load_state_dict(torch.load('/storage/home/ybr5070/group/homebytes/code/scripts/lstm_model_best.pth'))

    return train_losses, val_losses, train_accuracies, val_accuracies, train_f1s, val_f1s

# Plot metrics
def plot_metrics(train_metrics, val_metrics, metric_name, output_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_metrics, label=f'Train {metric_name}')
    plt.plot(val_metrics, label=f'Val {metric_name}')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    plt.title(f'{metric_name} over Epochs')
    plt.savefig(output_path)
    plt.close()

# main function
def reshape_features(features, sequence_length):
    # Reshape the features to (num_sequences, sequence_length, feature_size)
    # Calculate the number of valid sequences
    num_sequences = features.size(0) // sequence_length
    features = features[:num_sequences * sequence_length]  # Trim to fit exact multiples of sequence_length
    features = features.view(num_sequences, sequence_length, features.size(1))  # Reshape

    return features

def main():
    # Setup logging (use append mode to prevent overwriting)
    log_file_path = '/storage/home/ybr5070/group/homebytes/code/scripts/logs/train_lstm_final_overlap.log'
    logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', filemode='a')

    logging.info('Script started')

    sequence_length = 51  # Set the sequence length

    # Load training datasets
    bite_tensor_path = '/storage/group/klk37/default/homebytes/video/fbs/tensors_train_lstm/consolidated_bite_sequences.pt'
    non_bite_tensor_path = '/storage/group/klk37/default/homebytes/video/fbs/tensors_train_lstm/consolidated_non_bite_sequences.pt'

    bite_data = torch.load(bite_tensor_path)
    non_bite_data = torch.load(non_bite_tensor_path)

    bite_features, bite_labels = bite_data['features'], bite_data['labels']
    non_bite_features, non_bite_labels = non_bite_data['features'], non_bite_data['labels']

    # Log initial lengths
    logging.info(f"Loaded bite features length: {len(bite_features)}")
    logging.info(f"Loaded non-bite features length: {len(non_bite_features)}")
    logging.info(f"Loaded bite labels length: {len(bite_labels)}")
    logging.info(f"Loaded non-bite labels length: {len(non_bite_labels)}")

    # Reshape the features to match the sequence length
    bite_features = reshape_features(bite_features, sequence_length)
    non_bite_features = reshape_features(non_bite_features, sequence_length)

    # Ensure that labels match the number of sequences
    bite_labels = bite_labels[:len(bite_features)]
    non_bite_labels = non_bite_labels[:len(non_bite_features)]

    # Log reshaped lengths
    logging.info(f"Reshaped bite features shape: {bite_features.shape}")
    logging.info(f"Reshaped non-bite features shape: {non_bite_features.shape}")
    logging.info(f"Reshaped bite labels length: {len(bite_labels)}")
    logging.info(f"Reshaped non-bite labels length: {len(non_bite_labels)}")

    # Load validation datasets
    val_bite_tensor_path = '/storage/group/klk37/default/homebytes/video/fbs/tensors_val_lstm/consolidated_bite_sequences.pt'
    val_non_bite_tensor_path = '/storage/group/klk37/default/homebytes/video/fbs/tensors_val_lstm/consolidated_non_bite_sequences.pt'

    val_bite_data = torch.load(val_bite_tensor_path)
    val_non_bite_data = torch.load(val_non_bite_tensor_path)

    val_bite_features, val_bite_labels = val_bite_data['features'], val_bite_data['labels']
    val_non_bite_features, val_non_bite_labels = val_non_bite_data['features'], val_non_bite_data['labels']

    # Reshape the validation features
    val_bite_features = reshape_features(val_bite_features, sequence_length)
    val_non_bite_features = reshape_features(val_non_bite_features, sequence_length)

    val_bite_labels = val_bite_labels[:len(val_bite_features)]
    val_non_bite_labels = val_non_bite_labels[:len(val_non_bite_features)]

    # Concatenate features and labels for training data
    features = torch.cat((bite_features, non_bite_features), dim=0)
    labels = torch.cat((bite_labels, non_bite_labels), dim=0)

    # Concatenate features and labels for validation data
    val_features = torch.cat((val_bite_features, val_non_bite_features), dim=0)
    val_labels = torch.cat((val_bite_labels, val_non_bite_labels), dim=0)

    # Log final shapes
    logging.info(f"Combined features shape: {features.shape}, Combined labels shape: {labels.shape}")
    logging.info(f"Combined validation features shape: {val_features.shape}, Combined validation labels shape: {val_labels.shape}")

    # Parameters
    input_size = features.size(2)  # Now this should work
    hidden_size = 256
    num_layers = 2
    num_classes = 2
    num_epochs = 15
    batch_size = 32
    learning_rate = 0.001
    patience = 5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = CombinedDataset(features, labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = CombinedDataset(val_features, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize and train the model from scratch
    model = LSTMModel(input_size, hidden_size, num_layers, num_classes, device).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train on the full dataset with early stopping using validation feedback
    train_losses, val_losses, train_accuracies, val_accuracies, train_f1s, val_f1s = train_model_with_early_stopping(
        model, criterion, optimizer, train_loader, val_loader, num_epochs, device, patience
    )

    # Save the final trained model
    torch.save(model.state_dict(), '/storage/home/ybr5070/group/homebytes/code/scripts/lstm_model_final.pth')

    # Plot and save metrics
    plot_metrics(train_losses, val_losses, 'Loss', '/storage/home/ybr5070/group/homebytes/code/scripts/full_train_loss_overlap.png')
    plot_metrics(train_accuracies, val_accuracies, 'Accuracy', '/storage/home/ybr5070/group/homebytes/code/scripts/full_train_accuracy_overlap.png')
    plot_metrics(train_f1s, val_f1s, 'F1 Score', '/storage/home/ybr5070/group/homebytes/code/scripts/full_train_f1_score_overlap.png')

    logging.info('Training script finished')

if __name__ == '__main__':
    main()

