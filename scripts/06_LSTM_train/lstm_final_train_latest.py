import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

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

# Load data and reshape
def load_and_prepare_data(bite_tensor_path, non_bite_tensor_path, sequence_length):
    # Load the bite and non-bite data
    bite_data = torch.load(bite_tensor_path)
    non_bite_data = torch.load(non_bite_tensor_path)

    # Extract features and labels from the loaded data
    bite_features, bite_labels = bite_data['features'], bite_data['labels']
    non_bite_features, non_bite_labels = non_bite_data['features'], non_bite_data['labels']

    # Reshape features to match the sequence length
    bite_features = reshape_features(bite_features, sequence_length)
    non_bite_features = reshape_features(non_bite_features, sequence_length)

    # Ensure labels match the number of sequences
    bite_labels = bite_labels[:len(bite_features)]
    non_bite_labels = non_bite_labels[:len(non_bite_features)]

    # Concatenate features and labels for training set
    features = torch.cat((bite_features, non_bite_features), dim=0)
    labels = torch.cat((bite_labels, non_bite_labels), dim=0)

    return features, labels

# Save graphs
def save_graphs(train_losses, train_accuracies, path_prefix):
    # Plot loss graph
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{path_prefix}_loss.png')
    plt.close()

    # Plot accuracy graph
    plt.figure()
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'{path_prefix}_accuracy.png')
    plt.close()

# Main execution
def main():
    # Paths to data
    bite_tensor_path = '/storage/group/klk37/default/homebytes/video/fbs/tensors_train_lstm/consolidated_bite_sequences.pt'
    non_bite_tensor_path = '/storage/group/klk37/default/homebytes/video/fbs/tensors_train_lstm/consolidated_non_bite_sequences.pt'

    # Sequence length for LSTM
    sequence_length = 51

    # Load and prepare data
    features, labels = load_and_prepare_data(bite_tensor_path, non_bite_tensor_path, sequence_length)

    # Set device to CPU
    device = torch.device('cpu')

    # Define model parameters
    input_size = features.size(2)  # Number of input features per time step
    hidden_size = 256
    num_layers = 2
    num_classes = 2
    batch_size = 8
    learning_rate = 0.001
    num_epochs = 8  # Updated epochs

    # Create dataset and loader
    train_dataset = CombinedDataset(features, labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model
    model = LSTMModel(input_size, hidden_size, num_layers, num_classes, device).to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # For tracking losses and accuracies
    train_losses = []
    train_accuracies = []

    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        logging.info(f'Starting Epoch {epoch + 1}/{num_epochs}')

        for batch_features, batch_labels in tqdm(train_loader, desc="Training Batches", leave=False):
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

            # Forward pass
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss and correct predictions
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += batch_labels.size(0)
            correct_predictions += (predicted == batch_labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_predictions / total_predictions
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')

    # Save final model
    torch.save(model.state_dict(), '/storage/home/ybr5070/group/homebytes/code/scripts/lstm_model_final_latest.pth')

    # Save graphs
    save_graphs(train_losses, train_accuracies, '/storage/home/ybr5070/group/homebytes/code/scripts/lstm_model_final_latest')

if __name__ == '__main__':
    main()
