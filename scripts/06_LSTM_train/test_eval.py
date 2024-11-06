import os
import torch
import numpy as np
import logging
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score
from concurrent.futures import ThreadPoolExecutor

# Setup logging
log_file_path = '/storage/home/ybr5070/group/homebytes/code/scripts/logs/evaluate_lstm_test.log'
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

# Filter and pad sequences
def filter_and_pad_sequence(feat, label):
    if feat.size(0) < 50:
        return None, None  # Discard sequences with fewer than 50 frames
    if feat.size(0) == 50:
        feat = torch.cat((feat, feat[-1].unsqueeze(0)), dim=0)  # Pad by repeating the last frame
    return feat, label

def filter_and_pad_sequences_parallel(features, labels):
    filtered_features = []
    filtered_labels = []
    removed_count = 0
    padded_count = 0

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(filter_and_pad_sequence, features, labels))

    for feat, label in results:
        if feat is None:
            removed_count += 1
        else:
            if feat.size(0) == 51:
                padded_count += 1
            filtered_features.append(feat)
            filtered_labels.append(label)

    included_count = len(filtered_features)
    return filtered_features, filtered_labels, removed_count, padded_count, included_count

# Evaluate model on test set
def evaluate_model(model, criterion, test_loader, device):
    model.eval()
    test_running_loss = 0.0
    all_test_preds = []
    all_test_labels = []
    with torch.no_grad():
        for test_features, test_labels in test_loader:
            test_features, test_labels = test_features.to(device), test_labels.to(device)
            test_outputs = model(test_features)
            test_loss = criterion(test_outputs, test_labels)
            test_running_loss += test_loss.item()
            all_test_preds.extend(torch.argmax(test_outputs, dim=1).cpu().numpy())
            all_test_labels.extend(test_labels.cpu().numpy())

    test_loss = test_running_loss / len(test_loader)
    test_acc = accuracy_score(all_test_labels, all_test_preds)
    test_f1 = f1_score(all_test_labels, all_test_preds, average='weighted')

    logging.info(f'test Loss: {test_loss}, test Accuracy: {test_acc}, test F1 Score: {test_f1}')
    
    return test_loss, test_acc, test_f1

def main():
    logging.info('Evaluation script started')

    # Define parameters
    hidden_size = 256
    num_layers = 2
    num_classes = 2
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load test datasets
    test_bite_tensor_path = '/storage/group/klk37/default/homebytes/code/scripts/tensors_yolo_test/all_bite_sequences.pt'
    test_non_bite_tensor_path = '/storage/group/klk37/default/homebytes/code/scripts/tensors_yolo_test/all_non_bite_sequences.pt'

    test_bite_data = torch.load(test_bite_tensor_path)
    test_non_bite_data = torch.load(test_non_bite_tensor_path)

    test_bite_features, test_bite_labels = test_bite_data
    test_non_bite_features, test_non_bite_labels = test_non_bite_data

    # Filter and pad sequences with less than 50 frames
    test_bite_features, test_bite_labels, test_bite_removed, test_bite_padded, test_bite_included = filter_and_pad_sequences_parallel(test_bite_features, test_bite_labels)
    test_non_bite_features, test_non_bite_labels, test_non_bite_removed, test_non_bite_padded, test_non_bite_included = filter_and_pad_sequences_parallel(test_non_bite_features, test_non_bite_labels)

    # Log the number of sequences removed, padded, and included
    logging.info(f'Removed {test_bite_removed} test bite sequences and {test_non_bite_removed} test non-bite sequences.')
    logging.info(f'Padded {test_bite_padded} test bite sequences and {test_non_bite_padded} test non-bite sequences.')
    logging.info(f'Included {test_bite_included} test bite sequences and {test_non_bite_included} test non-bite sequences.')

    # Convert lists to tensors
    test_bite_features = torch.stack(test_bite_features)
    test_bite_labels = torch.tensor(test_bite_labels)
    test_non_bite_features = torch.stack(test_non_bite_features)
    test_non_bite_labels = torch.tensor(test_non_bite_labels)

    # Concatenate test features and labels
    test_features = torch.cat((test_bite_features, test_non_bite_features), dim=0)
    test_labels = torch.cat((test_bite_labels, test_non_bite_labels), dim=0)

    # Define input size
    input_size = test_features.size(2)

    # Create DataLoader for test set
    test_dataset = CombinedDataset(test_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load the best model
    best_model_path = '/storage/group/klk37/default/homebytes/code/scripts/lstm_model_best.pth'
    model = LSTMModel(input_size, hidden_size, num_layers, num_classes, device).to(device)
    model.load_state_dict(torch.load(best_model_path))

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Evaluate the model
    test_loss, test_acc, test_f1 = evaluate_model(model, criterion, test_loader, device)

    logging.info(f'Final test Loss: {test_loss}, Final test Accuracy: {test_acc}, Final test F1 Score: {test_f1}')
    logging.info('test Evaluation script finished')

if __name__ == '__main__':
    main()
