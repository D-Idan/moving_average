import random
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sympy.polys.polyconfig import query
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

from data.data_loader import load_data, preprocess_data
from strategies.DL.losses import TrendAwareLoss


def preprocess_data2(data):
    """
    Preprocess stock data for training the model.
    Args:
        data (pd.DataFrame): Raw stock data with 'Open', 'High', 'Low', 'Close', and 'Volume' columns.
    Returns:
        pd.DataFrame: Processed data with normalized features and calculated percentage change labels.
    """
    # Remove rows with missing data
    data = data.dropna()

    # Calculate percentage changes and future close price
    data['Pct_Change'] = data['Close'].rolling(window=15).mean().pct_change().rolling(window=50).mean()
    data['Pct_open'] = data['Open'].rolling(window=15).mean().pct_change().rolling(window=50).mean()
    data['Pct_low'] = data['Low'].rolling(window=15).mean().pct_change().rolling(window=50).mean()
    data['Pct_High'] = data['High'].rolling(window=15).mean().pct_change().rolling(window=50).mean()

    # Normalize the volume
    Volume_rolling_mean = data['Volume'].rolling(window=15).mean().pct_change().rolling(window=50).mean()
    data['Volume_Norm'] = Volume_rolling_mean

    # Drop rows with NaN values after the calculations
    data = data.dropna()

    # # Standardize input features
    # scaler = StandardScaler()
    # feature_cols = ['Volume_Norm', 'Pct_Change', 'Pct_open', 'Pct_low', 'Pct_High']
    # data[feature_cols] = scaler.fit_transform(data[feature_cols])

    # Calculate the label: percentage change in closing price in 5 days
    data['MA_Close_Current'] = data['Close'].rolling(window=150).mean()  # Current 5-day moving average
    data['MA_Close_Future'] = data['Close'].shift(-15).rolling(window=150).mean()  # Future 5-day moving average
    data['Pct_Future_Close'] = ((data['MA_Close_Future'] - data['MA_Close_Current']) / data['MA_Close_Current'])

    # Drop rows with NaN values after standardization
    data = data.dropna()

    return data


# Updated dataset class to handle percentage change label
class StockDataset(Dataset):
    def __init__(self, data_s, seq_lengths):
        """
        Dataset for loading stock data with variable sequence lengths.
        Args:
            data (pd.DataFrame): Stock data with features and labels.
            seq_lengths (list): List of sequence lengths for different LSTMs.
        """
        self.data_s = data_s
        self.seq_lengths = seq_lengths
        self.model_input_features = ['Volume_Norm', 'Pct_Change', 'Pct_open', 'Pct_low', 'Pct_High']
        self.model_output_features = ['Pct_Future_Close']
        self.model_index = self.data_s["Date"]

        self.model_data = self.data_s[self.model_input_features + self.model_output_features]
        # Assign index to the model data
        self.model_data.index = self.model_index

    def __len__(self):
        return len(self.data_s) - max(self.seq_lengths)

    def __getitem__(self, index):
        # Collect the sequences for each LSTM with different sequence lengths
        X = []
        for seq_len in self.seq_lengths:
            start = (max(self.seq_lengths) + index) - seq_len
            end = start + seq_len
            sequence = self.model_data[self.model_input_features].iloc[start:end].values
            X.append(torch.tensor(sequence, dtype=torch.float32))

        # Get the target label for prediction
        y = self.model_data[self.model_output_features].iloc[end].values # Pct_Change column is the label

        return X, torch.tensor(y, dtype=torch.float32)


class BiLSTM1(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, seq_lengths):
        """
        Bi-directional LSTM model with attention mechanism for price prediction.
        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of LSTM hidden units.
            num_layers (int): Number of LSTM layers.
            output_size (int): Number of output neurons.
            seq_lengths (list): Sequence lengths for different LSTM layers.
        """
        super(BiLSTM1, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # FC layer
        self.fc1 = nn.Linear(input_size, input_size)

        # LSTM layers
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)

        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(p=0.3)  # Dropout with a probability of 30%

        # Fully connected layers
        self.fc = nn.Sequential(
            # nn.ELU(),
            nn.Linear(hidden_size * len(seq_lengths), hidden_size * len(seq_lengths)),
            nn.LeakyReLU(),
            nn.Linear(hidden_size * len(seq_lengths), hidden_size * len(seq_lengths)),
            nn.LeakyReLU(),
            nn.Linear(hidden_size * len(seq_lengths), output_size),


            # nn.Linear(hidden_size * 2 * len(seq_lengths), output_size * 64),
            # nn.LeakyReLU(),
            # nn.Linear(output_size * 64, output_size ) #* 8),
            # # self.dropout,  # Dropout before final layer
            # # nn.LeakyReLU(),
            # # nn.Linear(output_size * 8, output_size)
        )

    def forward(self, x):
        hidden = (torch.rand(self.num_layers, x[0].size(0), self.hidden_size).to(x[0].device),
                  torch.rand(self.num_layers, x[0].size(0), self.hidden_size).to(x[0].device))
        # FC layer
        x = self.fc1(x[0])

        lstm_out, _ = self.lstm1(x, hidden)  # Forward pass through LSTM layer
        lstm_out = self.dropout(lstm_out)  # Apply dropout

        output = self.fc(lstm_out[:, -1, :])

        return output


def train_model(model, train_loader, val_loader, device, lr, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)#, weight_decay=1e-5)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    criterion = TrendAwareLoss(alpha=0.0)  # Adjust alpha to balance L1 and trend loss


    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_val, y_val in val_loader:
            X_val = [x.to(device) for x in X_val]
            y_val = y_val.float().to(device)
            val_output = model(X_val)
            val_loss += criterion(val_output, y_val).item()

    val_loss /= len(val_loader)
    best_val_loss = val_loss

    # Train the model
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X, y in train_loader:
            X = [x.to(device) for x in X]
            y = y.float().to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val = [x.to(device) for x in X_val]
                y_val = y_val.float().to(device)
                val_output = model(X_val)
                val_loss += criterion(val_output, y_val).item()

        val_loss /= len(val_loader)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.8f}, Val Loss: {val_loss:.8f}')
        # scheduler.step(val_loss)

        # Save the model if it improves on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")

    return model


def evaluate_model_with_indices(model, dataloader, device, dataset):
    model.eval()
    predictions, labels, index_map = [], [], []
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(dataloader):
            X = [x.to(device) for x in X]
            y = y.float().to(device)
            output = model(X)
            predictions.append(output.cpu().numpy())
            labels.append(y.cpu().numpy())
            index_map.extend(
                dataloader.dataset.indices[batch_idx * dataloader.batch_size: (batch_idx + 1) * dataloader.batch_size])

    return np.concatenate(predictions), np.concatenate(labels), index_map

def split_time_series_dataset(dataset, seq_lengths, batch_size):
    # Calculate block size and number of blocks
    block_size = batch_size * max(seq_lengths)
    num_blocks = len(dataset) // block_size

    # Create a list of block indices
    block_indices = list(range(num_blocks))

    # Shuffle and split the block indices according to the train/val/test ratios
    random.shuffle(block_indices)
    train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
    train_end = int(train_ratio * num_blocks)
    val_end = train_end + int(val_ratio * num_blocks)

    train_blocks = block_indices[:train_end]
    val_blocks = block_indices[train_end:val_end]
    test_blocks = block_indices[val_end:]

    # Flatten block indices to generate sample indices
    def get_sample_indices(blocks, block_size):
        return [i for block in blocks for i in range(block * block_size, (block + 1) * block_size)]

    train_indices = get_sample_indices(train_blocks, block_size)
    val_indices = get_sample_indices(val_blocks, block_size)
    test_indices = get_sample_indices(test_blocks, block_size)

    # Create Subsets based on selected indices
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    return train_loader, val_loader, test_loader , test_indices

def plot_predictions(predictions, labels):
    # Plot the predictions against the actual labels
    plt.figure(figsize=(14, 7))
    plt.plot(labels, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.title('Predictions vs Actual Labels')
    plt.legend()
    plt.show()


def plot_predictions_with_signals(predictions, labels, original_prices):
    plt.figure(figsize=(14, 7))

    # Plot actual stock prices
    plt.plot(original_prices, label='Stock Price', color='blue', alpha=0.6)

    # Confusion matrix components
    tp, fp, tn, fn = 0, 0, 0, 0

    # Track indices for true positives and true negatives for marking
    tp_indices = []
    tn_indices = []

    # Iterate over predictions and labels
    for i in range(len(predictions)):
        pred_direction = 1 if predictions[i] >= 0 else -1
        label_direction = 1 if labels[i] >= 0 else -1

        # Update confusion matrix counts and track indices
        if pred_direction == 1 and label_direction == 1:
            tp += 1
            tp_indices.append(i)
        elif pred_direction == 1 and label_direction == -1:
            fp += 1
        elif pred_direction == -1 and label_direction == -1:
            tn += 1
            tn_indices.append(i)
        elif pred_direction == -1 and label_direction == 1:
            fn += 1

        # Plot green for positive, red for negative predictions
        color = 'green' if pred_direction == 1 else 'red'
        plt.axvline(i, color=color, alpha=0.2)

    # Mark true positives and true negatives on the plot
    plt.scatter(tp_indices, [original_prices[i] for i in tp_indices], color='lime', marker='o', label='True Positive', zorder=5)
    plt.scatter(tn_indices, [original_prices[i] for i in tn_indices], color='orange', marker='x', label='True Negative', zorder=5)

    # Calculate percentages for the confusion matrix
    total = tp + fp + tn + fn
    tp_pct = (tp / total * 100) if total > 0 else 0
    fp_pct = (fp / total * 100) if total > 0 else 0
    tn_pct = (tn / total * 100) if total > 0 else 0
    fn_pct = (fn / total * 100) if total > 0 else 0

    # Add confusion matrix percentages as text on the plot
    confusion_text = (f"Confusion Matrix (Percentages):\n"
                      f"TP: {tp_pct:.2f}%, FP: {fp_pct:.2f}%\n"
                      f"TN: {tn_pct:.2f}%, FN: {fn_pct:.2f}%")
    plt.text(0.02, 0.95, confusion_text, transform=plt.gca().transAxes,
             fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

    # Add overall plot title
    plt.title(f'Stock Price with Prediction Signals')
    plt.legend()
    plt.show()


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

if __name__ == "__main__":
    # Example configuration and data loading
    seq_lengths = [24]  # Sequence lengths for LSTMs
    input_size = 5  # input features: Volume_Norm, Pct_Change and Pct_open, Pct_low, Pct_High
    hidden_size = 128
    num_layers = 6
    output_size = 1  # Predict price change in 5 days
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = 0.001
    epochs = 10
    batch_size = 12

    model_path = None
    # model_path = "best_model.pt" # None if you want to train the model

    # Load and preprocess data
    # ticker = "AAPL"
    ticker = "JPM"
    start_date = "2000-01-01"
    end_date = "2023-01-01"

    # Load and preprocess data
    raw_data = load_data(ticker, start_date, end_date)
    processed_data = preprocess_data(raw_data)
    processed_data = preprocess_data2(processed_data)
    dataset = StockDataset(processed_data, seq_lengths)
    # Split data into train, validation, and test
    train_loader, val_loader, test_loader , test_indices = split_time_series_dataset(dataset, seq_lengths, batch_size)

    # Initialize and train the model
    model = BiLSTM1(input_size, hidden_size, num_layers, output_size, seq_lengths).to(device)

    # Evaluate the model
    if model_path:
        trained_model = load_model(model, model_path)
    # Train the model
    trained_model = train_model(model, train_loader, val_loader, device, lr, epochs)

    # Evaluate the model on the test set
    predictions, labels, aligned_indices = evaluate_model_with_indices(trained_model, test_loader, device, dataset)
    print("Test Mean Squared Error:", ((predictions - labels) ** 2).mean())
    print("Test Mean Absolute Error:", np.abs(predictions - labels).mean())

    #Save the model
    torch.save(trained_model.state_dict(), "model.pt")

    # Plot the predictions
    plot_predictions(predictions, labels)

    # Original prices needed for the plot
    # Extract corresponding actual labels from the processed dataset
    first_index = np.argmin(np.abs(train_loader.dataset.dataset.data_s['Pct_Future_Close'].values - labels[0]))
    aligned_indices = np.array(aligned_indices) + (first_index  - aligned_indices[0])
    actual_labels = processed_data['Pct_Future_Close'].iloc[aligned_indices].values    # Ensure alignment for plotting
    # Plot comparison
    original_prices = processed_data['Close'].iloc[aligned_indices].values
    plot_predictions_with_signals(predictions, labels, original_prices)