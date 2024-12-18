import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sympy.polys.polyconfig import query
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

from data.data_loader import load_data, preprocess_data


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

    # Calculate the label: percentage change in closing price in 5 days
    data['Pct_Change'] = data['Close'].pct_change().rolling(window=20).mean() * 100
    data['Pct_open'] = data['Open'].pct_change().rolling(window=20).mean() * 100
    data['Pct_low'] = data['Low'].pct_change().rolling(window=20).mean() * 100
    data['Pct_High'] = data['High'].pct_change().rolling(window=20).mean() * 100

    data['Pct_Future_Close'] = data['Pct_Change'].shift(-1)
    # Normalize the target (Pct_Change) as well
    Volume_rolling_mean = data['Volume'].pct_change().rolling(window=40).mean()  * 100

    data['Volume_Norm'] = Volume_rolling_mean

    # Drop rows where label calculation would introduce NaN values
    data = data.dropna()

    return data


# Updated dataset class to handle percentage change label
class StockDataset(Dataset):
    def __init__(self, data, seq_lengths):
        """
        Dataset for loading stock data with variable sequence lengths.
        Args:
            data (pd.DataFrame): Stock data with features and labels.
            seq_lengths (list): List of sequence lengths for different LSTMs.
        """
        self.data = data
        self.seq_lengths = seq_lengths
        self.model_input_features = ['Volume_Norm', 'Pct_Change', 'Pct_open', 'Pct_low', 'Pct_High']
        self.model_output_features = ['Pct_Future_Close']
        self.model_index = self.data["Date"]

        self.model_data = self.data[self.model_input_features + self.model_output_features]
        # Assign index to the model data
        self.model_data.index = self.model_index



    def __len__(self):
        return len(self.data) - max(self.seq_lengths)

    def __getitem__(self, index):
        # Collect the sequences for each LSTM with different sequence lengths
        X = []
        for seq_len in self.seq_lengths:
            start = (max(self.seq_lengths) + index) - seq_len
            end = start + seq_len
            sequence = self.model_data[self.model_input_features].iloc[start:end].values
            X.append(torch.tensor(sequence, dtype=torch.float32))

        # Get the target label for prediction
        y = self.model_data[self.model_output_features].iloc[end] # Pct_Change column is the label

        return X, np.float32(y)


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

        # Dynamically create LSTM layers based on sequence lengths
        self.lstm_layers = nn.ModuleDict({
            f'lstm_{seq_len}': nn.LSTM(input_size, hidden_size, num_layers,
                                       batch_first=True, bidirectional=True)
            for seq_len in seq_lengths
        })

        # Final fully connected layer for output
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2 * len(seq_lengths), output_size * 64),
            nn.ReLU(),
            nn.Linear(output_size * 64, output_size * 8),
            nn.ReLU(),
            nn.Linear(output_size * 8, output_size)
        )

    def forward(self, x):
        # Forward pass through all LSTM layers and capture last hidden state
        lstm_outputs = []
        for seq_len, lstm in self.lstm_layers.items():
            # Convert the keys into a list of integers for comparison
            seq_lengths_int = [int(key.split('_')[1]) for key in self.lstm_layers.keys()]

            # Updated line for indexing
            input = x[seq_lengths_int.index(int(seq_len.split('_')[1]))].float()
            #Using previous h and c states for subsequent batches without resetting them can introduce
            # data leakage between independent sequences. Don't use h and c states from previous batches.
            lstm_out, _ = lstm(input)

            lstm_outputs.append(lstm_out[:, -1, :])

        combined_output = torch.cat(lstm_outputs, dim=1)

        output = self.fc(combined_output)
        return output


def train_model(model, dataloader, device, lr, epochs):
    # Training routine for the model
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Absolute L1 loss is used as the loss function
    criterion = nn.HuberLoss()
    # criterion = nn.L1Loss()
    model.train()

    # Save best model
    best_model = float('inf')

    for epoch in range(epochs):

        total_loss = 0
        for X, y in dataloader:
            X = [x.to(device) for x in X]
            y = y.float().to(device)  # Convert y to float
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if total_loss < best_model:
            best_model = total_loss
            torch.save(model.state_dict(), "best_model.pt")

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader):.8f}')

    return model


def evaluate_model(model, dataloader, device):
    # Model evaluation on the dataset
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X = [x.to(device) for x in X]
            y = y.float().to(device)  # Convert y to float
            output = model(X)
            predictions.append(output.cpu().numpy())
            labels.append(y.cpu().numpy())
    return np.concatenate(predictions), np.concatenate(labels)

def plot_predictions(predictions, labels):
    # Plot the predictions against the actual labels
    plt.figure(figsize=(14, 7))
    plt.plot(labels, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.title('Predictions vs Actual Labels')
    plt.legend()
    plt.show()

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

if __name__ == "__main__":
    # Example configuration and data loading
    seq_lengths = [5, 10, 20, 150]  # Sequence lengths for LSTMs
    input_size = 5  # input features: Volume_Norm, Pct_Change and Pct_open, Pct_low, Pct_High
    hidden_size = 128
    num_layers = 5
    output_size = 1  # Predict price change in 5 days
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = 10
    epochs = 500
    batch_size = 32

    model_path = "best_model.pt" # None if you want to train the model

    # Load and preprocess data
    # ticker = "AAPL"
    ticker = "JPM"
    start_date = "2018-01-01"
    end_date = "2023-01-01"

    # Load and preprocess data
    raw_data = load_data(ticker, start_date, end_date)
    processed_data = preprocess_data(raw_data)
    processed_data = preprocess_data2(processed_data)
    dataset = StockDataset(processed_data, seq_lengths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Initialize and train the model
    model = BiLSTM1(input_size, hidden_size, num_layers, output_size, seq_lengths).to(device)
    if model_path:
        model = load_model(model, model_path)
    # trained_model = train_model(model, dataloader, device, lr, epochs)

    # Evaluate the model
    if model_path:
        trained_model = load_model(model, model_path)
    predictions, labels = evaluate_model(trained_model, dataloader, device)
    print("Mean Squared Error:", ((predictions - labels) ** 2).mean())
    print("Mean Absolute Error:", np.abs(predictions - labels).mean())

    #Save the model
    torch.save(trained_model.state_dict(), "model.pt")

    # Plot the predictions
    plot_predictions(predictions, labels)


