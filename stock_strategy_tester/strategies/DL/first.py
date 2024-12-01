import torch
import torch.nn as nn
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
    data['Pct_Change'] = data['Close'].pct_change()
    data['Pct_Future_Close'] = data['Pct_Change'].pct_change().shift(-5)
    # Normalize the target (Pct_Change) as well
    Volume_rolling_mean = data['Volume'].rolling(window=300).mean()
    Volume_rolling_std = data['Volume'].rolling(window=300).std()
    data['Volume_Norm'] = (data['Volume'] - Volume_rolling_mean) / Volume_rolling_std

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
        self.model_features = ['Volume_Norm', 'Pct_Change', 'Pct_Future_Close']
        self.model_index = self.data["Date"]
        self.model_data = self.data[self.model_features]
        # Assign index to the model data
        self.model_data.index = self.model_index

    def __len__(self):
        return len(self.data) - max(self.seq_lengths)

    def __getitem__(self, index):
        # Collect the sequences for each LSTM with different sequence lengths
        X = []
        for seq_len in self.seq_lengths:
            if index >= seq_len:
                sequence = self.model_data.iloc[index - seq_len:index, :-1].values
            else:
                # For shorter sequences, pad with zeros
                sequence = np.zeros((seq_len, len(self.model_data.columns) - 1))
                sequence[:index, :] = self.model_data.iloc[:index, :-1].values
            X.append(torch.tensor(sequence, dtype=torch.float32))

        # Get the target label for prediction
        y = self.data.iloc[index, -1]  # Pct_Change column is the label

        return X, np.float32(y)


class BiLSTMAttentionModel(nn.Module):
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
        super(BiLSTMAttentionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Dynamically create LSTM layers based on sequence lengths
        self.lstm_layers = nn.ModuleDict({
            f'lstm_{seq_len}': nn.LSTM(seq_len, hidden_size, num_layers,
                                       batch_first=True, bidirectional=True)
            for seq_len in seq_lengths
        })

        # Attention layer to focus on key features from different LSTM outputs
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_size * 2 * len(seq_lengths), num_heads=4)
        self.q = nn.Linear(hidden_size * 2 * len(seq_lengths), hidden_size * 2 * len(seq_lengths))
        self.k = nn.Linear(hidden_size * 2 * len(seq_lengths), hidden_size * 2 * len(seq_lengths))
        self.v = nn.Linear(hidden_size * 2 * len(seq_lengths), hidden_size * 2 * len(seq_lengths))

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
            # from torch.Size([32, 5, 2]) to torch.Size([32, 2, 5])
            input = input.permute(0, 2, 1)
            lstm_out, _ = lstm(input)
            lstm_outputs.append(lstm_out[:, -1, :])

        combined_output = torch.cat(lstm_outputs, dim=1)

        # Attention mechanism
        q = self.q(combined_output)
        k = self.k(combined_output)
        v = self.v(combined_output)
        attention_output, attn_output_weights = self.multihead_attn(q, k, v)

        # Weighted sum of attention output
        weighted_output = torch.sum(attention_output, dim=0)

        output = self.fc(weighted_output)
        return output


def train_model(model, dataloader, device, lr, epochs):
    # Training routine for the model
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for X, y in dataloader:
            X = [x.to(device) for x in X]
            y = y.float().to(device)  # Convert y to float
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}')

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


if __name__ == "__main__":
    # Example configuration and data loading
    seq_lengths = [5, 10, 20, 150]  # Sequence lengths for LSTMs
    input_size = 5  # Open, High, Low, Close, Volume
    hidden_size = 64
    num_layers = 2
    output_size = 1  # Predict price change in 5 days
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = 0.001
    epochs = 150

    # Load and preprocess data
    # ticker = "AAPL"
    ticker = "JPM"
    start_date = "2020-01-01"
    end_date = "2023-01-01"

    # Load and preprocess data
    raw_data = load_data(ticker, start_date, end_date)
    processed_data = preprocess_data(raw_data)
    processed_data = preprocess_data2(processed_data)
    dataset = StockDataset(processed_data, seq_lengths)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Initialize and train the model
    model = BiLSTMAttentionModel(input_size, hidden_size, num_layers, output_size, seq_lengths).to(device)
    trained_model = train_model(model, dataloader, device, lr, epochs)

    # Evaluate the model
    predictions, labels = evaluate_model(trained_model, dataloader, device)
    print("Mean Squared Error:", ((predictions - labels) ** 2).mean())