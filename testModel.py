import math
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Set random seed
torch.manual_seed(42)

# Load data
df = pd.read_csv('./DataProcess/station/1037A/1037A_2020.csv')
# df = df.drop([df.columns[-2], df.columns[-3]], axis=1)
# df = df.head(10000)
df = df.dropna()  # Remove rows with missing values

# Data preprocessing
features = ['PM10', 'SO2', 'NO2', 'O3', 'CO']
target = 'PM2.5'

# Convert timestamp to numeric
df['采集时间'] = pd.to_datetime(df['采集时间'], format='%Y%m%d%H%M%S')
df['采集时间'] = df['采集时间'].apply(lambda x: x.timestamp())

# Filter target
df = df[df[target] <= 2000]

# Normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(df[features])
scaled_target = scaler.fit_transform(df[[target]].values)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare data for Informer
X_scaled = scaled_features.reshape((scaled_features.shape[0], 1, scaled_features.shape[1]))
y_scaled = scaled_target.reshape((scaled_target.shape[0], 1))

# Split dataset
train_size = int(0.99 * len(X_scaled))
X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]

# Create TensorDataset
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32).to(device),
                              torch.tensor(y_train, dtype=torch.float32).to(device))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32).to(device),
                             torch.tensor(y_test, dtype=torch.float32).to(device))

# Create DataLoader
batch_size = 1
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Define Informer
class Informer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_layers, dropout=0.1):
        super(Informer, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.self_attention_layers = nn.ModuleList(
            [nn.MultiheadAttention(hidden_dim, num_heads) for _ in range(num_layers)]
        )
        self.decoder = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Input layer
        x = self.input_layer(x)

        # Self-attention layers

        for layer in self.self_attention_layers:
            x, _ = layer(x, x, x)  # Self-attention

        # Output layer
        x = self.decoder(x)
        return x.squeeze(0)


# Model parameters
input_dim = 16  # Input feature dimension
hidden_dim = 64  # Hidden layer dimension
output_dim = 1  # Output dimension (target)
num_heads = 4  # Number of attention heads
num_layers = 2  # Number of self-attention layers

# Instantiate model
model = Informer(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                 num_heads=num_heads, num_layers=num_layers).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train model
for epoch in range(50):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# Model evaluation
model.eval()
with torch.no_grad():
    predictions = []
    actuals = []
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        predictions.extend(outputs.cpu().numpy().flatten())
        actuals.extend(labels.cpu().numpy().flatten())

# Inverse normalization
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
actuals = scaler.inverse_transform(np.array(actuals).reshape(-1, 1))
# 保存模型
torch.save(model, 'model.pth')

# 加载模型
model = torch.load('model.pth')
# Check shape
print(predictions.shape)
print(actuals.shape)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(actuals, label='Actual')
plt.plot(predictions, label='Predicted')
plt.title('Actual vs Predicted Engine Speed')
plt.xlabel('Sample Index')
plt.ylabel('Engine Speed')
plt.legend()
plt.show()
