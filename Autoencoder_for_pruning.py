import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Define Autoencoder class
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()

        # Encoder to compress the adjacency matrix into a lower-dimensional space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),  # Fully connected layer: input_dim compressed to 64
            nn.ReLU(),                 # ReLU activation
            nn.Linear(64, 32),         # Fully connected layer: 64 compressed to 32
            nn.ReLU(),                 # ReLU activation
            nn.Linear(32, latent_dim)  # Final layer: compress to latent_dim
        )

        # Decoder to reconstruct the adjacency matrix from the latent space
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), # Fully connected layer: latent_dim decompressed to 32
            nn.ReLU(),                 # ReLU activation
            nn.Linear(32, 64),         # Fully connected layer: 32 decompressed to 64
            nn.ReLU(),                 # ReLU activation
            nn.Linear(64, input_dim)   # Final layer: reconstruct input_dim features
        )

    def forward(self, x):
        # Pass input through the encoder
        latent = self.encoder(x)
        # Pass the latent representation through the decoder
        reconstructed = self.decoder(latent)
        return reconstructed, latent # Return both the reconstruction and the latent features

n_epochs = 100  # Specify number of epochs to train over

# Function to train the autoencoder model
def train_autoencoder(model, dataloader, criterion, optimizer, num_epochs=n_epochs):
    model.train() # Set model to training mode
    for epoch in range(n_epochs):
        total_loss = 0 # Accumulate loss over the epoch
        for data in dataloader:
            inputs = data[0] # Extract the input batch from the dataloader
            optimizer.zero_grad() # Reset gradients from previous batch

            # Forward pass: get reconstructed output and latent representation
            reconstructed, _ = model(inputs)
            
            # Calculate the loss (MSE between input and reconstructed output)
            loss = criterion(reconstructed, inputs)
            
            # Backward pass: calculate the gradients
            loss.backward()
            
            # Update model weights
            optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item()

        # Print loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{n_epochs}], Loss: {total_loss:.4f}")

# Function to prune relationships in the adjacency matrix based on reconstruction error
def prune_adjacency_matrix(model, adjacency_matrix, threshold=0.01):
    model.eval() # Set model to evaluation mode (disables dropout, batch norm, etc.)
    with torch.no_grad(): # Disable gradient computation for efficiency
        # Forward pass: get reconstructed output and latent representation
        reconstructed, _ = model(adjacency_matrix)
        
        # Calculate the reconstruction error for each relationship (element-wise error)
        errors = torch.abs(adjacency_matrix - reconstructed)
        
        # Prune relationships with reconstruction error above the threshold
        pruned_adjacency_matrix = adjacency_matrix.clone()
        pruned_adjacency_matrix[errors > threshold] = 0 # Remove relationships with reconstruction error above threshold
        
        print(f"Pruned {torch.sum(errors > threshold).item()} relationships out of {adjacency_matrix.numel()}")
        return pruned_adjacency_matrix

# Generate a sample adjacency matrix (******replace with actual geodesic-based attention matrix later******)
# Create a random adjacency matrix of size 1000x1000 for example use
num_points = 1000
adjacency_matrix_data = np.random.rand(num_points, num_points).astype(np.float32)

# Ensure the adjacency matrix is symmetric
adjacency_matrix_data = (adjacency_matrix_data + adjacency_matrix_data.T) / 2

# Convert the adjacency matrix to a PyTorch tensor
adjacency_matrix_tensor = torch.tensor(adjacency_matrix_data)

# Create a DataLoader for batching the data during training
dataset = TensorDataset(adjacency_matrix_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Define the autoencoder model
input_dim = num_points  # Number of features per point (e.g., adjacency matrix size)
latent_dim = 16         # Latent dimension (adjust for desired compression)
model = Autoencoder(input_dim, latent_dim)

# Define the loss function (Mean Squared Error) and optimizer (Adam)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the autoencoder model
print("Training the autoencoder...")
train_autoencoder(model, dataloader, criterion, optimizer, num_epochs=100)

# Prune the adjacency matrix based on reconstruction error
print("Pruning the adjacency matrix...")
pruned_adjacency_matrix = prune_adjacency_matrix(model, adjacency_matrix_tensor)

# Convert the pruned adjacency matrix back to a NumPy array (if needed)
pruned_adjacency_matrix_np = pruned_adjacency_matrix.numpy()

# Save the pruned adjacency matrix to a text file
np.savetxt("pruned_adjacency_matrix.txt", pruned_adjacency_matrix_np)
print("Pruned adjacency matrix saved to 'pruned_adjacency_matrix.txt'.")