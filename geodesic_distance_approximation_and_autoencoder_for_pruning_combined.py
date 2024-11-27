import numpy as np
import torch
from sklearn.cluster import KMeans
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import matplotlib.pyplot as plt

# Set the device for computations: CUDA (GPU) if available, else CPU
device = (
    "cuda"  # Use CUDA for GPU acceleration if available
    if torch.cuda.is_available()
    else "mps"  # Use Metal Performance Shaders if on a Mac
    if torch.backends.mps.is_available()
    else "cpu"  # Default to CPU if no GPU is available
)

# ----------------- Geodesic Distance Approximation Functions ---------------- #

# Load data
def get_data():
    return torch.tensor(np.load("d61.npy")).float() # Convert to float tensor

# Main function to approximate geodesic distances
def main(CONDENSE_FACTOR=0.5, NUM_NEIGHBORS=5):

    # Load the point cloud data
    data = get_data()

    # Cluster data to reduce the number of points
    # Number of clusters is proportional to "CONDENSE_FACTOR"
    data = KMeans(n_clusters=int(len(data) * CONDENSE_FACTOR)).fit(data.numpy()).cluster_centers_
    data = torch.tensor(data, device=device) # Move clustered data to the specified device (CPU/GPU)

    # Calculate pairwise distances between points
    N = data.shape[0] # Number of points after clustering
    D = torch.cdist(data, data) # Calculate pairwise Euclidean distances (N x N matrix)

    # Identify nearest neighbors for each point
    nearest_indices = D.argsort(dim=1)[:, 1 : NUM_NEIGHBORS + 1] # Get indices of nearest neighbors, excluding self

    # Construct the adjacency matrix (A)
    A = torch.zeros((N, N), dtype=torch.int32, device=device) # Initialize adjacency matrix
    rows = torch.arange(N, device=device).repeat_interleave(NUM_NEIGHBORS) # Repeat row indices for each neighbor
    cols = nearest_indices.flatten() # Flatten neighbor indices
    A[rows, cols] = 1 # Set 1 where a connection exists
    A = torch.maximum(A, A.T) # Make the matrix symmetric

    # Construct geodesic distance matrix (G)
    G = torch.where(A > 0, D, float("inf")) # Use pairwise distances where there is a connection, use inf otherwise
    G.fill_diagonal_(0) # Set diagonal to 0 (distance from a point to itself)

    # Calculate shortest paths between points
    for point in range(N):
        # Update distances by taking intermediate point into account
        G = torch.minimum(G, G[:, point].unsqueeze(1) + G[point, :])

    return G.cpu().numpy() # Return geodesic distance matrix as a NumPy array

# Function to run performance tests
def test():
    condense_factors = np.arange(0.1, 0.95, 0.05)
    num_neighbors = range(2, 21)
    num_points = range(1000, 10000, 1000)

    condense_times = []
    neighbor_times = []
    point_times = []

    for cf in condense_factors:
        start_time = time.time()
        main(CONDENSE_FACTOR=cf)
        condense_times.append(1000 * (time.time() - start_time))

    for nn in num_neighbors:
        start_time = time.time()
        main(NUM_NEIGHBORS=nn)
        neighbor_times.append(1000 * (time.time() - start_time))

    for n in num_points:
        start_time = time.time()
        main(NUM_POINTS=n)
        point_times.append((time.time() - start_time))

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.plot(condense_factors, condense_times, marker='o')
    plt.title("Computation Time vs Condense Factor")
    plt.xlabel("Condense Factor")
    plt.ylabel("Time (ms)")
    plt.grid()

    plt.subplot(1, 3, 2)
    plt.plot(num_neighbors, neighbor_times, marker='o')
    plt.title("Computation Time vs Num Neighbors")
    plt.xlabel("Num Neighbors")
    plt.ylabel("Time (ms)")
    plt.grid()

    plt.subplot(1, 3, 3)
    plt.plot(num_points, point_times, marker='o')
    plt.title("Computation Time vs Num Points")
    plt.xlabel("Num Points")
    plt.ylabel("Time (sec)")
    plt.grid()

    # Show plots
    plt.tight_layout()
    plt.show()

# ----------------- Functions to Prune Adjacency Matrix with Autoencoder ---------------- #

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

# Function to train the autoencoder model on input adjacency matrix
def train_autoencoder(model, dataloader, criterion, optimizer, num_epochs=50):

    model.train() # Set model to training mode
    for epoch in range(num_epochs):
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
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.6f}")

# Function to prune relationships in the adjacency matrix based on reconstruction error
def prune_adjacency_matrix(model, adjacency_matrix, threshold=0.009):

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

# ----------------- Generate & Prune Adjacency Matrix ---------------- #

# Choose number of epochs to train over
n_epochs = 50

if __name__ == '__main__':
    
    # Initilize runtime "stopwatch"
    start_time = time.time()

    # Generate the adjacency matrix using the first script
    print("Generating adjacency matrix...")
    adjacency_matrix = main(CONDENSE_FACTOR=0.5, NUM_NEIGHBORS=5)
    adjacency_matrix_tensor = torch.tensor(adjacency_matrix, device=device)

    # Ensure symmetry of adjacency matrix
    adjacency_matrix_tensor = (adjacency_matrix_tensor + adjacency_matrix_tensor.T) / 2

    # Define the autoencoder and hyperparameters
    input_dim = adjacency_matrix_tensor.shape[0]
    latent_dim = 16
    model = Autoencoder(input_dim, latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Prepare data for autoencoder
    dataset = TensorDataset(adjacency_matrix_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Train the autoencoder
    print("Training the autoencoder...")
    train_autoencoder(model, dataloader, criterion, optimizer, num_epochs=n_epochs)

    # Prune the adjacency matrix
    print("Pruning the adjacency matrix...")
    pruned_adjacency_matrix = prune_adjacency_matrix(model, adjacency_matrix_tensor)

    # Save the pruned adjacency matrix
    pruned_adjacency_matrix_np = pruned_adjacency_matrix.cpu().numpy()
    np.savetxt("pruned_adjacency_matrix.txt", pruned_adjacency_matrix_np)
    print("Pruned adjacency matrix saved to 'pruned_adjacency_matrix.txt'.")

    # Calculate and print time taken to run
    print(f"Time taken: {time.time() - start_time:.2f} seconds")