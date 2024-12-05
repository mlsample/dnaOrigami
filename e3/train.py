import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.dataset import DNADataset
from model.origami_model import DNAOrigamiModel
from utils.logger import get_logger

def main():
    # Load dataset
    trajectory_filepaths = ["dataset/data/trajectory.dat"]
    topology_filepaths = ["dataset/data/output.top"]
    dataset = DNADataset(trajectory_filepaths, topology_filepaths)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model, loss function, and optimizer
    input_dim = 13
    embed_dim = 64
    hidden_dim = 128
    layers = 3
    model = DNAOrigamiModel(input_dim, embed_dim, hidden_dim, layers)
    loss_function = nn.MSELoss()  # temp loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    epochs = 20
    # logger = Logger()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()

            nodes = batch['nodes']
            edge_index = batch['edge_index']
            edge_attr = batch['edge_attr']

            # Forward pass
            outputs = model(nodes, edge_index, edge_attr)
            loss = loss_function(outputs, nodes)  
            
            # Backward pass
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        logger.log(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), "dna_model.pth")

if __name__ == "__main__":
    main()