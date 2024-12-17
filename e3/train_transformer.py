import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from dataset.dataset import DNADataset
from model.transformer import Transformer, TransformerEncoder
from utils.logger import get_logger
import os
from ipy_oxdna.generate_replicas import ReplicaGroup
import pickle 

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    # dataset = DNADataset(trajectory_filepaths, topology_filepaths)
    with open('dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
    
    # Initialize model, loss function, and optimizer
    cfg = {'num_layers':4,
           'n_features':22,
           'd_model': 64,
           'nhead': 8,
           'd_ff': 64,
           'dropout_rate': 0.1,
           'device': device}
    
    model = TransformerEncoder(cfg).to(device)
    loss_function = nn.MSELoss()  # temp loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=125, factor=0.5)
    
    # Training loop
    epochs = 2000
    logger = get_logger(__name__)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in dataset:
            optimizer.zero_grad()

            src = batch['nodes'][:-1].to(device)
            n_confs = src.shape[0]
            n_particles = src.shape[1]
            src_mask = ~torch.ones((n_confs, n_particles), dtype=torch.bool).to(device)  # No masking, all positions attend to each other
            tgt = batch['nodes'][1:].to(device)
            tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(n_particles).to(device)

            # Forward pass
            # outputs = model(src, tgt, src_mask, tgt_mask)
            outputs = model(src, src_mask)
            loss = loss_function(outputs, tgt)  
            
            # Backward pass
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        avg_loss = running_loss / len(dataset)
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 250 == 0:
            print(scheduler.get_last_lr())
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        # logger.log(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), "dna_model.pth")

if __name__ == "__main__":
    main()