import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from dataset.dataset import DNADataset
from model.transformer import Transformer
from utils.logger import get_logger

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    trajectory_filepaths = ["dataset/data/trajectory.dat"]
    topology_filepaths = ["dataset/data/output.top"]
    dataset = DNADataset(trajectory_filepaths, topology_filepaths)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Initialize model, loss function, and optimizer
    cfg = {'num_layers':2, 'n_features':13, 'd_model': 52, 'nhead': 13, 'num_encoder_layers': 6, 'd_ff': 64, 'dropout_rate': 0.1, 'device': device}
    model = Transformer(cfg).to(device)
    loss_function = nn.MSELoss()  # temp loss function
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=125, factor=0.5, verbose=True)
    # Training loop
    epochs = 20000
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
            outputs = model(src, tgt, src_mask, tgt_mask)
            loss = loss_function(outputs, tgt)  
            
            # Backward pass
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        avg_loss = running_loss
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 250 == 0:
            print(scheduler.get_last_lr())
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        # logger.log(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), "dna_model.pth")

if __name__ == "__main__":
    main()