import torch
from torch.utils.data import Dataset
from utils.parsing import parse_dna_origami_data
from utils.tokenize import tokenize_trajectory

class DNADataset(Dataset):
    def __init__(self, trajectory_filepaths, topology_filepaths):
        # Tokenize and construct graph representations for each trajectory file
        self.graphs = [
            tokenize_trajectory(parse_dna_origami_data(fp, topo_fp))
            for fp, topo_fp in zip(trajectory_filepaths, topology_filepaths)
        ]

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        # Returns a dictionary with nodes, edge_index, and edge_attr
        return self.graphs[idx]  