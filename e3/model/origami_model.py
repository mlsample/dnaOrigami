import torch
import torch.nn as nn
from model.input_layer import DNAInputLayer
from model.se3_transformer import DNAOrigamiSE3Transformer
from e3nn.o3 import Irreps

class DNAOrigamiModel(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, layers=3):
        super(DNAOrigamiModel, self).__init__()

        # initialize DNAInputLayer to preprocess the input data
        self.input_layer = DNAInputLayer(input_dim, embed_dim)

        # SE(3)-Transformer model
        input_irreps = Irreps(f"{embed_dim}x0e")
        hidden_irreps = Irreps(f"{hidden_dim}x0e + {hidden_dim}x1o")
        output_irreps = Irreps(f"{embed_dim}x0e")
        self.se3_transformer = DNAOrigamiSE3Transformer(input_irreps, output_irreps, hidden_irreps, layers)

    def forward(self, nodes, edge_index, edge_attr):
        # Pass through the DNAInputLayer to embed and preprocess the input features
        embedded_nodes = self.input_layer(nodes)

        # Pass through the SE(3)-Transformer
        output = self.se3_transformer(embedded_nodes, edge_index, edge_attr)
        return output
