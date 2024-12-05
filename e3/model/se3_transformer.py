import torch
import torch.nn as nn
from e3nn.o3 import Irreps, FullyConnectedTensorProduct
from e3nn.util.jit import compile_mode

@compile_mode('trace')
class DNAOrigamiSE3Transformer(nn.Module):
    def __init__(self, input_irreps, output_irreps, hidden_irreps, layers=3):
        super().__init__()
        self.layers = layers

        self.input_layer = nn.Linear(input_irreps.dim, hidden_irreps.dim)
        #self.gate_layer = Gate(input_irreps, output_irreps)
        self.gate_layer = nn.ReLU() # temp gate layer

        # Equivariant hidden layers
        self.hidden_layers = nn.ModuleList([
            FullyConnectedTensorProduct(hidden_irreps, hidden_irreps, "0e")
            for _ in range(self.layers)
        ])

        # Final layer to produce output
        self.output_layer = nn.Linear(hidden_irreps.dim, output_irreps.dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.input_layer(x)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x, edge_index, edge_attr)

        x = self.gate_layer(x)
        x = self.output_layer(x)

        return x
