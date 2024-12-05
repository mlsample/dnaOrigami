import torch
import torch.nn as nn

class DNAInputLayer(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(DNAInputLayer, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, tokens):
        embedded = self.embedding(tokens)
        normalized = self.layer_norm(embedded)
        output = self.dropout(normalized)
        return output
