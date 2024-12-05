import numpy as np
import torch
from utils.geometry import normalize_vectors, pairwise_distances, angle_between_vectors

def tokenize_trajectory(data):
    # Extract positions and orientations
    # print(data)
    
    data = data.astype(float)
    positions = data[:, :, 0:3]  # x, y, z
    a1 = data[:, :, 3:6]         # a1x, a1y, a1z
    a3 = data[:, :, 6:9]         # a3x, a3y, a3z
    strand_index = data[:, :, 9]
    base = data[:, :, 10]
    neighbor_3 = data[:, :, 11]
    neighbor_5 = data[:, :, 12]

    # Center the pos relative to the first node for translation invariance
    center = positions[0]
    positions = positions - center
    
    # Normalize orientation vectors for rotational invariance
    positions = torch.tensor(positions, dtype=torch.float32)
    a1 = torch.tensor(a1, dtype=torch.float32)
    a3 = torch.tensor(a3, dtype=torch.float32)
    a1_normalized = normalize_vectors(a1)
    a3_normalized = normalize_vectors(a3)
    
    #Convert strand index to tensor
    strand_index = torch.tensor(strand_index, dtype=torch.float32)
    
    # Convert categorical data (base) to numerical if needed
    base_numeric = torch.tensor(base, dtype=torch.float32)
    
    # Convert neighbor indices to tensor
    neighbor_3 = torch.tensor(neighbor_3, dtype=torch.float32)
    neighbor_5 = torch.tensor(neighbor_5, dtype=torch.float)

    
    # Combine all features into nodes
    nodes = torch.cat([
        positions, a1_normalized, a3_normalized,
        strand_index.unsqueeze(2), base_numeric.unsqueeze(2),
        neighbor_3.unsqueeze(2), neighbor_5.unsqueeze(2)
    ], dim=2)
    
    print(nodes.shape)

    # edge indexes for a fully connected graph
    num_nodes = nodes.shape[1]
    # gen. all possible indices for unique node pairs that define edges
    edge_index = torch.combinations(torch.arange(num_nodes), r=2, with_replacement=False).T

    print(edge_index.shape) 

    # calc pairwise distances
    pairwise_dist_matrix = pairwise_distances(positions)
    
    print(pairwise_dist_matrix.shape)
    
    edge_attr_distances = pairwise_dist_matrix[edge_index[0], edge_index[1]].unsqueeze(1)

    # add angles as edge features
    edge_attr_angles = []
    for i, j in zip(edge_index[0], edge_index[1]):
        v1 = nodes[i, 3:6]  # a1 of node i
        v2 = nodes[j, 3:6]  # a1 of node j
        angle = angle_between_vectors(v1, v2)
        edge_attr_angles.append(angle)

    edge_attr_angles = torch.stack(edge_attr_angles).unsqueeze(1)

    # Combine distance and angle as edge attributes
    edge_attr = torch.cat([edge_attr_distances, edge_attr_angles], dim=1)

    # Return nodes, edge_index, and edge_attr
    return {
        'nodes': nodes,
        'edge_index': edge_index,
        'edge_attr': edge_attr
    }
