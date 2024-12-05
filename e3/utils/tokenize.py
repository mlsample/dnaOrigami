import numpy as np
import torch
from utils.geometry import normalize_vectors, pairwise_distances, angle_between_vectors

def tokenize_trajectory(data):
    # Extract positions and orientations
    
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
    

    # edge indexes for a fully connected graph
    num_nodes = nodes.shape[1]
    # gen. all possible indices for unique node pairs that define edges
    #Somthing is wrong here
    edge_index = torch.combinations(torch.arange(num_nodes), r=2, with_replacement=False).T

    # calc pairwise distances
    pairwise_dist_matrix = pairwise_distances(positions)
        
    edge_attr_distances = np.array([pairwise_dist_matrix[i, edge_index[0], edge_index[1]] for i in range(nodes.shape[0])]).reshape(nodes.shape[0], edge_index.shape[1], 1)


    # add angles as edge features
    all_edge_attr_angles = []
    for conf in range(nodes.shape[0]):
        edge_attr_angles = []
        for i, j in zip(edge_index[0], edge_index[1]):
            v1 = nodes[conf, i, 3:6]  # a1 of node i
            v2 = nodes[conf, j, 3:6]  # a1 of node j
            angle = angle_between_vectors(v1, v2)
            edge_attr_angles.append(angle)
        all_edge_attr_angles.append(edge_attr_angles)
    edge_attr_angles = np.array(all_edge_attr_angles).reshape(nodes.shape[0], edge_index.shape[1], 1)


    # Combine distance and angle as edge attributes
    edge_attr_angles = torch.tensor(edge_attr_angles, dtype=torch.float32)
    edge_attr_distances = torch.tensor(edge_attr_distances, dtype=torch.float32)
    
    edge_attr = torch.cat([edge_attr_distances, edge_attr_angles], dim=2)

    # Return nodes, edge_index, and edge_attr
    return {
        'nodes': nodes,
        'edge_index': edge_index,
        'edge_attr': edge_attr
    }
