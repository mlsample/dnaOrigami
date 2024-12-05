import torch

def pairwise_distances(positions):
    """
    Computes the pairwise Euclidean distances between all nodes.

    Parameters:
        positions (torch.Tensor): A tensor of shape (num_nodes, 3), representing the x, y, z positions of nodes.

    Returns:
        torch.Tensor: A tensor of shape (num_nodes, num_nodes) containing the pairwise distances.
    """
    dist_matrix = torch.cdist(positions, positions, p=2)  # Pairwise Euclidean distance between all nodes
    return dist_matrix

def angle_between_vectors(v1, v2):
    """
    Calculates the angle between two vectors in radians.

    Parameters:
        v1 (torch.Tensor): A tensor of shape (3,) representing the first vector.
        v2 (torch.Tensor): A tensor of shape (3,) representing the second vector.

    Returns:
        torch.Tensor: A scalar representing the angle between the two vectors.
    """
    cos_theta = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2) + 1e-9)
    angle = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))
    return angle

def normalize_vectors(vectors):
    """
    Normalizes a batch of vectors.

    Parameters:
        vectors (torch.Tensor): A tensor of shape (num_vectors, 3) representing the vectors to normalize.

    Returns:
        torch.Tensor: A tensor of the same shape as input, where each vector is normalized to unit length.
    """
    norms = torch.norm(vectors, dim=2, keepdim=True) + 1e-9
    normalized_vectors = vectors / norms
    return normalized_vectors

def cross_product(v1, v2):
    """
    Computes the cross product between two vectors.

    Parameters:
        v1 (torch.Tensor): A tensor of shape (3,) representing the first vector.
        v2 (torch.Tensor): A tensor of shape (3,) representing the second vector.

    Returns:
        torch.Tensor: A tensor of shape (3,) representing the cross product vector.
    """
    return torch.cross(v1, v2)

def dihedral_angle(p1, p2, p3, p4):
    """
    Calculates the dihedral angle formed by four points in 3D space.

    Parameters:
        p1, p2, p3, p4 (torch.Tensor): Tensors of shape (3,) representing the points.

    Returns:
        torch.Tensor: A scalar representing the dihedral angle in radians.
    """
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    # Normalize the vectors
    b1 = b1 / (torch.norm(b1) + 1e-9)
    b2 = b2 / (torch.norm(b2) + 1e-9)
    b3 = b3 / (torch.norm(b3) + 1e-9)

    # Calculate normals to the planes defined by (p1, p2, p3) and (p2, p3, p4)
    n1 = torch.cross(b1, b2)
    n2 = torch.cross(b2, b3)

    # Calculate angle between the normals
    angle = torch.atan2(
        torch.dot(torch.cross(n1, n2), b2) * torch.norm(b2),
        torch.dot(n1, n2)
    )

    return angle
