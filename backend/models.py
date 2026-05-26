import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_hand_edges():
    """
    Returns the edge connections representing the hand skeleton structure.
    MediaPipe landmarks index-based.
    """
    edges = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index finger
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle finger
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring finger
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Pinky
        (0, 17), (17, 18), (18, 19), (19, 20),
        # Transverse connections (between knuckle bases for lateral context)
        (5, 9), (9, 13), (13, 17)
    ]
    return edges

def build_adjacency_matrix(num_nodes=21):
    """
    Builds a symmetrically normalized adjacency matrix.
    A_hat = D^-1/2 * (A + I) * D^-1/2
    """
    adj = np.zeros((num_nodes, num_nodes))
    edges = get_hand_edges()
    for u, v in edges:
        adj[u, v] = 1.0
        adj[v, u] = 1.0
        
    # Add self loops
    adj = adj + np.eye(num_nodes)
    
    # Normalize
    row_sum = np.sum(adj, axis=1)
    d_inv_sqrt = np.zeros_like(row_sum)
    nonzero = row_sum > 0
    d_inv_sqrt[nonzero] = 1.0 / np.sqrt(row_sum[nonzero])
    
    # Do matrix multiplication in PyTorch to avoid NumPy BLAS warnings on Python 3.14+
    adj_tensor = torch.tensor(adj, dtype=torch.float32)
    d_inv_sqrt_tensor = torch.tensor(d_inv_sqrt, dtype=torch.float32)
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt_tensor)
    
    adj_normalized = torch.mm(torch.mm(d_mat_inv_sqrt, adj_tensor), d_mat_inv_sqrt)
    return adj_normalized

class GCNLayer(nn.Module):
    """
    Custom Graph Convolutional Layer.
    Calculates A_hat * X * W
    """
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # x: (batch_size, num_nodes, in_features)
        # adj: (num_nodes, num_nodes)
        
        # Linear projection (X * W)
        h = self.linear(x) # (batch_size, num_nodes, out_features)
        
        # Propagation (A_hat * H)
        # Broadcast multiplication: A_hat (21, 21) @ H (B, 21, F) -> (B, 21, F)
        out = torch.matmul(adj, h)
        return out

class HandGCN(nn.Module):
    """
    Graph Convolutional Network for Hand Gesture Recognition.
    """
    def __init__(self, in_features=3, num_classes=9):
        super(HandGCN, self).__init__()
        # Register adjacency matrix as buffer so it moves to GPU automatically if needed
        self.register_buffer("adj", build_adjacency_matrix())
        
        self.gcn1 = GCNLayer(in_features, 32)
        self.gcn2 = GCNLayer(32, 64)
        self.gcn3 = GCNLayer(64, 128)
        
        self.fc1 = nn.Linear(128 * 2, 128) # Global avg and max pool concatenated
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x: (batch_size, num_nodes, in_features)
        
        # Graph convolution layers
        h = F.relu(self.gcn1(x, self.adj))
        h = self.dropout(h)
        
        h = F.relu(self.gcn2(h, self.adj))
        h = self.dropout(h)
        
        h = F.relu(self.gcn3(h, self.adj))
        h = self.dropout(h)
        
        # Readout (Pooling): pool over the node dimension (dim=1)
        # Mean pool: (batch_size, out_features)
        mean_pool = torch.mean(h, dim=1)
        # Max pool: (batch_size, out_features)
        max_pool, _ = torch.max(h, dim=1)
        
        # Concatenate poolings
        g = torch.cat([mean_pool, max_pool], dim=1) # (batch_size, out_features * 2)
        
        # Fully Connected classification head
        g = F.relu(self.fc1(g))
        g = self.dropout(g)
        logits = self.fc2(g)
        
        return logits

class HandMLP(nn.Module):
    """
    Baseline General Multi-Layer Perceptron (MLP).
    Takes flattened node features.
    """
    def __init__(self, in_features=21*3, num_classes=9):
        super(HandMLP, self).__init__()
        self.fc1 = nn.Linear(in_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x: (batch_size, num_nodes, node_features) -> flatten to (batch_size, num_nodes * node_features)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        
        h = F.relu(self.fc2(h))
        h = self.dropout(h)
        
        logits = self.fc3(h)
        return logits
