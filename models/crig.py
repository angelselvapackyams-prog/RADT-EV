import torch
import torch.nn as nn
import torch.nn.functional as F

class CRIGModule(nn.Module):
    """
    Collective Reliability Intelligence Graph (CRIG)
    As specified in Section 4.2.
    """
    def __init__(self, d_h=128, num_nodes=6, num_layers=2, lambda_diff=0.1):
        super().__init__()
        self.d_h = d_h
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.lambda_diff = lambda_diff

        # A. Message Passing & Node Update (Eq. 15-16)
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * d_h + 1, d_h),
            nn.ReLU(),
            nn.Linear(d_h, d_h)
        )
        self.gru_node = nn.GRUCell(d_h, d_h)

        # B. Cross-Layer Dependency Encoding (Eq. 17)
        self.W_d = nn.Parameter(torch.randn(d_h, d_h) * 0.01)

        # C. Cascading Vulnerability Discovery (Eq. 18-19)
        self.beta = nn.Parameter(torch.tensor([0.4, 0.4, 0.2], dtype=torch.float32))

        # E. Graph Attention (Eq. 21-22)
        self.W_gat = nn.Linear(d_h, d_h, bias=False)
        self.attn_vec = nn.Parameter(torch.randn(2 * d_h, 1) * 0.1)
        self.W_z = nn.Linear(d_h, d_h, bias=False)

        # F. Dynamic Graph Evolution (Eq. 23-24)
        self.mlp_edge = nn.Sequential(
            nn.Linear(4 * d_h, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, h_nodes, adj_matrix=None):
        """
        h_nodes: (B, N, d_h) hidden reliability states of N ecosystem nodes
        adj_matrix: (B, N, N) weighted adjacency matrix
        Returns:
            z_graph: (B, N, d_h) ecosystem reliability intelligence representation
            vulnerability: (B, N) cascading vulnerability scores
            criticality: (B, N) normalized criticality scores
            K_diff: (B, N, N) graph diffusion matrix
            adj_updated: (B, N, N) updated dynamic adjacency matrix
        """
        B, N, H = h_nodes.shape
        device = h_nodes.device

        if adj_matrix is None:
            # Default connected graph topology
            adj_matrix = torch.ones(B, N, N, device=device) / N

        # 1. Message Passing (Eq. 15-16)
        h_curr = h_nodes
        for layer in range(self.num_layers):
            # Form pairwise inputs
            h_i = h_curr.unsqueeze(2).repeat(1, 1, N, 1) # (B, N, N, H)
            h_j = h_curr.unsqueeze(1).repeat(1, N, 1, 1) # (B, N, N, H)
            a_ij = adj_matrix.unsqueeze(-1)               # (B, N, N, 1)

            pair_in = torch.cat([h_i, h_j, a_ij], dim=-1) # (B, N, N, 2H+1)
            msg = self.msg_mlp(pair_in)                  # (B, N, N, H)
            
            # Aggregate messages
            agg_msg = torch.sum(msg * a_ij, dim=2)        # (B, N, H)

            # GRU update for each node
            h_flat = h_curr.view(-1, H)
            agg_flat = agg_msg.view(-1, H)
            h_updated_flat = self.gru_node(agg_flat, h_flat)
            h_curr = h_updated_flat.view(B, N, H)

        # 2. Graph Attention Mechanism (Eq. 21-22)
        Wh = self.W_gat(h_curr) # (B, N, H)
        Wh_i = Wh.unsqueeze(2).repeat(1, 1, N, 1)
        Wh_j = Wh.unsqueeze(1).repeat(1, N, 1, 1)
        concat_attn = torch.cat([Wh_i, Wh_j], dim=-1) # (B, N, N, 2H)
        
        attn_logits = F.leaky_relu(torch.matmul(concat_attn, self.attn_vec).squeeze(-1)) # (B, N, N)
        attn_weights = F.softmax(attn_logits + (1.0 - adj_matrix) * -1e9, dim=-1)         # (B, N, N)

        z_graph = torch.sigmoid(torch.matmul(attn_weights, self.W_z(h_curr))) # (B, N, H)

        # 3. Graph Diffusion Kernel K(t) = exp(-lambda * L) (Eq. 20)
        # Degree matrix D
        deg = torch.sum(adj_matrix, dim=-1) # (B, N)
        deg_inv_sqrt = torch.diag_embed(1.0 / (torch.sqrt(deg + 1e-6))) # (B, N, N)
        laplacian = torch.eye(N, device=device).unsqueeze(0) - torch.matmul(torch.matmul(deg_inv_sqrt, adj_matrix), deg_inv_sqrt) # (B, N, N)
        
        # Matrix exponential via Taylor approximation
        K_diff = torch.matrix_exp(-self.lambda_diff * laplacian) # (B, N, N)

        # 4. Cascading Vulnerability & Criticality (Eq. 18-19)
        norm_h = torch.norm(h_curr, dim=-1) # (B, N)
        neighbor_norm = torch.matmul(adj_matrix, norm_h.unsqueeze(-1)).squeeze(-1) # (B, N)
        pagerank_proxy = torch.sum(adj_matrix, dim=1) / N # (B, N)

        vulnerability = self.beta[0] * norm_h + self.beta[1] * neighbor_norm + self.beta[2] * pagerank_proxy
        criticality = F.softmax(vulnerability, dim=-1)

        # 5. Dynamic Graph Edge Update (Eq. 23)
        h_i_full = h_curr.unsqueeze(2).repeat(1, 1, N, 1)
        h_j_full = h_curr.unsqueeze(1).repeat(1, N, 1, 1)
        z_i_full = z_graph.unsqueeze(2).repeat(1, 1, N, 1)
        z_j_full = z_graph.unsqueeze(1).repeat(1, N, 1, 1)

        edge_in = torch.cat([h_i_full, h_j_full, z_i_full, z_j_full], dim=-1) # (B, N, N, 4H)
        adj_updated = self.mlp_edge(edge_in).squeeze(-1) # (B, N, N)

        return z_graph, vulnerability, criticality, K_diff, adj_updated
