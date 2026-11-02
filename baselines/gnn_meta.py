import torch
import torch.nn as nn


class GNN_Meta_Baseline(nn.Module):
    """
    4. GNN-Meta [25]: Graph Neural Network combined with Meta-Learning for
    Predictive Energy Optimization.

    The input state is projected into a small graph of latent nodes and refined with
    graph-style message passing before a policy head maps the pooled representation to
    a charging action.
    """

    def __init__(self, state_dim=10, action_dim=5, num_nodes=4, node_dim=8):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_nodes = num_nodes
        self.node_dim = node_dim

        self.node_embed = nn.Linear(state_dim, num_nodes * node_dim)
        self.gnn = nn.Sequential(
            nn.Linear(node_dim, node_dim * 2),
            nn.ReLU(),
            nn.Linear(node_dim * 2, node_dim),
        )
        self.meta_context = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, node_dim),
        )
        self.action_head = nn.Sequential(
            nn.Linear(node_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def _flatten_inputs(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x.reshape(-1, self.state_dim), x

    def forward(self, x):
        flat, original = self._flatten_inputs(x)
        node_feats = self.node_embed(flat).view(-1, self.num_nodes, self.node_dim)
        node_feats = self.gnn(node_feats)
        pooled = node_feats.mean(dim=1)
        context = self.meta_context(flat)
        fused = torch.cat([pooled, context], dim=-1)
        logits = self.action_head(fused)
        return logits.reshape(*original.shape[:-1], self.action_dim) if original.dim() > 1 else logits

    def get_action(self, x, temperature=1.0):
        logits = self.forward(x)
        probs = torch.softmax(logits / max(temperature, 1e-8), dim=-1)
        return probs
