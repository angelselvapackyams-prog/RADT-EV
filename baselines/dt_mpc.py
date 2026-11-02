import torch
import torch.nn as nn


class DT_MPC_Baseline(nn.Module):
    """
    1. DT-MPC [21]: Digital Twin-Based Distributed Model Predictive Control Strategy
    for Multi-Energy Virtual Power Plants.

    The baseline uses a compact latent state encoder and a predictive control head.
    It is trained end-to-end with the same supervised action targets used by the
    other baselines, while also exposing an auxiliary transition predictor for a
    more explicit digital-twin style state evolution model.
    """

    def __init__(self, state_dim=10, action_dim=5, hidden_dim=64):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.transition_head = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _flatten_inputs(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        flat = x.reshape(-1, self.state_dim)
        return flat, x

    def encode_state(self, x):
        flat, _ = self._flatten_inputs(x)
        latent = self.state_encoder(flat)
        return latent

    def forward(self, x):
        latent = self.encode_state(x)
        logits = self.action_head(latent)
        return logits.reshape(*x.shape[:-1], self.action_dim) if x.dim() > 1 else logits

    def predict_next_state(self, x, action):
        latent = self.encode_state(x)
        action = action.reshape(-1, self.action_dim)
        combined = torch.cat([latent, action], dim=-1)
        return self.transition_head(combined)

    def estimate_value(self, x):
        latent = self.encode_state(x)
        return self.value_head(latent)

    def get_action(self, x, temperature=1.0):
        logits = self.forward(x)
        probs = torch.softmax(logits / max(temperature, 1e-8), dim=-1)
        return probs
