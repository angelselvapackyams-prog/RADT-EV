import torch
import torch.nn as nn


class DRL_Baseline(nn.Module):
    """
    5. DRL [27]: Interpretable Deep Reinforcement Learning with Imitative Expert Experience
    for Smart EV Charging.

    This re-implementation uses a shared encoder with a policy head and a value head,
    plus a small imitation head to mimic expert behavior in a differentiable way.
    """

    def __init__(self, state_dim=10, action_dim=5, hidden_dim=128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.shared_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        self.imitation_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def _flatten_inputs(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x.reshape(-1, self.state_dim), x

    def encode_state(self, x):
        flat, _ = self._flatten_inputs(x)
        return self.shared_encoder(flat)

    def forward(self, x):
        latent = self.encode_state(x)
        logits = self.policy_head(latent)
        return logits.reshape(*x.shape[:-1], self.action_dim) if x.dim() > 1 else logits

    def estimate_value(self, x):
        latent = self.encode_state(x)
        return self.value_head(latent)

    def imitation_logits(self, x):
        latent = self.encode_state(x)
        return self.imitation_head(latent)

    def get_action(self, x, temperature=1.0):
        logits = self.forward(x)
        probs = torch.softmax(logits / max(temperature, 1e-8), dim=-1)
        return probs
