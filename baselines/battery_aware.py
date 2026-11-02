import torch
import torch.nn as nn


class Battery_Aware_Baseline(nn.Module):
    """
    3. Battery-Aware [23, 24]: Battery Health-Aware Charging Framework
    combining SOC/SOH co-monitoring with empirical degradation trajectory prediction.

    The policy consumes both the raw state and two health-aware signals: a projected
    state-of-charge estimate and a degradation risk estimate. These are learned from
    the same input features to encourage battery-conscious decisions.
    """

    def __init__(self, state_dim=10, action_dim=5, hidden_dim=128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.soc_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.degradation_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.policy = nn.Sequential(
            nn.Linear(state_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def _flatten_inputs(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x.reshape(-1, self.state_dim), x

    def estimate_health_features(self, x):
        flat, _ = self._flatten_inputs(x)
        soc = self.soc_head(flat).squeeze(-1)
        degradation = self.degradation_head(flat).squeeze(-1)
        return torch.stack([soc, degradation], dim=-1)

    def forward(self, x):
        flat, original = self._flatten_inputs(x)
        health_features = self.estimate_health_features(x)
        augmented = torch.cat([flat, health_features], dim=-1)
        logits = self.policy(augmented)
        return logits.reshape(*original.shape[:-1], self.action_dim) if original.dim() > 1 else logits

    def get_action(self, x, temperature=1.0):
        logits = self.forward(x)
        probs = torch.softmax(logits / max(temperature, 1e-8), dim=-1)
        return probs
