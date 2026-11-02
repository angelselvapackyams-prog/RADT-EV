import torch
import torch.nn as nn


class Meta_RL_Baseline(nn.Module):
    """
    6. Meta-RL [28]: Meta-Reinforcement Learning Based Adaptive Energy Storage Control
    Meets Dynamic Scenarios.

    The implementation uses a task-context encoder followed by an adaptation module,
    producing a compact, scenario-specific representation before the policy head maps
    the augmented state to a charging action.
    """

    def __init__(self, state_dim=10, action_dim=5, hidden_dim=128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.context_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.adaptation_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(state_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def _flatten_inputs(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x.reshape(-1, self.state_dim), x

    def encode_context(self, x):
        flat, _ = self._flatten_inputs(x)
        context = self.context_encoder(flat)
        adaptation = torch.tanh(self.adaptation_head(flat))
        return context + adaptation

    def forward(self, x):
        flat, original = self._flatten_inputs(x)
        context = self.encode_context(x)
        augmented = torch.cat([flat, context], dim=-1)
        logits = self.policy_head(augmented)
        return logits.reshape(*original.shape[:-1], self.action_dim) if original.dim() > 1 else logits

    def get_action(self, x, temperature=1.0):
        logits = self.forward(x)
        probs = torch.softmax(logits / max(temperature, 1e-8), dim=-1)
        return probs
