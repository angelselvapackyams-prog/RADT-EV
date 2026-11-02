import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReplayBuffer:
    def __init__(self, capacity=10000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []

    def push(self, state, action, reward, next_state, rho, priority=1.0):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
            self.priorities.pop(0)
        self.buffer.append((state, action, reward, next_state, rho))
        self.priorities.append(priority)

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return None
        prios = torch.tensor(self.priorities, dtype=torch.float32) ** self.alpha
        probs = prios / prios.sum()
        indices = torch.multinomial(probs, min(batch_size, len(self.buffer)), replacement=False)
        
        batch = [self.buffer[i] for i in indices]
        states = torch.stack([b[0] for b in batch])
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        next_states = torch.stack([b[3] for b in batch])
        rhos = torch.stack([b[4] for b in batch])

        return states, actions, rewards, next_states, rhos, indices

class QNetwork(nn.Module):
    def __init__(self, state_dim, rho_dim, action_dim):
        super().__init__()
        in_dim = state_dim + rho_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, state, rho):
        x = torch.cat([state, rho], dim=-1)
        return self.net(x)

class RIIMModule(nn.Module):
    """
    Reliability Immunization Intelligence Mechanism (RIIM)
    As specified in Section 4.3 and Table 3.
    """
    def __init__(self, state_dim=10, rho_dim=16, action_dim=5, ensemble_k=5, 
                 meta_lr=1e-3, inner_lr=5e-3, gamma=0.99, online_lr=1e-4):
        super().__init__()
        self.state_dim = state_dim
        self.rho_dim = rho_dim
        self.action_dim = action_dim
        self.ensemble_k = ensemble_k
        self.gamma = gamma

        # Policy Network MLP(256, 128, 64) -> Softmax (Table 3)
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim + rho_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

        # Ensemble of K Q-networks for Uncertainty Estimation (Eq. 30)
        self.q_ensemble = nn.ModuleList([
            QNetwork(state_dim, rho_dim, action_dim) for _ in range(ensemble_k)
        ])

        self.memory = ReplayBuffer(capacity=10000)
        self.gamma_risk = nn.Parameter(torch.tensor([0.4, 0.4, 0.2], dtype=torch.float32))

    def get_action(self, state, rho, deterministic=False):
        """
        Policy network prediction (Eq. 29)
        """
        x = torch.cat([state, rho], dim=-1)
        logits = self.policy_net(x)
        probs = F.softmax(logits, dim=-1)

        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            action = torch.multinomial(probs, 1).squeeze(-1)
        return action, probs

    def evaluate_uncertainty(self, state, rho):
        """
        Uncertainty estimation across K Q-networks (Eq. 30)
        U(s, rho) = (1/K) sum_{k=1}^K (Q_k(s, rho) - Q_mean(s, rho))^2
        """
        q_vals = torch.stack([q_net(state, rho) for q_net in self.q_ensemble], dim=0) # (K, B, action_dim)
        q_mean = q_vals.mean(dim=0) # (B, action_dim)
        uncertainty = torch.mean((q_vals - q_mean) ** 2, dim=0) # (B, action_dim)
        return uncertainty, q_mean

    def compute_reliability_risk(self, h_curr, h_nominal, h_future=None, neighbor_risk=0.0):
        """
        Eq. (25): Reliability Risk Assessment
        """
        diff_curr = torch.norm(h_curr - h_nominal, dim=-1)
        if h_future is not None:
            diff_fut = torch.norm(h_future - h_nominal, dim=-1)
        else:
            diff_fut = diff_curr * 1.1

        risk = self.gamma_risk[0] * diff_curr + self.gamma_risk[1] * diff_fut + self.gamma_risk[2] * neighbor_risk
        return risk
