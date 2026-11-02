import torch
import torch.nn as nn
import numpy as np

class RAMOOptimizer(nn.Module):
    """
    Reliability-Aware Multi-Objective Optimization (RAMO)
    As specified in Section 4.4 and Table 4.
    """
    def __init__(self, weights=None):
        super().__init__()
        # Default weight configuration (Table 4)
        if weights is None:
            self.raw_weights = torch.tensor([0.25, 0.15, 0.20, 0.25, 0.30, 0.20, 0.15], dtype=torch.float32)
        else:
            self.raw_weights = torch.tensor(weights, dtype=torch.float32)

        self.running_means = torch.ones(7, dtype=torch.float32)

    def normalize_weights(self):
        """
        Eq. (37): Adaptive Weight Normalization
        w_k(t) = (w_tilde_k / J_bar_k(t)) / sum_{m=1}^7 (w_tilde_m / J_bar_m(t))
        """
        raw = self.raw_weights.to(self.running_means.device)
        scaled = raw / (self.running_means + 1e-6)
        norm_w = scaled / torch.sum(scaled)
        return norm_w

    def compute_loss(self, l_deg, l_therm, l_cost, l_risk, l_eff, l_renew, l_sustain):
        """
        Eq. (33): Multi-Objective Loss Formulation
        L = w1*L_deg + w2*L_therm + w3*L_cost + w4*L_risk - w5*L_eff - w6*L_renew - w7*L_sustain
        """
        objs = torch.stack([
            l_deg.mean(), l_therm.mean(), l_cost.mean(), l_risk.mean(),
            l_eff.mean(), l_renew.mean(), l_sustain.mean()
        ])

        # Update running means
        self.running_means = self.running_means.to(l_deg.device)
        self.running_means = 0.9 * self.running_means + 0.1 * objs.detach()

        w = self.normalize_weights().to(l_deg.device)

        total_loss = (w[0] * l_deg.mean() + 
                      w[1] * l_therm.mean() + 
                      w[2] * l_cost.mean() + 
                      w[3] * l_risk.mean() - 
                      w[4] * l_eff.mean() - 
                      w[5] * l_renew.mean() - 
                      w[6] * l_sustain.mean())

        return total_loss, w
