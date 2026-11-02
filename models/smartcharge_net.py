import torch
import torch.nn as nn
from models.mrdt import MRDTModule
from models.crig import CRIGModule
from models.riim import RIIMModule
from models.ramo import RAMOOptimizer
from models.arfl import ARFLFeedbackLearner

class SmartChargeNetPlus(nn.Module):
    """
    SmartCharge-Net+: Complete Reliability-Aware Meta-Immunized Digital Twin Framework
    As described in Section 4 and Figure 1.
    Integrates MRDT, CRIG, RIIM, RAMO, and ARFL into a unified pipeline.
    """
    def __init__(self, d_x=10, d_h=128, d_rho=16, num_nodes=6, action_dim=5):
        super().__init__()
        self.d_x = d_x
        self.d_h = d_h
        self.d_rho = d_rho
        self.num_nodes = num_nodes
        self.action_dim = action_dim

        # Core Modules
        self.mrdt = MRDTModule(d_x=d_x, d_h=d_h, d_rho=d_rho)
        self.crig = CRIGModule(d_h=d_h, num_nodes=num_nodes)
        self.riim = RIIMModule(state_dim=d_x, rho_dim=d_rho, action_dim=action_dim)
        self.ramo = RAMOOptimizer()
        self.arfl = ARFLFeedbackLearner()

    def forward(self, x_seq, h_nominal=None):
        """
        x_seq: (B, T, d_x) raw sensor observations S(t)
        Returns:
            actions: selected charging control actions a(t)
            rho_t: reliability DNA profiles
            h_t: reconstructed hidden reliability states
            z_graph: CRIG ecosystem intelligence representations
            criticality: node criticality scores
            ramo_loss: total multi-objective optimization loss
        """
        B = x_seq.shape[0]
        device = x_seq.device

        if h_nominal is None:
            h_nominal = torch.zeros(B, self.d_h, device=device)

        # Step 1: MRDT - Reconstruct hidden reliability state H(t) & DNA profile rho_i(t)
        h_t, rho_t, tau_t, fatigue_t = self.mrdt(x_seq)

        # Step 2: CRIG - Graph reasoning, diffusion & vulnerability discovery Z(t)
        # Create multi-node state representation
        h_nodes = h_t.unsqueeze(1).repeat(1, self.num_nodes, 1) # (B, N, d_h)
        z_graph, vulnerability, criticality, K_diff, adj_updated = self.crig(h_nodes)

        # Step 3: RIIM - Policy generation pi*(t)
        x_last = x_seq[:, -1, :]
        actions, probs = self.riim.get_action(x_last, rho_t)

        # Step 4: RAMO - Multi-objective loss evaluation
        # Surrogate loss components
        l_deg = fatigue_t.squeeze(-1)
        l_therm = torch.abs(x_last[:, 9] - 25.0) / 35.0
        l_cost = x_last[:, 6] / 20.0
        l_risk = vulnerability.mean(dim=-1)
        l_eff = x_last[:, 2] # SOC
        l_renew = x_last[:, 4] / 500.0
        l_sustain = 1.0 - l_deg

        ramo_loss, weights = self.ramo.compute_loss(l_deg, l_therm, l_cost, l_risk, l_eff, l_renew, l_sustain)

        return {
            'actions': actions,
            'probs': probs,
            'h_t': h_t,
            'rho_t': rho_t,
            'tau_t': tau_t,
            'fatigue_t': fatigue_t,
            'z_graph': z_graph,
            'vulnerability': vulnerability,
            'criticality': criticality,
            'adj_updated': adj_updated,
            'ramo_loss': ramo_loss,
            'ramo_weights': weights
        }
