import torch
import torch.nn as nn
import torch.nn.functional as F

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.proj = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        # x: (B, C, T)
        res = self.proj(x)
        out = self.conv(x)
        out = out[:, :, :x.shape[-1]]  # causal truncation
        return self.relu(out + res)

class MRDTModule(nn.Module):
    """
    Multiscale Reliability DNA Twin (MRDT)
    As specified in Section 4.1 and Table 2.
    """
    def __init__(self, d_x=10, d_h=128, d_f=64, d_tau=32, d_rho=16):
        super().__init__()
        self.d_x = d_x
        self.d_h = d_h
        self.d_f = d_f
        self.d_tau = d_tau
        self.d_rho = d_rho

        # A. TCN Encoder & GRU Memory (Eq. 7)
        self.tcn = nn.Sequential(
            TCNBlock(d_x, d_h, kernel_size=3, dilation=1),
            TCNBlock(d_h, d_h, kernel_size=3, dilation=2)
        )
        self.gru = nn.GRUCell(d_h, d_h)

        # B. Multi-Head Attention Degradation Fusion (Eq. 8)
        self.deg_indicator_proj = nn.Linear(d_x, d_f)
        self.attn_weights = nn.Sequential(
            nn.Linear(d_x, d_f),
            nn.Tanh(),
            nn.Linear(d_f, 1)
        )

        # C. Aging Trajectory Learning (LSTM) (Eq. 9-10)
        self.lstm_trajectory = nn.LSTMCell(d_h + d_f + 1, d_tau)

        # D. Bilinear Cross-Asset Correlation (Eq. 11)
        self.bilinear_vec = nn.Parameter(torch.randn(2 * d_h, 1) * 0.1)

        # E. Electro-Thermal Fatigue Miner Accumulation (Eq. 12-13)
        self.stress_coeff = nn.Parameter(torch.tensor([0.25, 0.15, 0.20], dtype=torch.float32))

        # F. Reliability DNA Profiling (MLP) (Eq. 14)
        in_dim_mlp = d_h + d_f + d_tau + 1 + 1
        self.mlp_profiler = nn.Sequential(
            nn.Linear(in_dim_mlp, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, d_rho)
        )

    def forward(self, x_seq, h_prev=None, tau_prev=None, fatigue_prev=None):
        """
        x_seq: (B, T, d_x) or (B, d_x)
        Returns:
            h_t: (B, d_h) reconstructed hidden reliability state
            rho_t: (B, d_rho) reliability DNA profile
            tau_t: (B, d_tau) aging trajectory
            fatigue_t: (B, 1) Miner fatigue accumulation
        """
        if x_seq.dim() == 2:
            x_seq = x_seq.unsqueeze(1) # (B, 1, d_x)
        
        B, T, D = x_seq.shape
        device = x_seq.device

        if h_prev is None:
            h_prev = torch.zeros(B, self.d_h, device=device)
        if tau_prev is None:
            tau_prev = torch.zeros(B, self.d_tau, device=device)
        if fatigue_prev is None:
            fatigue_prev = torch.zeros(B, 1, device=device)

        # 1. TCN Feature extraction
        x_tcn_in = x_seq.transpose(1, 2) # (B, d_x, T)
        tcn_out = self.tcn(x_tcn_in).transpose(1, 2)[:, -1, :] # (B, d_h)

        # GRU update
        h_t = self.gru(tcn_out, h_prev) # (B, d_h)

        # 2. Degradation Fusion (Eq. 8)
        x_last = x_seq[:, -1, :]
        d_indicators = self.deg_indicator_proj(x_last) # (B, d_f)
        attn_a = F.softmax(self.attn_weights(x_last), dim=-1) # (B, 1)
        fused_d = d_indicators * attn_a # (B, d_f)

        # 3. Fatigue Evolution Modeling (Eq. 12-13)
        # Features: Temp (idx 9), Current/Demand (idx 3), SOC rate (idx 2)
        T_i = x_last[:, 9:10]
        I_i = x_last[:, 3:4]
        soc_i = x_last[:, 2:3]
        stress_t = self.stress_coeff[0] * T_i + self.stress_coeff[1] * (I_i ** 2) + self.stress_coeff[2] * torch.abs(soc_i)
        fatigue_t = fatigue_prev + stress_t * 0.001

        # 4. Aging Trajectory Learning (Eq. 9-10)
        lstm_in = torch.cat([h_t, fused_d, fatigue_t], dim=-1)
        h_tau, c_tau = self.lstm_trajectory(lstm_in, (tau_prev, torch.zeros_like(tau_prev)))
        tau_t = h_tau

        # 5. Cross-Asset Correlation (Eq. 11)
        corr_score = torch.sigmoid(torch.matmul(torch.cat([h_t, h_t], dim=-1), self.bilinear_vec)) # (B, 1)

        # 6. DNA Profiling (Eq. 14)
        dna_in = torch.cat([h_t, fused_d, tau_t, corr_score, fatigue_t], dim=-1)
        rho_t = self.mlp_profiler(dna_in)

        return h_t, rho_t, tau_t, fatigue_t
