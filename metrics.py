import torch
import numpy as np

def compute_rps(h_states, h_nominal, sigma_ref=1.0):
    """
    Eq (41): Reliability Preservation Score (RPS)
    RPS = (1/T) sum_{t=1}^T (1/|V|) sum_{i in V} exp( - || h_i(t) - h_i^nominal || / sigma_ref )
    """
    if isinstance(h_states, np.ndarray):
        h_states = torch.tensor(h_states, dtype=torch.float32)
    if isinstance(h_nominal, np.ndarray):
        h_nominal = torch.tensor(h_nominal, dtype=torch.float32)

    diff = torch.norm(h_states - h_nominal, dim=-1)
    exponent = - diff / sigma_ref
    score = torch.exp(exponent).mean().item()
    return float(np.clip(score, 0.0, 1.0))

def compute_bhsi(soh_initial, soh_final):
    """
    Eq (42): Battery Health Sustainability Index (BHSI)
    BHSI = (1 / |V_EV|) sum_{i in V_EV} SOH_i(T) / SOH_i(0)
    """
    ratio = soh_final / (soh_initial + 1e-6)
    score = float(np.mean(ratio))
    return float(np.clip(score, 0.0, 1.0))

def compute_rpr(delta_h_i, delta_h_j):
    """
    Eq (43): Reliability Propagation Resilience (RPR)
    RPR = 1 - (1 / |E|) sum_{(i,j) in E} (delta_h_j / delta_h_i) * 1(delta_h_j > 0)
    """
    delta_h_i = np.asarray(delta_h_i) + 1e-6
    delta_h_j = np.asarray(delta_h_j)
    
    mask = (delta_h_j > 0).astype(float)
    ratio = (delta_h_j / delta_h_i) * mask
    score = 1.0 - float(np.mean(ratio))
    return float(np.clip(score, 0.0, 1.0))

def compute_cirf(t_reliable, t_total):
    """
    Eq (44): Charging Infrastructure Reliability Factor (CIRF)
    CIRF = (1 / |V_CS|) sum_{i in V_CS} t_reliable^(i) / t_total^(i)
    """
    ratio = t_reliable / (t_total + 1e-6)
    score = float(np.mean(ratio))
    return float(np.clip(score, 0.0, 1.0))

def compute_rie(rps_full, rps_no_riim):
    """
    Eq (45): Reliability Immunization Effectiveness (RIE)
    RIE = (RPS_full - RPS_no_riim) / RPS_full
    """
    rie = (rps_full - rps_no_riim) / (rps_full + 1e-6)
    return float(np.clip(rie, 0.0, 1.0))
