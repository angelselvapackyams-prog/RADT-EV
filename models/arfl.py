import torch
import torch.nn as nn

class ARFLFeedbackLearner(nn.Module):
    """
    Adaptive Reliability Feedback Learning (ARFL)
    As specified in Section 4.5.
    """
    def __init__(self, num_metrics=4, lr=1e-4, conv_eps=1e-4):
        super().__init__()
        self.num_metrics = num_metrics
        self.lr = lr
        self.conv_eps = conv_eps
        self.lambda_coeffs = nn.Parameter(torch.tensor([0.3, 0.3, 0.2, 0.2], dtype=torch.float32))

    def compute_feedback(self, delta_metrics):
        """
        Eq (38): Aggregate feedback signal F(t) = sum_{m=1}^M lambda_m * delta_L_m(t)
        """
        if delta_metrics.dim() == 1:
            delta_metrics = delta_metrics.unsqueeze(0)
        feedback = torch.matmul(delta_metrics, self.lambda_coeffs)
        return feedback

    def check_convergence(self, param_change_norm):
        """
        Eq (40): || theta_{t+1} - theta_t || <= eps_conv
        """
        return param_change_norm <= self.conv_eps
