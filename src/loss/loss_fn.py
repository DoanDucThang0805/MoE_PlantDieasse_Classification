import torch
import torch.nn as nn
import torch.nn.functional as F


class MoELoss(nn.Module):
    def __init__(self, alpha: float = 0.01):
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()


    def forward(self, logits, targets, clean_logits, top_k_indices):
        """
        logits:         (B, num_classes)
        targets:        (B,)
        clean_logits:   (B, N)
        top_k_indices:  (B, K)
        """

        # ===== Task loss =====
        task_loss = self.ce(logits, targets)

        # ===== Auxiliary loss =====
        aux_loss = self._load_balance_loss(clean_logits, top_k_indices)

        # ===== Total =====
        total_loss = task_loss + self.alpha * aux_loss

        return total_loss


    def _load_balance_loss(self, clean_logits, top_k_indices):
        B, N = clean_logits.shape
        K = top_k_indices.shape[1]

        # ----- P_i: router intention -----
        probs = torch.softmax(clean_logits, dim=-1)  # (B, N)
        P = probs.mean(dim=0)
        P = P / (P.sum() + 1e-9)

        # ----- f_i: actual routing -----
        expert_mask = F.one_hot(top_k_indices, num_classes=N)  # (B, K, N)
        expert_mask = expert_mask.sum(dim=1).float()           # (B, N)

        f = expert_mask.mean(dim=0) / K
        f = f / (f.sum() + 1e-9)

        # ----- loss -----
        loss = N * torch.sum(f * P)

        return loss
    