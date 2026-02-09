import torch
import torch.nn as nn
import torch.nn.functional as F


class MoeLoss(nn.Module):
    """
    Production-ready MoE loss:
        L_total = L_task + alpha * L_aux

    Inputs:
        logits:         (B, num_classes)
        targets:        (B,)
        router_output:  (B, num_experts)   # softmax probs
        topk_indices:   (B, topk)

    Usage:
        loss_fn = MoeLoss(num_experts=3, alpha=0.01)
        loss = loss_fn(logits, y, router_output, topk_indices)
        loss.backward()
    """

    def __init__(self, num_experts: int, alpha: float = 0.01):
        super().__init__()
        self.num_experts = num_experts
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets, router_output, topk_indices):
        # ===== Task loss =====
        task_loss = self.ce(logits, targets)

        # ===== Load balancing loss =====
        aux_loss = self._load_balance_loss(router_output, topk_indices)

        total_loss = task_loss + self.alpha * aux_loss
        return total_loss

    def _load_balance_loss(self, router_output, topk_indices):
        """
        router_output: (B, N)
        topk_indices:  (B, K)
        """

        B, N = router_output.shape

        # ----- P_i: average router probability per expert -----
        # (N,)
        P = router_output.mean(dim=0)

        # ----- f_i: fraction of tokens sent to each expert -----
        # Convert topk indices -> one-hot mask
        # (B, K, N)
        expert_mask = F.one_hot(topk_indices, num_classes=N)

        # Sum over topk -> (B, N)
        expert_mask = expert_mask.sum(dim=1).float()

        # Average over batch -> (N,)
        f = expert_mask.mean(dim=0)

        # ----- Aux loss -----
        aux_loss = N * torch.sum(f * P)

        return aux_loss
