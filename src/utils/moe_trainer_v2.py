import os
from datetime import datetime
import logging
import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from metric.metric import accuracy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MoETrainer:
    def __init__(
        self,
        num_epochs: int,
        device: torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model: nn.Module,
        num_experts: int,
        num_classes: int,   # ✅ thêm
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        batch_size: int,
        checkpoint_dir: str = 'checkpoints',
        lr_reduction_rate: float = 0.5,
        min_lr: float = 1e-7,
        lr_reduction_patience: int = 10,
        val_acc_threshold: float = 1e-5,
        early_stopping_patience: int = 50,
        save_best: bool = True,
        use_weighted_routing: bool = True   # ✅ chọn mode
    ):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model.to(device)
        self.num_experts = num_experts
        self.num_classes = num_classes
        self.criterion = criterion
        self.optimizer = optimizer
        self.checkpoint_dir = checkpoint_dir
        self.use_weighted_routing = use_weighted_routing

        # ✅ thống kê expert
        self.expert_sample_count = torch.zeros(num_experts)
        self.expert_class_count = torch.zeros(num_experts, num_classes)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=lr_reduction_rate,
            patience=lr_reduction_patience,
            threshold=val_acc_threshold,
            min_lr=min_lr,
        )

        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = os.path.join(self.checkpoint_dir, f"run_{self.run_id}")
        os.makedirs(self.run_dir, exist_ok=True)

        if logger.hasHandlers():
            logger.handlers.clear()

        file_handler = logging.FileHandler(os.path.join(self.run_dir, 'training.log'))
        logger.addHandler(file_handler)

        self.train_loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

    # =========================================================
    # 🔥 Expert Statistics Update
    # =========================================================
    def _update_expert_stats(self, topk_indices, labels, combined_weight=None):
        B, K = topk_indices.shape

        for b in range(B):
            for k in range(K):
                expert_id = topk_indices[b, k].item()
                label = labels[b].item()

                if self.use_weighted_routing and combined_weight is not None:
                    weight = combined_weight[b, k].item()
                else:
                    weight = 1.0

                self.expert_sample_count[expert_id] += weight
                self.expert_class_count[expert_id, label] += weight

    # =========================================================
    # 🔥 Visualization
    # =========================================================
    def _plot_expert_stats(self):
        # ---- Expert usage ----
        plt.figure()
        plt.bar(range(self.num_experts), self.expert_sample_count.numpy())
        plt.xlabel("Expert ID")
        plt.ylabel("Number of samples")
        plt.title("Expert Usage")
        plt.savefig(os.path.join(self.run_dir, "expert_usage.png"))
        plt.close()

        # ---- Heatmap ----
        plt.figure(figsize=(10, 6))
        plt.imshow(self.expert_class_count.numpy(), aspect='auto')
        plt.colorbar()
        plt.xlabel("Class ID")
        plt.ylabel("Expert ID")
        plt.title("Expert vs Class Distribution")
        plt.savefig(os.path.join(self.run_dir, "expert_class_heatmap.png"))
        plt.close()

        # ---- Normalized heatmap ----
        norm = self.expert_class_count / (
            self.expert_class_count.sum(dim=1, keepdim=True) + 1e-9
        )

        plt.figure(figsize=(10, 6))
        plt.imshow(norm.numpy(), aspect='auto')
        plt.colorbar()
        plt.xlabel("Class ID")
        plt.ylabel("Expert ID")
        plt.title("Normalized Expert Specialization")
        plt.savefig(os.path.join(self.run_dir, "expert_class_normalized.png"))
        plt.close()

        logger.info("Saved expert analysis plots.")

    # =========================================================
    # 🔥 Training Loop
    # =========================================================
    def train(self):
        best_val_acc = -float("inf")
        no_improve_count = 0

        for epoch in tqdm.tqdm(range(self.num_epochs), desc="Epochs"):

            # ================= TRAIN =================
            self.model.train()
            train_loss, train_acc = 0, 0

            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                logits, combined_weight, topk_indices = self.model(images)

                loss = self.criterion(logits, labels, combined_weight, topk_indices)

                preds = torch.argmax(logits, dim=1)
                acc = accuracy(preds, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # ✅ update expert stats
                self._update_expert_stats(topk_indices, labels, combined_weight)

                train_loss += loss.item()
                train_acc += acc

            train_loss /= len(self.train_loader)
            train_acc /= len(self.train_loader)

            self.train_loss_history.append(train_loss)
            self.train_acc_history.append(train_acc)

            # ================= VALID =================
            self.model.eval()
            val_loss, val_acc = 0, 0

            with torch.inference_mode():
                for images, labels in self.val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)

                    logits, combined_weight, topk_indices = self.model(images)

                    loss = self.criterion(logits, labels, combined_weight, topk_indices)

                    preds = torch.argmax(logits, dim=1)
                    acc = accuracy(preds, labels)

                    val_loss += loss.item()
                    val_acc += acc

            val_loss /= len(self.val_loader)
            val_acc /= len(self.val_loader)

            self.val_loss_history.append(val_loss)
            self.val_acc_history.append(val_acc)

            logger.info(
                f"Epoch[{epoch+1}] "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )

            # ================= Scheduler =================
            self.scheduler.step(val_acc)

            # ================= Early stopping =================
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= self.early_stopping_patience:
                logger.info("Early stopping triggered.")
                break

        # ================= AFTER TRAIN =================
        self._plot_expert_stats()