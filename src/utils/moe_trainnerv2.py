"""
Training Module for Plant Disease Classification Models.
"""

import os
from datetime import datetime
import logging

import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

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
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        batch_size: int,
        checkpoint_dir: str = "checkpoints",
        warmup_epochs: int = 10,
        min_lr: float = 1e-6,
        val_acc_threshold: float = 1e-5,
        early_stopping_patience: int = 50,
        max_grad_norm: float = 1.0,
        save_best: bool = True,
    ) -> None:

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.checkpoint_dir = checkpoint_dir
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.val_acc_threshold = val_acc_threshold
        self.early_stopping_patience = early_stopping_patience
        self.max_grad_norm = max_grad_norm
        self.save_best = save_best

        self.expert_sample_count = torch.zeros(self.model.num_experts)
        self.expert_class_count = torch.zeros(
            self.model.num_experts, self.model.num_classes
        )

        cosine_epochs = max(num_epochs - warmup_epochs, 1)

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )

        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=cosine_epochs,
            eta_min=min_lr,
        )

        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = os.path.join(self.checkpoint_dir, f"run_{self.run_id}")
        os.makedirs(self.run_dir, exist_ok=True)

        logger.propagate = False
        logger.handlers = [
            h for h in logger.handlers
            if not isinstance(h, logging.FileHandler)
        ]
        file_handler = logging.FileHandler(
            os.path.join(self.run_dir, "training.log")
        )
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

        self.train_loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.lr_history = []

    def _save_checkpoint(self, path: str, epoch: int):

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "train_loss_history": self.train_loss_history,
                "val_loss_history": self.val_loss_history,
                "train_acc_history": self.train_acc_history,
                "val_acc_history": self.val_acc_history,
                "lr_history": self.lr_history,
                "num_classes": self.model.num_classes,
                "num_experts": self.model.num_experts,
                "top_k": self.model.top_k,
                "temperature": self.model.temperature,
            },
            path,
        )

        logger.info(f"Saved checkpoint: {path}")

    def _monitor_expert_usage(self, topk_indices: torch.Tensor, labels: torch.Tensor):

        N = self.model.num_experts
        C = self.model.num_classes

        expert_mask = torch.nn.functional.one_hot(topk_indices, num_classes=N).float()
        expert_mask = expert_mask.sum(dim=1)

        self.expert_sample_count += expert_mask.sum(dim=0).cpu()

        label_onehot = torch.nn.functional.one_hot(labels, num_classes=C).float()

        class_count = expert_mask.T @ label_onehot

        self.expert_class_count += class_count.cpu()

    def train(self):

        best_val_acc = -float("inf")
        best_epoch = -1
        no_improve_count = 0

        for epoch in tqdm.tqdm(range(self.num_epochs), desc="Epochs"):

            # ---------------- TRAIN ----------------
            self.model.train()

            train_running_loss = 0.0
            train_running_correct = 0.0

            for images, labels in self.train_loader:

                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)

                logits, clean_logits, topk_indices = self.model(images)

                loss = self.criterion(
                    logits,
                    labels,
                    clean_logits,
                    topk_indices,
                    self.model.moe_layer.experts,
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm,
                )

                self.optimizer.step()

                with torch.no_grad():

                    self._monitor_expert_usage(
                        topk_indices.detach(),
                        labels.detach(),
                    )

                    probs = torch.softmax(logits, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    acc = accuracy(preds, labels)

                train_running_loss += loss.item()
                train_running_correct += acc

            train_loss = train_running_loss / len(self.train_loader)
            train_acc = train_running_correct / len(self.train_loader)

            self.train_loss_history.append(train_loss)
            self.train_acc_history.append(train_acc)

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.lr_history.append(current_lr)

            logger.info(
                f"Epoch[{epoch+1}/{self.num_epochs}] "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% "
                f"LR: {current_lr:.2e}"
            )

            # ---------------- VALIDATION ----------------

            self.model.eval()

            val_running_loss = 0.0
            val_running_correct = 0.0

            with torch.inference_mode():

                for images, labels in self.val_loader:

                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    logits, clean_logits, topk_indices = self.model(images)

                    loss = self.criterion(
                        logits,
                        labels,
                        clean_logits,
                        topk_indices,
                        self.model.moe_layer.experts,
                    )

                    preds = torch.argmax(logits, dim=1)
                    acc = accuracy(preds, labels)

                    val_running_loss += loss.item()
                    val_running_correct += acc

            validation_loss = val_running_loss / len(self.val_loader)
            validation_acc = val_running_correct / len(self.val_loader)

            self.val_loss_history.append(validation_loss)
            self.val_acc_history.append(validation_acc)

            logger.info(
                f"Epoch[{epoch+1}/{self.num_epochs}] "
                f"Val Loss: {validation_loss:.4f}, "
                f"Val Acc: {validation_acc:.2f}%"
            )

            self.scheduler.step()

            if validation_acc > best_val_acc + self.val_acc_threshold:

                logger.info(
                    f"Validation accuracy improved "
                    f"({best_val_acc:.4f} -> {validation_acc:.4f})."
                )

                best_val_acc = validation_acc
                best_epoch = epoch + 1
                no_improve_count = 0

                if self.save_best:

                    best_path = os.path.join(
                        self.run_dir,
                        "best_checkpoint.pth",
                    )

                    self._save_checkpoint(best_path, epoch + 1)

            else:

                no_improve_count += 1

                logger.info(
                    f"No improvement for {no_improve_count} epoch(s)."
                )

            if no_improve_count >= self.early_stopping_patience:

                logger.info(
                    f"Early stopping triggered."
                )

                break

        last_path = os.path.join(self.run_dir, "last_checkpoint.pth")

        self._save_checkpoint(last_path, epoch + 1)

        logger.info(
            f"Training finished. Best val acc: {best_val_acc:.4f} "
            f"at epoch {best_epoch}"
        )

        # ---------------- PLOTS ----------------

        plt.figure(figsize=(18, 5))

        plt.subplot(1, 3, 1)
        plt.plot(self.train_loss_history, label="train_loss")
        plt.plot(self.val_loss_history, label="val_loss")
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(self.train_acc_history, label="train_acc")
        plt.plot(self.val_acc_history, label="val_acc")
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(self.lr_history)
        plt.title("Learning Rate")
        plt.xlabel("Epoch")
        plt.ylabel("LR")
        plt.yscale("log")

        plt.tight_layout()

        plot_path = os.path.join(self.run_dir, "loss_acc_plot.png")

        plt.savefig(plot_path)
        plt.close()

        # ---------------- EXPERT USAGE ----------------

        plt.figure()

        plt.bar(
            range(self.model.num_experts),
            self.expert_sample_count.numpy(),
        )

        plt.title("Expert Usage")
        plt.xlabel("Expert ID")
        plt.ylabel("Sample Count")

        usage_path = os.path.join(
            self.run_dir,
            "expert_usage.png",
        )

        plt.savefig(usage_path)

        plt.close()

        # ---------------- HEATMAP ----------------

        heatmap = self.expert_class_count.numpy()

        heatmap_norm = heatmap / (
            heatmap.sum(axis=1, keepdims=True) + 1e-9
        )

        plt.figure(figsize=(10, 6))

        im = plt.imshow(
            heatmap_norm,
            cmap="Blues",
        )

        plt.colorbar(im)
        plt.title("Expert-Class Distribution (Normalized)")
        plt.xlabel("Class")
        plt.ylabel("Expert")

        heatmap_path = os.path.join(
            self.run_dir,
            "expert_class_heatmap.png",
        )

        plt.savefig(heatmap_path)

        plt.close()

        logger.info(
            f"Saved plots to {self.run_dir}"
        )
