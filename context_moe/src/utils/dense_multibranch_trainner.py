"""
Trainer for dense multi-branch classification models.

The MobileNetV3SmallDenseMultiBranch model runs every expert branch and averages
their logits, so the training objective is the same cross-entropy objective used
for a normal classifier. Gradients from the averaged logits still update all
branches.
"""

import logging
import os
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import torch
import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader

from metric.metric import accuracy


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DenseMultiBranchTrainer:
    """
    Training loop for dense multi-branch models.

    Expected dataloader batch format matches the project dataset:
        (images, labels, context)

    Context is ignored because dense multi-branch models do not route samples by
    context; all branches are evaluated for every image.
    """

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
        checkpoints_dir: str = "checkpoints",
        lr_reduction_rate: float = 0.5,
        min_lr: float = 1e-7,
        lr_reduction_patience: int = 10,
        val_acc_threshold: float = 1e-5,
        early_stopping_patience: int = 50,
        max_grad_norm: float | None = 1.0,
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
        self.checkpoints_dir = checkpoints_dir
        self.lr_reduction_rate = lr_reduction_rate
        self.min_lr = min_lr
        self.lr_reduction_patience = lr_reduction_patience
        self.val_acc_threshold = val_acc_threshold
        self.early_stopping_patience = early_stopping_patience
        self.max_grad_norm = max_grad_norm
        self.save_best = save_best

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=lr_reduction_rate,
            patience=lr_reduction_patience,
            threshold=val_acc_threshold,
            min_lr=min_lr,
        )

        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = os.path.join(self.checkpoints_dir, f"run_{self.run_id}")
        os.makedirs(self.run_dir, exist_ok=True)

        logger.propagate = False
        logger.handlers = [
            handler for handler in logger.handlers
            if not isinstance(handler, logging.FileHandler)
        ]
        file_handler = logging.FileHandler(os.path.join(self.run_dir, "training.log"))
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

        self.train_loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.lr_history = []

    def _num_experts(self) -> int | None:
        experts = getattr(self.model, "experts", None)
        if experts is None:
            return getattr(self.model, "num_experts", None)
        return len(experts)

    def _extract_logits(self, model_output: Any) -> torch.Tensor:
        """
        Return final logits from common model output formats.

        Current dense multi-branch model returns a tensor directly. This also
        supports future variants that may return (logits, branch_logits) or a dict
        with a "logits" key.
        """
        if torch.is_tensor(model_output):
            return model_output

        if isinstance(model_output, dict):
            if "logits" in model_output:
                return model_output["logits"]
            if "final_output" in model_output:
                return model_output["final_output"]

        if isinstance(model_output, (tuple, list)) and model_output:
            first_output = model_output[0]
            if torch.is_tensor(first_output):
                return first_output

        raise TypeError(
            "DenseMultiBranchTrainer expected model output to be logits, "
            "a tuple/list whose first item is logits, or a dict containing logits."
        )

    def _save_checkpoint(self, path: str, epoch: int) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_loss_history": self.train_loss_history,
            "val_loss_history": self.val_loss_history,
            "train_acc_history": self.train_acc_history,
            "val_acc_history": self.val_acc_history,
            "lr_history": self.lr_history,
            "batch_size": self.batch_size,
            "num_experts": self._num_experts(),
        }

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")

    def _run_one_epoch(self) -> tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        running_correct = 0.0

        for images, labels, _ in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            logits = self._extract_logits(self.model(images))
            loss = self.criterion(logits, labels)
            loss.backward()

            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm,
                )

            self.optimizer.step()

            with torch.no_grad():
                preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
                acc = accuracy(preds, labels)

            running_loss += loss.item()
            running_correct += acc

        return (
            running_loss / len(self.train_loader),
            running_correct / len(self.train_loader),
        )

    def _validate(self) -> tuple[float, float]:
        self.model.eval()
        running_loss = 0.0
        running_correct = 0.0

        with torch.inference_mode():
            for images, labels, _ in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self._extract_logits(self.model(images))
                loss = self.criterion(logits, labels)
                preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
                acc = accuracy(preds, labels)

                running_loss += loss.item()
                running_correct += acc

        return (
            running_loss / len(self.val_loader),
            running_correct / len(self.val_loader),
        )

    def train(self) -> None:
        best_val_acc = -float("inf")
        best_epoch = -1
        no_improve_count = 0
        last_epoch = 0

        for epoch in tqdm.tqdm(range(self.num_epochs), desc="Epochs"):
            last_epoch = epoch + 1

            train_loss, train_acc = self._run_one_epoch()
            self.train_loss_history.append(train_loss)
            self.train_acc_history.append(train_acc)

            validation_loss, validation_acc = self._validate()
            self.val_loss_history.append(validation_loss)
            self.val_acc_history.append(validation_acc)

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.lr_history.append(current_lr)

            logger.info(
                f"Epoch[{last_epoch}/{self.num_epochs}] "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% "
                f"Val Loss: {validation_loss:.4f}, Val Acc: {validation_acc:.2f}% "
                f"LR: {current_lr:.2e}"
            )

            self.scheduler.step(validation_acc)

            if validation_acc > best_val_acc + self.val_acc_threshold:
                logger.info(
                    f"Validation accuracy improved "
                    f"({best_val_acc:.4f} -> {validation_acc:.4f})."
                )
                best_val_acc = validation_acc
                best_epoch = last_epoch
                no_improve_count = 0

                if self.save_best:
                    best_path = os.path.join(self.run_dir, "best_checkpoint.pth")
                    self._save_checkpoint(best_path, last_epoch)
            else:
                no_improve_count += 1
                logger.info(f"No improvement for {no_improve_count} epoch(s).")

            if no_improve_count >= self.early_stopping_patience:
                logger.info(
                    "Early stopping triggered. "
                    f"No improvement in validation acc for "
                    f"{self.early_stopping_patience} epochs."
                )
                break

        last_path = os.path.join(self.run_dir, "last_checkpoint.pth")
        self._save_checkpoint(last_path, last_epoch)

        logger.info(
            f"Training finished. Best val acc: {best_val_acc:.4f} "
            f"at epoch {best_epoch}"
        )

        self._save_plots()

    def _save_plots(self) -> None:
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

        logger.info(f"Saved training plot to {plot_path}")
