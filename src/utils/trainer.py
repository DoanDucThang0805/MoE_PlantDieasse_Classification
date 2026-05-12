"""
Training Module for Plant Disease Classification Models.

This module provides a comprehensive Trainer class for training neural networks on 
plant disease classification tasks. It includes features like learning rate scheduling,
early stopping, checkpoint management, and training visualization.

Features:
    - Learning rate reduction on plateau
    - Early stopping mechanism
    - Checkpoint saving (best and last models)
    - Training history tracking and visualization
    - Automatic logging to file
"""

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

class Trainer:
    """
    Training orchestrator for neural network models.
    
    Manages the complete training pipeline including forward/backward passes,
    validation, learning rate scheduling, early stopping, and checkpoint management.
    Automatically tracks and visualizes training metrics.
    
    Attributes:
        num_epochs (int): Total number of training epochs
        device (torch.device): Device to train on (CPU or CUDA)
        model (nn.Module): Neural network model to train
        optimizer (optim.Optimizer): Optimization algorithm
        criterion (nn.Module): Loss function
        scheduler (torch.optim.lr_scheduler.ReduceLROnPlateau): Learning rate scheduler
        run_dir (str): Directory to save checkpoints and logs
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
        checkpoints_dir: str = 'checkpoints',
        lr_reduction_rate: float = 0.5,          # 50%
        min_lr: float = 1e-7,
        lr_reduction_patience: int = 10,
        val_acc_threshold: float = 1e-5,
        early_stopping_patience: int = 50,
        save_best: bool = True
    ) -> None:
        """
        Initialize the Trainer.
        
        Args:
            num_epochs (int): Number of training epochs
            device (torch.device): Device to train on (torch.device("cuda") or torch.device("cpu"))
            train_loader (DataLoader): DataLoader for training dataset
            val_loader (DataLoader): DataLoader for validation dataset
            model (nn.Module): Neural network model to train
            criterion (nn.Module): Loss function (e.g., nn.CrossEntropyLoss())
            optimizer (optim.Optimizer): Optimizer (e.g., torch.optim.Adam())
            batch_size (int): Batch size for training
            checkpoints_dir (str, optional): Directory to save checkpoints. Defaults to 'checkpoints'.
            lr_reduction_rate (float, optional): Factor to reduce learning rate. Defaults to 0.5.
            min_lr (float, optional): Minimum learning rate. Defaults to 1e-7.
            lr_reduction_patience (int, optional): Patience for LR reduction. Defaults to 10.
            val_acc_threshold (float, optional): Threshold for validation improvement. Defaults to 1e-5.
            early_stopping_patience (int, optional): Patience for early stopping. Defaults to 50.
            save_best (bool, optional): Whether to save best model. Defaults to True.
        """
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
        self.save_best = save_best

        # scheduler: reduce LR on plateau, monitoring validation accuracy (mode='max')
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=lr_reduction_rate,
            patience=lr_reduction_patience,
            threshold=val_acc_threshold,
            min_lr=min_lr,
        )

        # run_id & run_dir
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = os.path.join(self.checkpoints_dir, f"run_{self.run_id}")
        os.makedirs(self.run_dir, exist_ok=True)

        if logger.hasHandlers():
            logger.handlers.clear()
            
        # logging file
        file_handler = logging.FileHandler(os.path.join(self.run_dir, 'training.log'))
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

        # history
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

    def _save_checkpoint(self, path: str, epoch: int):
        """
        Save model checkpoint.
        
        Saves the model state, optimizer state, and training history to a checkpoint file.
        
        Args:
            path (str): File path to save the checkpoint
            epoch (int): Current epoch number
        """
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_loss_history": self.train_loss_history,
            "val_loss_history": self.val_loss_history,
            "train_acc_history": self.train_acc_history,
            "val_acc_history": self.val_acc_history
        }, path)
        logger.info(f"Saved checkpoint: {path}")

    def train(self):
        """
        Execute the complete training loop.
        
        Performs training with validation for each epoch, implements learning rate scheduling,
        early stopping, and saves the best and last checkpoints. Generates and saves a 
        training visualization plot at the end.
        
        The training loop:
        1. Trains model on training set
        2. Validates on validation set
        3. Updates learning rate based on validation accuracy
        4. Implements early stopping if no improvement
        5. Saves best and last model checkpoints
        6. Visualizes loss and accuracy trends
        """
        best_val_acc = -float("inf")
        best_epoch = -1
        no_improve_count = 0

        for epoch in tqdm.tqdm(range(self.num_epochs), desc="Epochs"):
            # ---- Train ----
            self.model.train()
            train_running_loss = 0.0
            train_running_correct = 0.0
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                logits = self.model(images)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                loss = self.criterion(logits, labels)
                acc = accuracy(preds, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_running_loss += loss.item()
                train_running_correct += acc
            train_loss = train_running_loss / len(self.train_loader)
            train_acc = train_running_correct / len(self.train_loader)
            self.train_loss_history.append(train_loss)
            self.train_acc_history.append(train_acc)
            logger.info(f"Epoch[{epoch+1}/{self.num_epochs}] "
                        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

            # ---- Validation ----
            self.model.eval()
            val_running_loss = 0.0
            val_running_correct = 0.0
            with torch.inference_mode():
                for images, labels in self.val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    logits = self.model(images)
                    probs = torch.softmax(logits, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    loss = self.criterion(logits, labels)
                    acc = accuracy(preds, labels)
                    val_running_loss += loss.item()
                    val_running_correct += acc
            validation_loss = val_running_loss / len(self.val_loader)
            validation_acc = val_running_correct / len(self.val_loader)
            self.val_acc_history.append(validation_acc)
            self.val_loss_history.append(validation_loss)
            logger.info(f"Epoch[{epoch + 1}/{self.num_epochs}] "
                        f"Val Loss: {validation_loss:.4f}, Val Acc: {validation_acc:.2f}%")

            # ---- Scheduler step (monitor validation accuracy) ----
            # ReduceLROnPlateau expects the metric (we use validation_acc)
            try:
                self.scheduler.step(validation_acc)
            except Exception as e:
                # fail-safe - scheduler usually accepts a scalar
                logger.warning(f"Scheduler step failed: {e}")

            # ---- Check improvement and save best model ----
            if validation_acc > best_val_acc + self.val_acc_threshold:
                logger.info(f"Validation accuracy improved ({best_val_acc:.4f} -> {validation_acc:.4f}).")
                best_val_acc = validation_acc
                best_epoch = epoch + 1
                no_improve_count = 0
                if self.save_best:
                    best_path = os.path.join(self.run_dir, "best_checkpoint.pth")
                    self._save_checkpoint(best_path, epoch + 1)
            else:
                no_improve_count += 1
                logger.info(f"No improvement for {no_improve_count} epoch(s).")

            # ---- Early stopping ----
            if no_improve_count >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered. No improvement in validation acc for {self.early_stopping_patience} epochs.")
                break

        # ---- Save last checkpoint ----
        last_path = os.path.join(self.run_dir, "last_checkpoint.pth")
        self._save_checkpoint(last_path, epoch + 1)

        logger.info(f"Training finished. Best val acc: {best_val_acc:.4f} at epoch {best_epoch}")

        # ---- Plot loss/acc ----
        plt.figure(figsize=(12, 8))
        plt.subplot(1, 2, 1)
        plt.plot(self.train_loss_history, label="train_loss")
        plt.plot(self.val_loss_history, label="val_loss")
        plt.title("Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss Values")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.train_acc_history, label="train_acc")
        plt.plot(self.val_acc_history, label="val_acc")
        plt.title("Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.legend()

        plt.tight_layout()
        plot_path = os.path.join(self.run_dir, "loss_acc_plot.png")
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Saved training plot to {plot_path}")

