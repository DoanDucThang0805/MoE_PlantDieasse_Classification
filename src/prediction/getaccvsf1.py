"""
Model evaluation and benchmarking module for plant disease classification.

This module provides utilities for evaluating pretrained models and MoE-based models
on plant disease classification tasks. It computes accuracy and F1 scores across
multiple model runs and exports results to CSV format.

Author: Plant Disease Classification Team
Date: 2024
"""

import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from dataset.plantdoc_dataset import test_dataset
from models.pretrained_model.efficientnetv2m import model as efficientnetv2m
from models.pretrained_model.efficientnetv2s import model as efficientnetv2s
from models.pretrained_model.mobilenetv3_large import model as mobilenetv3_large
from models.pretrained_model.mobilenetv3_small import model as mobilenetv3_small
from models.pretrained_model.resnet50 import model as resnet50
from models.pretrained_model.shufflenet import model as shufflenet
from models.pretrained_model.squeezenet import model as squeezenet
from models.pretrained_model.vgg16 import model as vgg16

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_BATCH_SIZE = 32
CHECKPOINT_FILENAME = "best_checkpoint.pth"
BEST_METRIC_COLUMN = "mean_macro_f1"


class ModelBenchmarker:
    """
    Evaluate and benchmark multiple model checkpoints on plant disease classification.

    This class loads pretrained models and their checkpoints, evaluates them on a test
    dataset, and computes performance metrics (accuracy and macro F1 score). It supports
    both individual model checkpoints and multiple runs of the same model for statistical
    analysis.

    Attributes:
        checkpoint_directory (Path): Root directory containing model checkpoints
        model_type_category (str): Category of models ("MoE" or "pretrain_weight")
        model_dict (Dict[str, torch.nn.Module]): Dictionary mapping model names to model instances
        test_dataloader (DataLoader): DataLoader for the test dataset
        device (torch.device): Device for model inference (cuda or cpu)
    """

    def __init__(
        self,
        checkpoint_directory: Path,
        model_type_category: str = "pretrain_weight",
        model_dict: Optional[Dict[str, torch.nn.Module]] = None,
        test_dataloader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the ModelBenchmarker.

        Args:
            checkpoint_directory (Path): Root directory containing model checkpoints
            model_type_category (str): Type of model category, either "MoE" or "pretrain_weight".
                                       Defaults to "pretrain_weight"
            model_dict (Dict[str, torch.nn.Module], optional): Dictionary mapping model names
                                                                to model instances
            test_dataloader (DataLoader, optional): DataLoader for test dataset
            device (torch.device, optional): Device for computation (cuda or cpu)
        """
        self.checkpoint_directory = checkpoint_directory
        self.model_type_category = model_type_category
        self.model_dict = model_dict
        self.test_dataloader = test_dataloader
        self.device = device

    def get_model_checkpoint_paths(
        self,
        model_type_directory: Path
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Discover all model checkpoints in the directory structure.

        Recursively searches through the directory structure to find checkpoint files
        organized by model name and runtime. Expected structure:
        model_type_directory/
            model_name_1/
                runtime_1/
                    best_checkpoint.pth
                runtime_2/
                    best_checkpoint.pth
            model_name_2/
                ...

        Args:
            model_type_directory (Path): Directory containing model subdirectories

        Returns:
            Dict[str, List[Dict[str, str]]]: Dictionary mapping model names to list
                                            of checkpoint metadata dicts containing
                                            'runtime' and 'checkpoint_path' keys

        Example:
            {
                "resnet50": [
                    {"runtime": "run_1", "checkpoint_path": "/path/to/checkpoint.pth"},
                    {"runtime": "run_2", "checkpoint_path": "/path/to/checkpoint.pth"}
                ],
                "vgg16": [...]
            }
        """
        checkpoints = {}
        model_names = os.listdir(model_type_directory)
        logger.info(f"Found model directories: {model_names}")

        for model_name in model_names:
            model_checkpoints_directory = os.path.join(
                model_type_directory, model_name
            )

            if not os.path.isdir(model_checkpoints_directory):
                logger.warning(
                    f"Skipping {model_checkpoints_directory}: not a directory"
                )
                continue

            checkpoints[model_name] = []

            for runtime_name in sorted(os.listdir(model_checkpoints_directory)):
                runtime_directory = os.path.join(
                    model_checkpoints_directory, runtime_name
                )

                if not os.path.isdir(runtime_directory):
                    continue

                checkpoint_path = os.path.join(
                    runtime_directory, CHECKPOINT_FILENAME
                )

                if os.path.isfile(checkpoint_path):
                    checkpoints[model_name].append(
                        {
                            "runtime": runtime_name,
                            "checkpoint_path": checkpoint_path
                        }
                    )
                    logger.info(
                        f"Found checkpoint for {model_name} at {checkpoint_path}"
                    )
                else:
                    logger.warning(
                        f"No checkpoint found at expected path: {checkpoint_path}"
                    )

        return checkpoints

    def load_checkpoint(
        self,
        checkpoint_path: str
    ) -> Optional[Dict]:
        """
        Load a model checkpoint from disk.

        Args:
            checkpoint_path (str): Path to the checkpoint file (.pth)

        Returns:
            Optional[Dict]: Loaded checkpoint dictionary containing model state,
                           or None if checkpoint doesn't exist

        Raises:
            RuntimeError: If checkpoint file is corrupted or cannot be loaded
        """
        if not os.path.isfile(checkpoint_path):
            logger.error(f"Checkpoint file does not exist: {checkpoint_path}")
            return None

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cuda")
            logger.info(f"Successfully loaded checkpoint from {checkpoint_path}")
            return checkpoint
        except Exception as error:
            logger.error(
                f"Failed to load checkpoint from {checkpoint_path}: {str(error)}"
            )
            return None

    @torch.inference_mode()
    def evaluate_checkpoint(
        self,
        model: torch.nn.Module,
        checkpoint_path: str
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Evaluate a single model checkpoint on the test dataset.

        Loads the checkpoint into the model, runs inference on all test samples,
        and computes accuracy and macro F1 score.

        Args:
            model (torch.nn.Module): Model instance to evaluate
            checkpoint_path (str): Path to the checkpoint file

        Returns:
            Tuple[Optional[float], Optional[float]]: Tuple of (accuracy, macro_f1_score).
                                                    Returns (None, None) if evaluation fails

        Example:
            >>> accuracy, f1 = benchmarker.evaluate_checkpoint(model, checkpoint_path)
            >>> print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        """
        checkpoint = self.load_checkpoint(checkpoint_path)

        if checkpoint is None:
            return None, None

        try:
            model.load_state_dict(checkpoint["model_state_dict"])
        except KeyError as error:
            logger.error(
                f"Checkpoint missing 'model_state_dict' key: {str(error)}"
            )
            return None, None

        model.to(self.device)
        model.eval()

        all_predictions = []
        all_labels = []

        for images, labels in self.test_dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())

        all_predictions_array = torch.cat(all_predictions).numpy()
        all_labels_array = torch.cat(all_labels).numpy()

        accuracy = accuracy_score(all_labels_array, all_predictions_array)
        macro_f1 = f1_score(
            all_labels_array, all_predictions_array, average="macro"
        )

        logger.info(
            f"{checkpoint_path} -> Accuracy={accuracy:.4f} | MacroF1={macro_f1:.4f}"
        )

        torch.cuda.empty_cache()

        return accuracy, macro_f1

    def __call__(self) -> Dict[str, Dict[str, float]]:
        """
        Execute the full benchmarking pipeline.

        Retrieves all checkpoints, evaluates each model across all its runs,
        and computes mean and standard deviation of metrics.

        Returns:
            Dict[str, Dict[str, float]]: Dictionary mapping model names to their
                                        performance statistics with keys:
                                        - mean_accuracy
                                        - std_accuracy
                                        - mean_macro_f1
                                        - std_macro_f1

        Example:
            >>> results = benchmarker()
            >>> for model_name, metrics in results.items():
            ...     print(f"{model_name}: Acc={metrics['mean_accuracy']:.4f}")
        """
        results = {}
        model_type_directory = os.path.join(
            self.checkpoint_directory, self.model_type_category
        )

        checkpoints = self.get_model_checkpoint_paths(model_type_directory)

        for model_name, model_checkpoint_runs in checkpoints.items():
            if model_name not in self.model_dict:
                logger.warning(
                    f"Model '{model_name}' not found in model dictionary"
                )
                continue

            accuracies = []
            macro_f1_scores = []

            for checkpoint_info in model_checkpoint_runs:
                checkpoint_path = checkpoint_info["checkpoint_path"]

                # Create a deep copy to avoid overwriting weights
                model_copy = deepcopy(self.model_dict[model_name])

                accuracy, macro_f1 = self.evaluate_checkpoint(
                    model_copy, checkpoint_path
                )

                if accuracy is not None:
                    accuracies.append(accuracy)
                    macro_f1_scores.append(macro_f1)

            if len(accuracies) == 0:
                logger.warning(f"No valid results for model {model_name}")
                continue

            results[model_name] = {
                "mean_accuracy": float(np.mean(accuracies)),
                "std_accuracy": float(np.std(accuracies)),
                "mean_macro_f1": float(np.mean(macro_f1_scores)),
                "std_macro_f1": float(np.std(macro_f1_scores)),
            }

        return results
    


# Model registry
MODEL_REGISTRY = {
    "efficientnetv2m": efficientnetv2m,
    "efficientnetv2s": efficientnetv2s,
    "mobilenetv3_large": mobilenetv3_large,
    "mobilenetv3_small": mobilenetv3_small,
    "resnet50": resnet50,
    "shufflenet": shufflenet,
    "squeezenet": squeezenet,
    "vgg16": vgg16
}


def export_results_to_csv(
    results: Dict[str, Dict[str, float]],
    output_path: Path
) -> None:
    """
    Export benchmark results to a CSV file sorted by macro F1 score.

    Converts the results dictionary to a pandas DataFrame and saves it as CSV,
    sorted in descending order by mean_macro_f1 score (preferred metric for papers).

    Args:
        results (Dict[str, Dict[str, float]]): Benchmark results dictionary mapping
                                              model names to performance metrics
        output_path (Path): Output CSV file path

    Returns:
        None

    Raises:
        IOError: If the output file cannot be written

    Example:
        >>> results = {"resnet50": {"mean_accuracy": 0.95, ...}, ...}
        >>> export_results_to_csv(results, Path("results.csv"))
    """
    rows = []

    for model_name, metrics in results.items():
        row = {
            "model": model_name,
            "mean_accuracy": metrics["mean_accuracy"],
            "std_accuracy": metrics["std_accuracy"],
            "mean_macro_f1": metrics["mean_macro_f1"],
            "std_macro_f1": metrics["std_macro_f1"],
        }
        rows.append(row)

    results_dataframe = pd.DataFrame(rows)

    # Sort by macro F1 score in descending order (preferred for research papers)
    results_dataframe = results_dataframe.sort_values(
        by=BEST_METRIC_COLUMN, ascending=False
    )

    results_dataframe.to_csv(output_path, index=False)
    logger.info(f"Benchmark results saved to {output_path}")

    print("\n📊 Benchmark Results")
    print(results_dataframe.to_string(index=False))


if __name__ == "__main__":
    """Main execution script for model benchmarking."""
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create test dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=DEFAULT_BATCH_SIZE,
        shuffle=False
    )

    # Initialize benchmarker
    benchmarker = ModelBenchmarker(
        checkpoint_directory=Path(
            "/media/data/minhht/moe_plantdeasse/checkpoints/plantdoc"
        ),
        model_type_category="pretrain_weight",
        model_dict=MODEL_REGISTRY,
        test_dataloader=test_loader,
        device=device
    )

    # Run benchmarking
    benchmark_results = benchmarker()

    # Export results
    output_csv_path = Path(
        "/media/data/minhht/moe_plantdeasse/src/prediction/benchmark_results.csv"
    )
    export_results_to_csv(benchmark_results, output_csv_path)
    