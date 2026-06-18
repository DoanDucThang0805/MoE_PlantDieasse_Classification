"""
MoE Model Evaluation Module

Evaluates trained Mixture of Experts (MoE) models on test datasets.
Automatically discovers checkpoints and computes accuracy and macro-F1 scores.

Usage:
    python getaccvsf1.py --model_name mobilenetv3small_moe --type_model moe_contextaware_temp1.0 \\
        --dataset_name plantdoc --export_to_csv --csv_filename results.csv

Example:
    from getaccvsf1 import GetAccandmF1ScoreMoE
    
    evaluator = GetAccandmF1ScoreMoE(
        model_name='mobilenetv3small_moe',
        type_model='moe_contextaware_temp1.0',
        dataset_name='plantdoc',
        export_to_csv=True,
        csv_filename='results.csv'
    )
    df = evaluator.export_to_df()
"""

import os
from pathlib import Path
import argparse
from typing import List, Dict, Tuple, Union, Literal, Any

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from models.moe.linear_model import MoEModel
from dataset.slif_tomato_dataset import build_datasets

import logging

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class GetAccandmF1ScoreMoE:
    """
    Evaluates MoE models by computing accuracy and macro-F1 scores on test dataset.
    
    Automatically discovers trained checkpoints from directory structure, loads models,
    and evaluates them. Supports aggregation by expert count and top-k selection.
    
    Attributes:
        model_name (str): Model architecture name (e.g., 'mobilenetv3small_moe')
        type_model (str): Model type/config (e.g., 'moe_contextaware_temp1.0')
        dataset_name (str): Dataset name for checkpoint discovery
        csv_store_dir (Path): Directory to save CSV results
        export_to_csv (bool): Whether to export results to CSV
        csv_filename (str): Output CSV filename
    """
    
    def __init__(
        self, 
        model_name: str, 
        type_model: str, 
        dataset_name: str,
        csv_filename: str,
        csv_store_dir: Path = Path("./"), 
        export_to_csv: bool=False
    ) -> None:
        """
        Initialize the evaluator.
        
        Args:
            model_name: Model architecture name
            type_model: Model type/configuration variant
            dataset_name: Dataset name for checkpoint paths
            csv_filename: Output CSV filename
            csv_store_dir: Directory for CSV export (default: current directory)
            export_to_csv: Enable CSV export (default: False)
        """
        self.model_name = model_name
        self.type_model = type_model
        self.dataset_name = dataset_name
        self.csv_store_dir = csv_store_dir
        self.export_to_csv = export_to_csv
        self.csv_filename = csv_filename

        logger.info(
            f"Initialized GetAccandmF1ScoreMoE: "
            f"model_name={model_name}, type_model={type_model}, dataset={dataset_name}, "
            f"export_csv={export_to_csv}"
        )


    def checkpoint_paths(self, model_name: str, type_model: str, dataset_name: str) -> List[Dict[str, Any]]:
        """
        Discover all checkpoint paths from directory structure.
        
        Searches: checkpoints/{dataset}/{type_model}/{model_name}/{n}_experts/top_{k}/seed_{s}/run_{t}/
        
        Returns:
            List of dicts with keys: 'num_expert', 'top_k', 'seed', 'checkpoint_path'
            
        Raises:
            FileNotFoundError: If checkpoint root directory doesn't exist
        """
        root_path = Path(f'../checkpoints/{dataset_name}/{type_model}/{model_name}')
        logger.info(f"Discovering checkpoints from: {root_path}")
        
        if not root_path.exists():
            logger.error(f"Checkpoint root directory not found: {root_path}")
            raise FileNotFoundError(f"Checkpoint root directory not found: {root_path}")
        
        list_paths = []
        num_experts = sorted(os.listdir(root_path))
        logger.debug(f"Found {len(num_experts)} expert configurations: {num_experts}")
        
        for num_expert in num_experts:
            topk_s = sorted(os.listdir(root_path/num_expert))
            logger.debug(f"  {num_expert}: {len(topk_s)} top_k variants")
            
            for top_k in topk_s:
                for seed in ['seed_42', 'seed_43', 'seed_44', 'seed_45', 'seed_46']:
                    try:
                        best_checkpoint_path = next((root_path/num_expert/top_k/seed).rglob("best_checkpoint.pth"))
                        list_paths.append(
                            {
                                "num_expert": num_expert.split('_')[0],
                                "top_k": top_k.split('_')[1],
                                "seed": seed.split('_')[1],
                                "checkpoint_path": best_checkpoint_path
                            }
                        )
                        logger.debug(
                            f"    Found checkpoint: experts={num_expert}, top_k={top_k}, {seed} -> {best_checkpoint_path.name}"
                        )
                    except StopIteration:
                        logger.debug(f"    No checkpoint found for: {num_expert}/{top_k}/{seed}")
                        continue
        
        logger.info(f"Discovered {len(list_paths)} checkpoints total")
        return list_paths
    

    def extract_checkpoint(self, checkpoint_path: Union[Path, str]) -> Dict[str, Any]:
        """
        Extract model hyperparameters from checkpoint file.
        
        Args:
            checkpoint_path: Path to best_checkpoint.pth file
            
        Returns:
            Dict with keys: 'context_dim', 'num_classes', 'num_experts', 'top_k', 'router_mode', 'temperature'
            
        Raises:
            Exception: If checkpoint loading or key extraction fails
        """
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            context_dim = checkpoint["context_dim"]
            num_classes = checkpoint["num_classes"]
            num_experts = checkpoint["num_experts"]
            top_k = checkpoint["top_k"]
            router_mode = checkpoint["router_mode"]
            temperature = checkpoint["temperature"]
            
            logger.debug(
                f"Extracted checkpoint config: experts={num_experts}, top_k={top_k}, "
                f"classes={num_classes}, context_dim={context_dim}, temp={temperature}"
            )
            
            return {
                "context_dim" : context_dim,
                "num_classes": num_classes,
                "num_experts": num_experts,
                "top_k": top_k,
                "router_mode": router_mode,
                "temperature": temperature,
            }
        except Exception as e:
            logger.error(f"Failed to extract checkpoint from {checkpoint_path}: {e}")
            raise
        

    def load_checkpoint(self, model: MoEModel, checkpoint_path: Union[Path, str]) -> MoEModel:
        """
        Load model weights from checkpoint into model instance.
        
        Args:
            model: MoEModel instance to load weights into
            checkpoint_path: Path to best_checkpoint.pth file
            
        Returns:
            Model with loaded weights
            
        Raises:
            Exception: If checkpoint loading fails
        """
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state_dict = checkpoint["model_state_dict"]
            model.load_state_dict(state_dict=state_dict)
            logger.debug(f"Successfully loaded checkpoint from {checkpoint_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            raise
    

    def create_model(
        self,
        context_dim: int,
        num_classes: int,
        num_experts: int,
        top_k: int,
        router_mode: Literal['noisy', 'context_aware'],
        temperature: float
    ) -> MoEModel:
        """
        Create MoEModel instance with specified hyperparameters.
        
        Args:
            context_dim: Context feature dimension
            num_classes: Number of output classes
            num_experts: Number of experts in mixture
            top_k: Number of top experts to select
            router_mode: Routing strategy ('noisy' or 'context_aware')
            temperature: Temperature for gating softmax
            
        Returns:
            Initialized MoEModel instance
        """
        logger.debug(
            f"Creating MoE model: experts={num_experts}, top_k={top_k}, "
            f"classes={num_classes}, router_mode={router_mode}, temp={temperature}"
        )
        model = MoEModel(
            context_dim=context_dim,
            num_classes=num_classes,
            num_experts=num_experts,
            top_k=top_k,
            router_mode=router_mode,
            temperature=temperature
        )
        return model


    def create_dataset(self, use_context: bool=True) -> DataLoader:
        """
        Load test dataset and create DataLoader.
        
        Args:
            use_context: Include context features in dataset (default: True)
            
        Returns:
            DataLoader with batch_size=32 and shuffle=False
        """
        logger.info(f"Loading test dataset (use_context={use_context})...")
        _, _, test_dataset = build_datasets(use_context)
        logger.info(f"Test dataset loaded: {len(test_dataset)} samples")
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        return test_loader


    @torch.inference_mode(True)
    def run_inference(self, model: MoEModel, data_loader: DataLoader) -> Tuple[float, float]:
        """
        Run inference on test dataset and compute metrics.
        
        Args:
            model: MoEModel instance in eval mode
            data_loader: DataLoader with test samples
            
        Returns:
            Tuple of (accuracy, macro_f1_score)
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        logger.debug(f"Running inference on {device}...")
        
        all_labels = []
        all_predicts = []
        batch_count = 0
        
        for batch in data_loader:
            batch_count += 1
            images, labels, contexts = batch
            images, labels, contexts = images.to(device), labels.to(device), contexts.to(device)
            
            logits, _, _ = model(images, contexts)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_predicts.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        all_labels = np.array(all_labels)
        all_predicts = np.array(all_predicts)
        accuracy = accuracy_score(all_labels, all_predicts)
        macro_f1 = f1_score(all_labels, all_predicts, average="macro")
        
        logger.debug(f"Inference completed: {batch_count} batches processed")
        logger.debug(f"Results: Accuracy={accuracy:.4f}, Macro-F1={macro_f1:.4f}")
        
        return accuracy, macro_f1


    def acc_and_mf1_score(self) -> List[Dict[str, Any]]:
        """
        Evaluate all discovered checkpoints and compute metrics.
        
        Main evaluation pipeline that:
        1. Discovers all checkpoints
        2. Loads and evaluates each model
        3. Computes accuracy and macro-F1 for each
        4. Collects results with model configuration info
        
        Returns:
            List of dicts with keys: 'num_experts', 'top_k', 'seed', 'accuracy', 'macro_f1'
        """
        logger.info("=" * 80)
        logger.info("Starting accuracy and Macro-F1 score calculation...")
        logger.info("=" * 80)
        
        results = []
        list_checkpoint_paths = self.checkpoint_paths(
            model_name = self.model_name,
            type_model = self.type_model,
            dataset_name= self.dataset_name
        )
        
        if not list_checkpoint_paths:
            logger.error("No checkpoints found!")
            return results
        
        test_loader = self.create_dataset(use_context=True)
        
        logger.info(f"Processing {len(list_checkpoint_paths)} checkpoints...")
        logger.info("-" * 80)

        for idx, checkpoint_info in enumerate(list_checkpoint_paths, 1):
            checkpoint_path = checkpoint_info['checkpoint_path']
            num_expert = checkpoint_info["num_expert"]
            top_k = checkpoint_info["top_k"]
            seed = checkpoint_info["seed"]
            
            logger.info(f"[{idx}/{len(list_checkpoint_paths)}] experts={num_expert}, top_k={top_k}, seed={seed}")
            
            try:
                checkpoint_config = self.extract_checkpoint(checkpoint_path)
                model = self.create_model(
                    context_dim=checkpoint_config["context_dim"],
                    num_classes=checkpoint_config["num_classes"],
                    num_experts=checkpoint_config["num_experts"],
                    top_k=checkpoint_config["top_k"],
                    router_mode=checkpoint_config["router_mode"],
                    temperature=checkpoint_config["temperature"]
                )
                model = self.load_checkpoint(model=model, checkpoint_path=checkpoint_path)
                accuracy, macro_f1 = self.run_inference(model=model, data_loader=test_loader)
                
                results.append(
                    {
                        "num_experts": num_expert,
                        "top_k": top_k,
                        "seed": seed,
                        "accuracy": accuracy,
                        "macro_f1": macro_f1
                    }
                )
                
                logger.info(f"  ✓ Accuracy: {accuracy:.4f}, Macro-F1: {macro_f1:.4f}")
                
            except Exception as e:
                logger.error(f"  ✗ Failed to process checkpoint: {e}")
                continue
        
        logger.info("-" * 80)
        logger.info(f"Completed processing. Results: {len(results)}/{len(list_checkpoint_paths)} successful")
        
        return results


    def export_to_df(self) -> pd.DataFrame:
        """
        Evaluate all models and aggregate results by expert count and top-k.
        
        Process:
        1. Run acc_and_mf1_score() to evaluate all models
        2. Convert results to DataFrame
        3. Aggregate by grouping on (num_experts, top_k) - averaging across seeds
        4. Optionally export to CSV
        
        Returns:
            DataFrame with columns: 'num_experts', 'top_k', 'accuracy', 'macro_f1' (aggregated)
        """
        logger.info("Exporting results to DataFrame...")
        acc_and_macrof1 = self.acc_and_mf1_score()
        
        if not acc_and_macrof1:
            logger.warning("No results to export!")
            return pd.DataFrame()
        
        df = pd.DataFrame(acc_and_macrof1)
        logger.debug(f"Raw results DataFrame: {len(df)} rows")
        logger.debug(f"Columns: {list(df.columns)}")
        
        # Calculate both mean and std for accuracy and macro_f1
        agg_dict = {
            "accuracy": ["mean", "std"],
            "macro_f1": ["mean", "std"]
        }
        df = df.groupby(["num_experts", "top_k"])[["accuracy", "macro_f1"]].agg(agg_dict).reset_index()
        df.columns = ["num_experts", "top_k", "accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std"]
        
        logger.info(f"Aggregated results by (num_experts, top_k): {len(df)} rows")
        logger.debug(f"\nAggregated Results:\n{df.to_string()}")
        
        if self.export_to_csv:
            csv_path = self.csv_store_dir / self.csv_filename
            df.to_csv(csv_path, index=False)
            logger.info(f"Results exported to CSV: {csv_path}")
        else:
            logger.info("CSV export disabled")
        
        return df


def main():
    """
    Command-line interface for model evaluation.
    
    Usage:
        python getaccvsf1.py --model_name mobilenetv3small_moe \\
            --type_model moe_contextaware_temp1.0 --dataset_name plantdoc \\
            --export_to_csv --csv_filename results.csv
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, default='mobilenetv3small_moe')
    parser.add_argument("--type_model", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True, default="plantdoc")
    parser.add_argument("--csv_store_dir", type=str, default="./results")
    parser.add_argument("--export_to_csv", action="store_true")
    parser.add_argument("--csv_filename", type=str)

    args = parser.parse_args()

    evaluator = GetAccandmF1ScoreMoE(
        model_name=args.model_name,
        type_model=args.type_model,
        dataset_name=args.dataset_name,
        csv_filename=args.csv_filename,
        csv_store_dir=Path(args.csv_store_dir),
        export_to_csv=args.export_to_csv
    )

    df = evaluator.export_to_df()

    print(df)


if __name__ == "__main__":
    main()