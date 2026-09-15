import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader


SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset.plantdoc_dataset import build_datasets
from models.dense_multibranch.mobilenetv3_small_dense_multibranch import (
    MobileNetV3SmallDenseMultiBranch,
)


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class GetAccvsMacroF1DenseMultiBranch:
    def __init__(
        self,
        checkpoint_dirs: Union[str, Path],
        csv_store_dir: Union[str, Path],
        csv_filename: str,
        export_csv: bool = False,
    ) -> None:
        self.checkpoint_dirs = self._resolve_path(checkpoint_dirs)
        self.csv_store_dir = self._resolve_path(csv_store_dir)
        self.csv_filename = csv_filename
        self.export_csv = export_csv

    def _resolve_path(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        if path.is_absolute():
            return path
        if path.exists():
            return path
        return PROJECT_DIR / path

    def get_checkpoint_paths(self) -> List[Dict[str, Any]]:
        """
        Find one best checkpoint for each num_experts/seed directory.

        Expected structure:
            checkpoint_dirs/4_experts/seed_42/run_YYYYmmdd-HHMMSS/best_checkpoint.pth

        If a seed has multiple runs, the newest checkpoint is selected.
        """
        if not self.checkpoint_dirs.exists():
            raise FileNotFoundError(
                f"Checkpoint directory not found: {self.checkpoint_dirs}"
            )

        checkpoint_infos = []
        expert_dirs = sorted(
            path for path in self.checkpoint_dirs.iterdir()
            if path.is_dir() and path.name.endswith("_experts")
        )

        for expert_dir in expert_dirs:
            seed_dirs = sorted(
                path for path in expert_dir.iterdir()
                if path.is_dir() and path.name.startswith("seed_")
            )

            for seed_dir in seed_dirs:
                checkpoints = sorted(
                    seed_dir.rglob("best_checkpoint.pth"),
                    key=lambda path: (path.stat().st_mtime, str(path)),
                    reverse=True,
                )

                if not checkpoints:
                    logger.warning("No best_checkpoint.pth found under %s", seed_dir)
                    continue

                checkpoint_infos.append(
                    {
                        "num_experts": expert_dir.name.replace("_experts", ""),
                        "seed": seed_dir.name.replace("seed_", ""),
                        "checkpoint_path": checkpoints[0],
                    }
                )

        logger.info("Found %d checkpoint(s)", len(checkpoint_infos))
        return checkpoint_infos

    def extract_checkpoint_config(self, checkpoint_path: Union[str, Path]) -> Dict[str, int]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        num_experts = checkpoint.get("num_experts")
        if num_experts is None:
            expert_ids = {
                key.split(".")[1]
                for key in state_dict
                if key.startswith("experts.") and len(key.split(".")) > 2
            }
            num_experts = len(expert_ids)

        num_classes = checkpoint.get("num_classes")
        if num_classes is None:
            classifier_weight = state_dict["experts.0.classifier.4.weight"]
            num_classes = classifier_weight.shape[0]

        return {
            "num_classes": int(num_classes),
            "num_experts": int(num_experts),
        }

    def create_model(self, num_classes: int, num_experts: int) -> nn.Module:
        return MobileNetV3SmallDenseMultiBranch(
            num_classes=num_classes,
            num_experts=num_experts,
        )

    def load_checkpoint(
        self,
        model: nn.Module,
        checkpoint_path: Union[str, Path],
    ) -> nn.Module:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        return model

    def create_dataloader(self, batch_size: int = 32) -> DataLoader:
        _, _, test_dataset = build_datasets(use_context=False)
        return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    @torch.inference_mode()
    def evaluate_model(
        self,
        model: nn.Module,
        test_loader: DataLoader,
    ) -> Tuple[float, float]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        all_labels = []
        all_predictions = []

        for images, labels, _ in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_predictions)
        macro_f1 = f1_score(all_labels, all_predictions, average="macro")
        return accuracy, macro_f1

    def acc_and_mf1_score(self) -> List[Dict[str, Any]]:
        results = []
        checkpoint_infos = self.get_checkpoint_paths()

        if not checkpoint_infos:
            logger.warning("No checkpoints found to evaluate")
            return results

        test_loader = self.create_dataloader()

        for idx, checkpoint_info in enumerate(checkpoint_infos, start=1):
            num_experts = checkpoint_info["num_experts"]
            seed = checkpoint_info["seed"]
            checkpoint_path = checkpoint_info["checkpoint_path"]

            logger.info(
                "[%d/%d] Evaluating num_experts=%s seed=%s: %s",
                idx,
                len(checkpoint_infos),
                num_experts,
                seed,
                checkpoint_path,
            )

            try:
                config = self.extract_checkpoint_config(checkpoint_path)
                model = self.create_model(
                    num_classes=config["num_classes"],
                    num_experts=config["num_experts"],
                )
                model = self.load_checkpoint(model, checkpoint_path)
                accuracy, macro_f1 = self.evaluate_model(model, test_loader)
            except Exception as exc:
                logger.error(
                    "Skip num_experts=%s seed=%s because evaluation failed: %s",
                    num_experts,
                    seed,
                    str(exc).splitlines()[0],
                )
                continue

            results.append(
                {
                    "num_experts": num_experts,
                    "seed": seed,
                    "checkpoint_path": str(checkpoint_path),
                    "accuracy": accuracy,
                    "macro_f1": macro_f1,
                }
            )
            logger.info(
                "num_experts=%s seed=%s accuracy=%.4f macro_f1=%.4f",
                num_experts,
                seed,
                accuracy,
                macro_f1,
            )

        return results

    def export_results_to_csv(self, results: pd.DataFrame) -> None:
        self.csv_store_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.csv_store_dir / self.csv_filename
        results.to_csv(csv_path, index=False)
        logger.info("Results exported to CSV: %s", csv_path)

    def export_to_df(self) -> pd.DataFrame:
        per_seed_results = self.acc_and_mf1_score()

        if not per_seed_results:
            return pd.DataFrame()

        df = pd.DataFrame(per_seed_results)
        aggregated_df = (
            df.groupby(["num_experts"])[["accuracy", "macro_f1"]]
            .agg(["mean", "std"])
            .reset_index()
        )
        aggregated_df.columns = [
            "num_experts",
            "accuracy_mean",
            "accuracy_std",
            "macro_f1_mean",
            "macro_f1_std",
        ]
        aggregated_df.insert(1, "num_seeds", df.groupby("num_experts").size().values)

        if self.export_csv:
            self.export_results_to_csv(aggregated_df)

        return aggregated_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_dirs",
        type=str,
        default="checkpoints/plantdoc/dense_multibranch/mobilenetv3small_dense_multibranch",
    )
    parser.add_argument("--csv_store_dir", type=str, default="./results")
    parser.add_argument(
        "--csv_filename",
        type=str,
        default="mobilenetv3small_dense_multibranch_results.csv",
    )
    parser.add_argument("--export_csv", action="store_true")
    args = parser.parse_args()

    evaluator = GetAccvsMacroF1DenseMultiBranch(
        checkpoint_dirs=args.checkpoint_dirs,
        csv_store_dir=args.csv_store_dir,
        csv_filename=args.csv_filename,
        export_csv=args.export_csv,
    )
    df = evaluator.export_to_df()
    print(df)


if __name__ == "__main__":
    main()
