from pathlib import Path
import argparse
import io
import sys
from contextlib import redirect_stdout
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
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
from models.moe.gating import ContextAwareGating, ContextAwareLinearGating
from models.moe.model import MoEModel


MODEL_TYPES = [
    "moe",
    "dense_multibranch",
    "mobilenetv3_small",
    "widened_mlp_head",
    "shufflenet",
]


class PairedCheckpointTest:
    def __init__(
        self,
        model_a_name: str,
        model_a_dir: Path,
        model_a_type: str,
        model_b_name: str,
        model_b_dir: Path,
        model_b_type: str,
        output_csv: Path,
        split: str,
        batch_size: int,
        seeds: List[str],
    ):
        self.model_a_name = model_a_name
        self.model_a_dir = self.resolve_path(model_a_dir)
        self.model_a_type = model_a_type
        self.model_b_name = model_b_name
        self.model_b_dir = self.resolve_path(model_b_dir)
        self.model_b_type = model_b_type
        self.output_csv = self.resolve_path(output_csv)
        self.split = split
        self.batch_size = batch_size
        self.seeds = [str(seed) for seed in seeds]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def resolve_path(path: Path) -> Path:
        path = Path(path)
        if path.is_absolute():
            return path
        return PROJECT_DIR / path

    def find_seed_checkpoints(self, root_dir: Path) -> Dict[str, Path]:
        if not root_dir.exists():
            raise FileNotFoundError(f"Checkpoint root not found: {root_dir}")

        checkpoints = {}
        for seed in self.seeds:
            seed_dir_name = f"seed_{seed}"
            seed_dirs = sorted(
                path for path in root_dir.rglob(seed_dir_name)
                if path.is_dir() and path.name == seed_dir_name
            )
            if not seed_dirs:
                continue

            candidates = []
            for seed_dir in seed_dirs:
                candidates.extend(seed_dir.rglob("best_checkpoint.pth"))

            if not candidates:
                continue

            # FIX 3: Cảnh báo nếu tìm thấy nhiều hơn 1 checkpoint cho cùng seed
            # để tránh chọn nhầm do mtime thay đổi khi copy/re-save file
            if len(candidates) > 1:
                candidates_sorted = sorted(
                    candidates,
                    key=lambda path: (path.stat().st_mtime, str(path)),
                    reverse=True,
                )
                print(
                    f"WARNING: seed_{seed} has {len(candidates)} checkpoints. "
                    f"Picking latest by mtime: {candidates_sorted[0]}"
                )
                checkpoints[seed] = candidates_sorted[0]
            else:
                checkpoints[seed] = candidates[0]

        return checkpoints

    # FIX 4: Tách build_datasets ra khỏi create_dataloader
    # để tránh gọi build_datasets 2 lần (một lần cho model A, một lần cho model B)
    # khi dataset là giống nhau và chỉ khác use_context
    def build_dataloader(self, dataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def get_split_dataset(self, use_context: bool):
        train_dataset, val_dataset, test_dataset = build_datasets(use_context=use_context)
        datasets = {
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset,
        }
        return datasets[self.split]

    def load_checkpoint_file(self, checkpoint_path: Path) -> Dict[str, Any]:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        return {
            "checkpoint": checkpoint,
            "state_dict": state_dict,
        }

    @staticmethod
    def infer_num_classes(state_dict: Dict[str, torch.Tensor], checkpoint: Dict[str, Any]) -> int:
        if isinstance(checkpoint, dict) and "num_classes" in checkpoint:
            return int(checkpoint["num_classes"])

        classifier_keys = [
            key for key, value in state_dict.items()
            if key.endswith("weight") and value.ndim == 2
        ]
        if not classifier_keys:
            raise ValueError("Cannot infer num_classes from checkpoint")

        return int(state_dict[classifier_keys[-1]].shape[0])

    @staticmethod
    def infer_num_experts(state_dict: Dict[str, torch.Tensor], checkpoint: Dict[str, Any]) -> int:
        if isinstance(checkpoint, dict) and "num_experts" in checkpoint:
            return int(checkpoint["num_experts"])

        expert_ids = {
            key.split(".")[1]
            for key in state_dict
            if key.startswith("experts.") and len(key.split(".")) > 2
        }
        if not expert_ids:
            raise ValueError("Cannot infer num_experts from checkpoint")

        return len(expert_ids)

    @staticmethod
    def uses_linear_context_gating(state_dict: Dict[str, torch.Tensor]) -> bool:
        return (
            "moe_layer.gating.gate_projector.weight" in state_dict
            and "moe_layer.gating.gate_projector.0.weight" not in state_dict
        )

    @staticmethod
    def uses_mlp_context_gating(state_dict: Dict[str, torch.Tensor]) -> bool:
        return "moe_layer.gating.gate_projector.0.weight" in state_dict

    def create_moe_model(
        self,
        checkpoint: Dict[str, Any],
        state_dict: Dict[str, torch.Tensor],
    ) -> MoEModel:
        model = MoEModel(
            context_dim=checkpoint["context_dim"],
            num_classes=checkpoint["num_classes"],
            num_experts=checkpoint["num_experts"],
            top_k=checkpoint["top_k"],
            router_mode=checkpoint["router_mode"],
            temperature=checkpoint["temperature"],
        )
        if self.uses_linear_context_gating(state_dict):
            model.moe_layer.gating = ContextAwareLinearGating(
                model_dim=model.feature_extractor.output_dim,
                context_dim=checkpoint["context_dim"],
                num_experts=checkpoint["num_experts"],
                top_k=checkpoint["top_k"],
                temperature=checkpoint["temperature"],
            )
        elif self.uses_mlp_context_gating(state_dict):
            model.moe_layer.gating = ContextAwareGating(
                model_dim=model.feature_extractor.output_dim,
                context_dim=checkpoint["context_dim"],
                num_experts=checkpoint["num_experts"],
                top_k=checkpoint["top_k"],
                temperature=checkpoint["temperature"],
            )
        else:
            # FIX 2: Trước đây code im lặng dùng gating mặc định của MoEModel
            # dẫn đến load_state_dict() có thể bị key mismatch hoặc load sai weights
            raise ValueError(
                "Cannot detect gating type from state_dict keys. "
                "Expected 'moe_layer.gating.gate_projector.weight' (Linear) "
                "or 'moe_layer.gating.gate_projector.0.weight' (MLP). "
                f"Keys found: {[k for k in state_dict if 'gating' in k]}"
            )
        return model

    def create_pretrained_model(self, model_type: str, num_classes: int) -> nn.Module:
        if model_type == "mobilenetv3_small":
            from torchvision.models import mobilenet_v3_small
            return mobilenet_v3_small(weights=None, num_classes=num_classes)

        if model_type == "mobilenetv3_large":
            from torchvision.models import mobilenet_v3_large
            return mobilenet_v3_large(weights=None, num_classes=num_classes)

        if model_type == "widened_mlp_head":
            from torchvision.models import mobilenet_v3_small
            model = mobilenet_v3_small(weights=None)
            in_features = model.classifier[0].in_features
            model.classifier = nn.Sequential(
                nn.Linear(in_features, 2048),
                nn.Hardswish(),
                nn.Dropout(0.2),
                nn.Linear(2048, 512),
                nn.Hardswish(),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes),
            )
            return model

        if model_type == "shufflenet":
            from torchvision.models import shufflenet_v2_x2_0
            model = shufflenet_v2_x2_0(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model

        if model_type == "resnet50":
            from torchvision.models import resnet50
            model = resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model

        if model_type == "efficientnet_b4":
            import timm
            return timm.create_model(
                model_name="efficientnet_b4",
                pretrained=False,
                num_classes=num_classes,
            )

        raise ValueError(f"Unsupported pretrained model_type: {model_type}")

    def create_model(self, model_type: str, checkpoint_path: Path) -> Tuple[nn.Module, bool]:
        loaded = self.load_checkpoint_file(checkpoint_path)
        checkpoint = loaded["checkpoint"]
        state_dict = loaded["state_dict"]

        if model_type == "moe":
            model = self.create_moe_model(checkpoint, state_dict)
            model.load_state_dict(state_dict)
            return model, True

        if model_type == "dense_multibranch":
            num_classes = self.infer_num_classes(state_dict, checkpoint)
            num_experts = self.infer_num_experts(state_dict, checkpoint)
            model = MobileNetV3SmallDenseMultiBranch(
                num_classes=num_classes,
                num_experts=num_experts,
            )
            model.load_state_dict(state_dict)
            return model, False

        num_classes = self.infer_num_classes(state_dict, checkpoint)
        with redirect_stdout(io.StringIO()):
            model = self.create_pretrained_model(model_type, num_classes)
        model.load_state_dict(state_dict)
        return model, False

    @torch.inference_mode()
    def evaluate_checkpoint(
        self,
        checkpoint_path: Path,
        model_type: str,
        dataloader: DataLoader,
    ) -> Tuple[float, float]:
        model, uses_context = self.create_model(model_type, checkpoint_path)
        model.to(self.device)
        model.eval()

        labels_all = []
        preds_all = []

        for images, labels, context in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            if uses_context:
                context = context.to(self.device)
                logits, _, _ = model(images, context)
            else:
                logits = model(images)

            preds = torch.argmax(logits, dim=1)
            labels_all.extend(labels.cpu().numpy())
            preds_all.extend(preds.cpu().numpy())

        accuracy = accuracy_score(labels_all, preds_all)
        macro_f1 = f1_score(labels_all, preds_all, average="macro")
        return float(accuracy), float(macro_f1)

    def evaluate_model_group(
        self,
        model_name: str,
        model_type: str,
        checkpoint_dir: Path,
        dataloader: DataLoader,  # FIX 4: nhận dataloader từ ngoài thay vì tự tạo
    ) -> pd.DataFrame:
        seed_to_checkpoint = self.find_seed_checkpoints(checkpoint_dir)
        missing = sorted(set(self.seeds) - set(seed_to_checkpoint.keys()))
        if missing:
            print(f"WARNING: {model_name} missing seeds: {','.join(missing)}")

        rows = []
        for seed in self.seeds:
            checkpoint_path = seed_to_checkpoint.get(seed)
            if checkpoint_path is None:
                continue

            print(f"Evaluating {model_name} seed={seed}: {checkpoint_path}")
            accuracy, macro_f1 = self.evaluate_checkpoint(
                checkpoint_path=checkpoint_path,
                model_type=model_type,
                dataloader=dataloader,
            )
            rows.append({
                "model": model_name,
                "model_type": model_type,
                "seed": seed,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "checkpoint_path": str(checkpoint_path),
            })

        return pd.DataFrame(rows)

    @staticmethod
    def _verdict(t_pvalue: float, w_pvalue: float, alpha: float = 0.05) -> str:
        # FIX 5: Thêm cột verdict dạng readable cho reviewer
        # Chỉ kết luận "significant" khi CẢ HAI test đều p < alpha
        # đúng theo yêu cầu Session 6
        if t_pvalue < alpha and w_pvalue < alpha:
            return "significant"
        elif t_pvalue < alpha or w_pvalue < alpha:
            return "inconclusive"
        return "not_significant"

    @staticmethod
    def run_tests(
        comparison: str,
        metric: str,
        paired_df: pd.DataFrame,
        model_a_name: str,
        model_b_name: str,
    ) -> Dict[str, Any]:
        a_values = paired_df[f"{metric}_a"].to_numpy(dtype=float)
        b_values = paired_df[f"{metric}_b"].to_numpy(dtype=float)
        deltas = a_values - b_values

        t_result = stats.ttest_rel(a_values, b_values)

        try:
            wilcoxon_result = stats.wilcoxon(a_values, b_values, zero_method="wilcox")
            wilcoxon_statistic = float(wilcoxon_result.statistic)
            wilcoxon_pvalue = float(wilcoxon_result.pvalue)
        except ValueError:
            wilcoxon_statistic = np.nan
            wilcoxon_pvalue = np.nan

        t_pvalue = float(t_result.pvalue)
        w_pvalue = wilcoxon_pvalue

        return {
            "comparison": comparison,
            "metric": metric,
            "n_seeds": len(paired_df),
            "seed_ids": ";".join(paired_df["seed"].astype(str).tolist()),
            "model_a": model_a_name,
            "model_b": model_b_name,
            "model_a_mean": float(a_values.mean()),
            "model_a_std": float(a_values.std(ddof=1)) if len(a_values) > 1 else np.nan,
            "model_b_mean": float(b_values.mean()),
            "model_b_std": float(b_values.std(ddof=1)) if len(b_values) > 1 else np.nan,
            "mean_delta": float(deltas.mean()),
            "std_delta": float(deltas.std(ddof=1)) if len(deltas) > 1 else np.nan,
            "paired_t_statistic": float(t_result.statistic),
            "paired_t_pvalue": t_pvalue,
            "wilcoxon_statistic": wilcoxon_statistic,
            "wilcoxon_pvalue": w_pvalue,
            # FIX 1: Tách thành 3 cột riêng biệt thay vì chỉ check t-test
            # significant_0_05 cũ chỉ dùng t_pvalue → vi phạm yêu cầu Session 6
            "significant_t_0_05":        bool(t_pvalue < 0.05),
            "significant_wilcoxon_0_05": bool(w_pvalue < 0.05),
            "significant_both_0_05":     bool(t_pvalue < 0.05 and w_pvalue < 0.05),
            # FIX 5: Cột verdict dạng text để dễ đọc trong báo cáo
            "verdict": PairedCheckpointTest._verdict(t_pvalue, w_pvalue),
        }

    def run(self) -> pd.DataFrame:
        # FIX 4: Tạo dataloader một lần cho mỗi model type
        # thay vì gọi create_dataloader bên trong evaluate_model_group
        print("Building dataset for model A...")
        dataset_a = self.get_split_dataset(use_context=(self.model_a_type == "moe"))
        dataloader_a = self.build_dataloader(dataset_a)

        print("Building dataset for model B...")
        dataset_b = self.get_split_dataset(use_context=(self.model_b_type == "moe"))
        dataloader_b = self.build_dataloader(dataset_b)

        model_a_df = self.evaluate_model_group(
            model_name=self.model_a_name,
            model_type=self.model_a_type,
            checkpoint_dir=self.model_a_dir,
            dataloader=dataloader_a,
        )
        model_b_df = self.evaluate_model_group(
            model_name=self.model_b_name,
            model_type=self.model_b_type,
            checkpoint_dir=self.model_b_dir,
            dataloader=dataloader_b,
        )

        paired_df = model_a_df.merge(
            model_b_df,
            on="seed",
            how="inner",
            suffixes=("_a", "_b"),
        )

        if paired_df.empty:
            raise ValueError("No paired seeds found between the two checkpoint dirs")

        if len(paired_df) < 2:
            raise ValueError("At least 2 paired seeds are required for paired tests")

        comparison = f"{self.model_a_name} vs {self.model_b_name}"
        results = [
            self.run_tests(
                comparison=comparison,
                metric="accuracy",
                paired_df=paired_df,
                model_a_name=self.model_a_name,
                model_b_name=self.model_b_name,
            ),
            self.run_tests(
                comparison=comparison,
                metric="macro_f1",
                paired_df=paired_df,
                model_a_name=self.model_a_name,
                model_b_name=self.model_b_name,
            ),
        ]

        output_df = pd.DataFrame(results)
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(self.output_csv, index=False)
        print(output_df.to_string(index=False))
        print(f"\nSaved paired test table: {self.output_csv}")
        return output_df


def parse_seeds(seed_text: str) -> List[str]:
    return [seed.strip() for seed in seed_text.split(",") if seed.strip()]


def get_args():
    parser = argparse.ArgumentParser(
        description="Run paired statistical tests between two checkpoint groups.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_a_name", type=str, required=True)
    parser.add_argument("--model_a_dir", type=Path, required=True)
    parser.add_argument("--model_a_type", type=str, choices=MODEL_TYPES, required=True)
    parser.add_argument("--model_b_name", type=str, required=True)
    parser.add_argument("--model_b_dir", type=Path, required=True)
    parser.add_argument("--model_b_type", type=str, choices=MODEL_TYPES, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,43,44,45,46",
        help="Comma-separated paired seed IDs",
    )
    return parser.parse_args()


def main():
    args = get_args()
    runner = PairedCheckpointTest(
        model_a_name=args.model_a_name,
        model_a_dir=args.model_a_dir,
        model_a_type=args.model_a_type,
        model_b_name=args.model_b_name,
        model_b_dir=args.model_b_dir,
        model_b_type=args.model_b_type,
        output_csv=args.output_csv,
        split=args.split,
        batch_size=args.batch_size,
        seeds=parse_seeds(args.seeds),
    )
    runner.run()


if __name__ == "__main__":
    main()