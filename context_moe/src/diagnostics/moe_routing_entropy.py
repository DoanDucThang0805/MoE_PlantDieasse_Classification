from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset.plantdoc_dataset import build_datasets
from models.moe.gating import ContextAwareLinearGating
from models.moe.model import MoEModel


class MoERoutingEntropy:
    def __init__(
        self,
        checkpoint_path: Path,
        output_dir: Path,
        split: str,
        csv_name: str,
        plot_name: str,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.output_dir = Path(output_dir)
        self.split = split
        self.csv_name = csv_name
        self.plot_name = plot_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        return {
            "model_state_dict": checkpoint["model_state_dict"],
            "num_classes": checkpoint["num_classes"],
            "num_experts": checkpoint["num_experts"],
            "top_k": checkpoint["top_k"],
            "temperature": checkpoint["temperature"],
            "context_dim": checkpoint["context_dim"],
            "router_mode": checkpoint["router_mode"],
        }

    def create_model(self, checkpoint_info: dict) -> MoEModel:
        model = MoEModel(
            num_classes=checkpoint_info["num_classes"],
            num_experts=checkpoint_info["num_experts"],
            top_k=checkpoint_info["top_k"],
            temperature=checkpoint_info["temperature"],
            context_dim=checkpoint_info["context_dim"],
            router_mode=checkpoint_info["router_mode"],
        )
        if self.uses_linear_context_gating(checkpoint_info["model_state_dict"]):
            model.moe_layer.gating = ContextAwareLinearGating(
                model_dim=model.feature_extractor.output_dim,
                context_dim=checkpoint_info["context_dim"],
                num_experts=checkpoint_info["num_experts"],
                top_k=checkpoint_info["top_k"],
                temperature=checkpoint_info["temperature"],
            )
        model.load_state_dict(checkpoint_info["model_state_dict"])
        model.to(self.device)
        model.eval()
        return model

    @staticmethod
    def uses_linear_context_gating(state_dict: dict) -> bool:
        return (
            "moe_layer.gating.gate_projector.weight" in state_dict
            and "moe_layer.gating.gate_projector.0.weight" not in state_dict
        )

    def create_dataloader(self, batch_size: int):
        train_dataset, val_dataset, test_dataset = build_datasets(use_context=True)
        datasets = {
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset,
        }
        dataset = datasets[self.split]
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return dataloader, dataset

    def collect_class_wise_routing(
        self,
        model: MoEModel,
        dataloader: DataLoader,
    ):
        class_expert_counts = torch.zeros(
            model.num_classes,
            model.num_experts,
            dtype=torch.float32,
        )
        class_counts = torch.zeros(model.num_classes, dtype=torch.float32)

        with torch.inference_mode():
            for images, labels, context in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                if model.router_mode == "context_aware":
                    context = context.to(self.device)
                    _, _, topk_indices = model(images, context)
                else:
                    _, _, topk_indices = model(images)

                expert_mask = F.one_hot(
                    topk_indices,
                    num_classes=model.num_experts,
                ).sum(dim=1).float()
                expert_mask = expert_mask.clamp(max=1)

                for class_id in range(model.num_classes):
                    class_mask = labels == class_id
                    class_counts[class_id] += class_mask.sum().cpu()

                    if class_mask.any():
                        class_expert_counts[class_id] += (
                            expert_mask[class_mask].sum(dim=0).cpu()
                        )

        rho = class_expert_counts / (class_counts.unsqueeze(1) + 1e-9)
        return rho, class_counts

    def compute_entropy(
        self,
        rho: torch.Tensor,
        class_counts: torch.Tensor,
        idx_to_class: dict,
    ) -> pd.DataFrame:
        eps = 1e-9
        routing_probs = rho / (rho.sum(dim=1, keepdim=True) + eps)
        entropy = -(routing_probs * torch.log(routing_probs + eps)).sum(dim=1)
        normalized_entropy = entropy / torch.log(
            torch.tensor(rho.shape[1], dtype=torch.float32)
        )

        class_ids = list(range(rho.shape[0]))
        class_names = [idx_to_class.get(class_id, str(class_id)) for class_id in class_ids]

        return pd.DataFrame({
            "class_id": class_ids,
            "class_name": class_names,
            "num_samples": class_counts.numpy().astype(int),
            "entropy": entropy.numpy(),
            "normalized_entropy": normalized_entropy.numpy(),
        })

    def save_entropy(self, entropy_df: pd.DataFrame):
        csv_path = self.output_dir / self.csv_name
        plot_path = self.output_dir / self.plot_name

        entropy_df.to_csv(csv_path, index=False)

        plt.figure(figsize=(10, max(4, len(entropy_df) * 0.35)))
        plt.barh(
            entropy_df["class_name"],
            entropy_df["normalized_entropy"],
            color="#59A14F",
        )
        plt.xlabel("Normalized routing entropy")
        plt.ylabel("Class")
        plt.title(f"Routing Entropy per Class ({self.split})")
        plt.xlim(0, 1)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()

        return csv_path, plot_path

    def run(self, batch_size: int):
        checkpoint_info = self.extract_checkpoint()
        model = self.create_model(checkpoint_info)
        dataloader, dataset = self.create_dataloader(batch_size)
        rho, class_counts = self.collect_class_wise_routing(model, dataloader)
        entropy_df = self.compute_entropy(
            rho=rho,
            class_counts=class_counts,
            idx_to_class=dataset.idx_to_class,
        )
        csv_path, plot_path = self.save_entropy(entropy_df)

        print(entropy_df.to_string(index=False))
        print(f"Saved CSV: {csv_path}")
        print(f"Saved plot: {plot_path}")


def get_args():
    parser = argparse.ArgumentParser(
        description="Compute routing entropy per class from a MoE checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to MoE checkpoint, usually best_checkpoint.pth",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to save entropy CSV and plot",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split for diagnostics",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for diagnostic inference",
    )
    parser.add_argument(
        "--csv_name",
        type=str,
        default="routing_entropy_per_class.csv",
        help="Output CSV file name",
    )
    parser.add_argument(
        "--plot_name",
        type=str,
        default="routing_entropy_per_class.png",
        help="Output entropy plot file name",
    )
    return parser.parse_args()


def main():
    args = get_args()
    diagnostics = MoERoutingEntropy(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        split=args.split,
        csv_name=args.csv_name,
        plot_name=args.plot_name,
    )
    diagnostics.run(batch_size=args.batch_size)


if __name__ == "__main__":
    main()
