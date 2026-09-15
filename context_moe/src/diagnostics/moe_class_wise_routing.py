from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset.plantdoc_dataset import build_datasets
from models.moe.model import MoEModel


class MoEClassWiseRouting:
    def __init__(
        self,
        checkpoint_path: Path,
        output_dir: Path,
        split: str,
        csv_name: str,
        heatmap_name: str,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.output_dir = Path(output_dir)
        self.split = split
        self.csv_name = csv_name
        self.heatmap_name = heatmap_name
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
        model.load_state_dict(checkpoint_info["model_state_dict"])
        model.to(self.device)
        model.eval()
        return model

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

    def save_class_wise_routing(
        self,
        rho: torch.Tensor,
        class_counts: torch.Tensor,
        idx_to_class: dict,
    ):
        rho_np = rho.numpy()
        class_counts_np = class_counts.numpy().astype(int)
        class_ids = list(range(rho_np.shape[0]))
        class_names = [idx_to_class.get(class_id, str(class_id)) for class_id in class_ids]

        routing_df = pd.DataFrame(
            rho_np,
            columns=[f"expert_{expert_id}" for expert_id in range(rho_np.shape[1])],
        )
        routing_df.insert(0, "class_id", class_ids)
        routing_df.insert(1, "class_name", class_names)
        routing_df.insert(2, "num_samples", class_counts_np)

        csv_path = self.output_dir / self.csv_name
        heatmap_path = self.output_dir / self.heatmap_name

        routing_df.to_csv(csv_path, index=False)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            rho_np,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
            xticklabels=[str(expert_id + 1) for expert_id in range(rho_np.shape[1])],
            yticklabels=class_names,
            annot_kws={"size": 15},
            linewidths=0.5,
            cbar_kws={"label": "Activation rate", "shrink": 0.8},
            ax=ax,
        )

        ax.set_xlabel("Expert", fontsize=15, labelpad=10)
        ax.set_ylabel("Class", fontsize=15, labelpad=10)
        # ax.set_title("Class-wise Expert Activation Heatmap", fontsize=14, pad=12)
        ax.tick_params(axis='x', labelsize=15, rotation=0)
        ax.tick_params(axis='y', labelsize=15, rotation=0)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=15)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)

        # Colorbar font
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=15)
        cbar.set_label("Activation rate", fontsize=15)

        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
        plt.close()
        return csv_path, heatmap_path, routing_df
    
    def run(self, batch_size: int):
        checkpoint_info = self.extract_checkpoint()
        model = self.create_model(checkpoint_info)
        dataloader, dataset = self.create_dataloader(batch_size)
        rho, class_counts = self.collect_class_wise_routing(model, dataloader)
        csv_path, heatmap_path, routing_df = self.save_class_wise_routing(
            rho=rho,
            class_counts=class_counts,
            idx_to_class=dataset.idx_to_class,
        )

        print(routing_df.to_string(index=False))
        print(f"Saved CSV: {csv_path}")
        print(f"Saved heatmap: {heatmap_path}")


def get_args():
    parser = argparse.ArgumentParser(
        description="Compute class-wise MoE routing heatmap.",
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
        help="Directory to save class-wise routing CSV and heatmap",
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
        default="class_wise_routing.csv",
        help="Output CSV file name",
    )
    parser.add_argument(
        "--heatmap_name",
        type=str,
        default="class_wise_routing_heatmap.png",
        help="Output heatmap image file name",
    )
    return parser.parse_args()


def main():
    args = get_args()
    diagnostics = MoEClassWiseRouting(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        split=args.split,
        csv_name=args.csv_name,
        heatmap_name=args.heatmap_name,
    )
    diagnostics.run(batch_size=args.batch_size)


if __name__ == "__main__":
    main()
