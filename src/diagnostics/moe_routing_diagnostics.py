from pathlib import Path
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from models.moe.gating import ContextAwareLinearGating
from models.moe.model import MoEModel
from dataset.plantdoc_dataset import build_datasets


class MoeRoutingDiagnostics:
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

    
    def extract_checkpoint(self, checkpoint_path: Path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model_state_dict = checkpoint['model_state_dict']
        num_classes = checkpoint['num_classes']
        num_experts = checkpoint['num_experts']
        top_k = checkpoint['top_k']
        temperature = checkpoint['temperature']
        context_dim = checkpoint['context_dim']
        router_mode = checkpoint['router_mode']
        return {
            'model_state_dict': model_state_dict,
            'num_classes': num_classes,
            'num_experts': num_experts,
            'top_k': top_k,
            'temperature': temperature,
            'context_dim': context_dim,
            'router_mode': router_mode
        }
    

    def create_model(self, num_classes: int, num_experts: int, top_k: int, temperature: float, context_dim: int, router_mode: str) -> MoEModel:
        model = MoEModel(
            num_classes=num_classes,
            num_experts=num_experts,
            top_k=top_k,
            temperature=temperature,
            context_dim=context_dim,
            router_mode=router_mode
        )
        return model
    

    def load_checkpoint(self, model: MoEModel, checkpoint_path: Path):
        checkpoint_info = self.extract_checkpoint(checkpoint_path)
        model.load_state_dict(checkpoint_info['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model
    

    def create_dataloader(self, batch_size: int = 32) -> DataLoader:
        train_dataset, val_dataset, test_dataset = build_datasets(use_context=True)
        datasets = {
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset,
        }
        dataset = datasets[self.split]
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return dataloader

    @staticmethod
    def uses_linear_context_gating(state_dict: dict) -> bool:
        return (
            "moe_layer.gating.gate_projector.weight" in state_dict
            and "moe_layer.gating.gate_projector.0.weight" not in state_dict
        )
    
    def collect_global_expert_usage(
        self,
        model: MoEModel,
        dataloader: DataLoader,
    ) -> pd.DataFrame:
        expert_counts = torch.zeros(model.num_experts, dtype=torch.long)
        total_selections = 0

        with torch.inference_mode():
            for images, labels, context in dataloader:
                images = images.to(self.device)

                if model.router_mode == "context_aware":
                    context = context.to(self.device)
                    _, _, topk_indices = model(images, context)
                else:
                    _, _, topk_indices = model(images)

                topk_indices = topk_indices.cpu()
                expert_counts += torch.bincount(
                    topk_indices.reshape(-1),
                    minlength=model.num_experts,
                )
                total_selections += topk_indices.numel()

        usage_rate = expert_counts.float() / max(total_selections, 1)

        return pd.DataFrame({
            "expert": list(range(model.num_experts)),
            "count": expert_counts.numpy(),
            "usage_rate": usage_rate.numpy(),
            "usage_percent": (usage_rate * 100).numpy(),
        })

    def save_global_expert_usage(self, usage_df: pd.DataFrame):
        csv_path = self.output_dir / self.csv_name
        plot_path = self.output_dir / self.plot_name

        usage_df.to_csv(csv_path, index=False)

        plt.figure(figsize=(6, 4))
        plt.bar(
            (usage_df["expert"] + 1).astype(str),
            usage_df["usage_percent"],
            color="#4C78A8",
        )
        plt.xlabel("Expert")
        plt.ylabel("Usage (%)")
        plt.title(f"Global Expert Utilization")
        plt.ylim(0, 100)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()

        return csv_path, plot_path

    def run_global_expert_usage(self, batch_size: int = 32):
        checkpoint_info = self.extract_checkpoint(self.checkpoint_path)
        model = self.create_model(
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

        test_loader = self.create_dataloader(batch_size=batch_size)
        usage_df = self.collect_global_expert_usage(model, test_loader)
        csv_path, plot_path = self.save_global_expert_usage(usage_df)

        print(usage_df.to_string(index=False))
        print(f"Saved CSV: {csv_path}")
        print(f"Saved plot: {plot_path}")

    def run_diagnostics(self, batch_size: int = 32):
        self.run_global_expert_usage(batch_size=batch_size)


def get_args():
    parser = argparse.ArgumentParser(
        description="MoE routing diagnostics",
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
        help="Directory to save diagnostic CSV and plots",
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
        default="global_expert_usage.csv",
        help="Output CSV file name",
    )
    parser.add_argument(
        "--plot_name",
        type=str,
        default="global_expert_usage.png",
        help="Output plot file name",
    )
    return parser.parse_args()


def main():
    args = get_args()
    diagnostics = MoeRoutingDiagnostics(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        split=args.split,
        csv_name=args.csv_name,
        plot_name=args.plot_name,
    )
    diagnostics.run_diagnostics(batch_size=args.batch_size)


if __name__ == "__main__":
    main()
