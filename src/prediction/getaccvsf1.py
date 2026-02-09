import os
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from models.mobilenetv2 import model as mobilenetv2
from models.mobilenetv3_small import model as mobilenetv3_small
from models.mobilenetv3_large import model as mobilenetv3_large
from models.mobilenetv4 import model as mobilenetv4
from dataset.plantdoc_dataset import test_dataset


test_ds = DataLoader(test_dataset, batch_size=32, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



checkpoint_paths = {
    "mobilenetv2": "/media/data/minhht/moe_plantdeasse/checkpoints/plantdoc/pretrain_weight/mobilenetv2/run_20260126-115525/best_checkpoint.pth",
    "mobilenetv3_small": "/media/data/minhht/moe_plantdeasse/checkpoints/plantdoc/pretrain_weight/mobilenetv3_small/run_20260126-124333/best_checkpoint.pth",
    "mobilenetv3_large": "/media/data/minhht/moe_plantdeasse/checkpoints/plantdoc/pretrain_weight/mobilenetv3_large/run_20260126-122233/best_checkpoint.pth",
    "mobbilenetv4": "/media/data/minhht/moe_plantdeasse/checkpoints/plantdoc/pretrain_weight/mobilenetv4/run_20260126-125702/best_checkpoint.pth"
}

models = {
    "mobilenetv2": mobilenetv2,
    "mobilenetv3_small": mobilenetv3_small,
    "mobilenetv3_large": mobilenetv3_large,
    "mobbilenetv4": mobilenetv4
}


@torch.no_grad()
def evaluate_checkpoint(model, checkpoint_path, test_loader, device):
    # load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return acc, macro_f1

results = []
for model_name, checkpoint_path in checkpoint_paths.items():
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = models[model_name]
    acc, macro_f1 = evaluate_checkpoint(model, checkpoint_path, test_ds, device)
    results.append({
        "model": model_name,
        "accuracy": acc,
        "macro_f1": macro_f1
    })

df_results = pd.DataFrame(results)

# Sort by Macro-F1 (optional, rất hay dùng cho paper)
df_results = df_results.sort_values(by="macro_f1", ascending=False)

# # Save CSV
output_csv = "/media/data/minhht/moe_plantdeasse/src/prediction/acc_vs_f1.csv"
df_results.to_csv(output_csv, index=False)

# print(f"\n📁 Results saved to: {output_csv}")
print(df_results)
