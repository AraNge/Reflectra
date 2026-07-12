import json
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
history_path = PROJECT_ROOT / "results" / "train_projection_history.json"

with open(history_path, "r", encoding="utf-8") as f:
    history = json.load(f)

epochs = [x["epoch"] for x in history]
train_loss = [x["train_loss"] for x in history]
val_loss = [x.get("val_loss") for x in history]

plt.figure(figsize=(7,5))

plt.plot(epochs, train_loss, marker="o", label="Train Loss")
plt.plot(epochs, val_loss, marker="o", label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Projection Head Training")
plt.grid(True)
plt.legend()

output = PROJECT_ROOT / "results" / "loss_curve.png"

plt.savefig(output, dpi=300)

print(f"Saved graph to: {output}")

plt.show()