import os
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from transformers import ViTImageProcessor, ViTForImageClassification, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# ============================================================
# DETERMINISTIC SEEDS
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device for ViT training: {device}")

# ============================================================
# PREPROCESSING & DATA LOADERS
# ============================================================
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

# Identical, clean preprocessing: NO destructive affine transformations
clean_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=processor.image_mean,
        std=processor.image_std
    )
])

train_data = ImageFolder("data/train", transform=clean_transform)
val_data = ImageFolder("data/val", transform=clean_transform)

print(f"Class to idx: {train_data.class_to_idx}")

# Batch size 14 -> 10 batches per epoch on 140 training images
BATCH_SIZE = 14
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# MODEL ARCHITECTURE & FINE-TUNING STRATEGY
# ============================================================
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=2,
    id2label={0: "genuine", 1: "tampered"},
    label2id={"genuine": 0, "tampered": 1},
    ignore_mismatched_sizes=True
)

# Freeze all layers initially
for param in model.parameters():
    param.requires_grad = False

# Unfreeze top 2 transformer layers (Layer 10 and 11) + Classifier Head
for layer in model.vit.layers[-2:]:
    for param in layer.parameters():
        param.requires_grad = True

for param in model.classifier.parameters():
    param.requires_grad = True

model.to(device)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.1f}%)")

# ============================================================
# OPTIMIZER & SCHEDULER (Differential Learning Rates)
# ============================================================
# Pretrained backbone layers: 2e-5, Classifier head: 1e-4
optimizer_grouped_parameters = [
    {
        "params": [p for layer in model.vit.layers[-2:] for p in layer.parameters() if p.requires_grad],
        "lr": 2e-5,
        "weight_decay": 0.01
    },
    {
        "params": [p for p in model.classifier.parameters() if p.requires_grad],
        "lr": 1e-4,
        "weight_decay": 0.01
    }
]

optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

EPOCHS = 10
total_steps = len(train_loader) * EPOCHS
warmup_steps = len(train_loader) * 1  # 1 epoch warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

criterion = nn.CrossEntropyLoss()

# ============================================================
# TRAINING & EVALUATION LOOP
# ============================================================
best_val_auc = 0.0
save_dir = "models/vit/saved"
os.makedirs(save_dir, exist_ok=True)

print("\n" + "=" * 80)
print(f"{'Epoch':<6} | {'Tr Loss':<8} {'Tr Acc':<7} | {'Val Loss':<9} {'Val Acc':<8} {'Val AUC':<8} {'Val F1':<7} | {'Gen Prob Mean':<14} {'Tamp Prob Mean':<14}")
print("-" * 80)

history = []

for epoch in range(1, EPOCHS + 1):
    # Train
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(pixel_values=images)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs.logits, dim=1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    avg_train_loss = train_loss / train_total
    avg_train_acc = train_correct / train_total

    # Validate
    model.eval()
    val_loss = 0.0
    val_labels_all = []
    val_preds_all = []
    val_probs_all = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(pixel_values=images)
            loss = criterion(outputs.logits, labels)
            val_loss += loss.item() * images.size(0)

            probs = torch.softmax(outputs.logits, dim=1)[:, 1]
            preds = torch.argmax(outputs.logits, dim=1)

            val_labels_all.extend(labels.cpu().numpy())
            val_preds_all.extend(preds.cpu().numpy())
            val_probs_all.extend(probs.cpu().numpy())

    val_total = len(val_labels_all)
    avg_val_loss = val_loss / val_total
    val_labels_all = np.array(val_labels_all)
    val_preds_all = np.array(val_preds_all)
    val_probs_all = np.array(val_probs_all)

    val_acc = accuracy_score(val_labels_all, val_preds_all)
    val_prec = precision_score(val_labels_all, val_preds_all, zero_division=0)
    val_rec = recall_score(val_labels_all, val_preds_all, zero_division=0)
    val_f1 = f1_score(val_labels_all, val_preds_all, zero_division=0)
    try:
        val_auc = roc_auc_score(val_labels_all, val_probs_all)
    except Exception:
        val_auc = 0.5

    gen_probs = val_probs_all[val_labels_all == 0]
    tamp_probs = val_probs_all[val_labels_all == 1]
    gen_mean = gen_probs.mean() if len(gen_probs) > 0 else 0.0
    tamp_mean = tamp_probs.mean() if len(tamp_probs) > 0 else 0.0

    print(f"{epoch:<6} | {avg_train_loss:<8.4f} {avg_train_acc:<7.4f} | {avg_val_loss:<9.4f} {val_acc:<8.4f} {val_auc:<8.4f} {val_f1:<7.4f} | {gen_mean:<14.4f} {tamp_mean:<14.4f}")

    epoch_record = {
        "epoch": epoch,
        "train_loss": avg_train_loss,
        "train_acc": avg_train_acc,
        "val_loss": avg_val_loss,
        "val_acc": val_acc,
        "val_auc": val_auc,
        "val_prec": val_prec,
        "val_rec": val_rec,
        "val_f1": val_f1,
        "gen_prob_mean": gen_mean,
        "tamp_prob_mean": tamp_mean,
        "confusion_matrix": confusion_matrix(val_labels_all, val_preds_all)
    }
    history.append(epoch_record)

    # Save best checkpoint strictly based on VALIDATION ROC-AUC
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        model.save_pretrained(save_dir)
        processor.save_pretrained(save_dir)
        print(f"  --> Saved new best checkpoint at Epoch {epoch} with Val ROC-AUC = {val_auc:.4f}")

print("=" * 80)
print(f"Best Validation ROC-AUC achieved: {best_val_auc:.4f}")
print("Final model saved to:", save_dir)
