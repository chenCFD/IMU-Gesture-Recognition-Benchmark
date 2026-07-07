import os
import argparse
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from torch.utils.data import DataLoader
from imu_hgr_dataloader import imu_gesture_dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# =========================================================================
# ===== Configuration & Argument Parsing =====
# =========================================================================
EPOCHS = 5
BATCH_SIZE = 32
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset Parameter Mapping
DATASET_MAP = {
    "imu1": {"timestamp": 50, "imu_feature": 6, "n_classes": 5},
    "imu2": {"timestamp": 24, "imu_feature": 12, "n_classes": 7},
    "imu3": {"timestamp": 90, "imu_feature": 18, "n_classes": 8},
}

def parse_args():
    parser = argparse.ArgumentParser(description="Universal Training Script for IMU Gesture Recognition")
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        help="Name of the model file inside the 'models' folder (e.g., HUAWEI_transformer_2026, RCNN)"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True, 
        choices=["imu1", "imu2", "imu3"], 
        help="Target dataset configuration"
    )
    return parser.parse_args()


# =========================================================================
# ===== Train / Eval Loops =====
# =========================================================================
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_cls = torch.argmax(y, dim=1)  # Convert one-hot to class index

        optimizer.zero_grad()
        out = model(x)

        loss = criterion(out, y_cls)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = torch.argmax(out, dim=1)
        correct += (pred == y_cls).sum().item()
        total += y.size(0)

    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_cls = torch.argmax(y, dim=1)

            out = model(x)
            loss = criterion(out, y_cls)

            total_loss += loss.item()
            pred = torch.argmax(out, dim=1)
            correct += (pred == y_cls).sum().item()
            total += y.size(0)

    return total_loss / len(loader), correct / total


def get_predictions(model, loader):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_cls = torch.argmax(y, dim=1)
            out = model(x)
            pred = torch.argmax(out, dim=1)

            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(y_cls.cpu().numpy())
            
    return np.array(all_targets), np.array(all_preds)


# =========================================================================
# ===== Main Process =====
# =========================================================================
def main():
    args = parse_args()
    config = DATASET_MAP[args.dataset]
    n_classes = config["n_classes"]

    # Setup Checkpoint Directory
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    MODEL_DIR = os.path.join("ckpt", f"{args.model}_{args.dataset}_{current_time}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"[Info] Artifacts will be saved to: {MODEL_DIR}")

    # Load Dataset Setup
    print(f"[Info] Loading dataset folders for {args.dataset}...")
    train_ds = imu_gesture_dataset(f"./feature/splited/train_{args.dataset}")
    valid_ds = imu_gesture_dataset(f"./feature/splited/valid_{args.dataset}")
    test_ds  = imu_gesture_dataset(f"./feature/splited/test_{args.dataset}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    # Dynamic Model Import
    print(f"[Info] Importing model module 'models.{args.model}'...")
    try:
        model_module = importlib.import_module(f"models.{args.model}")
    except ModuleNotFoundError:
        print(f"[Error] Could not find file: models/{args.model}.py")
        return

    # Intelligent Class Resolution Strategy
    ModelClass = None
    
    # Priority 1: Look for a class matching the filename exactly
    if hasattr(model_module, args.model):
        ModelClass = getattr(model_module, args.model)
    else:
        # Priority 2: Fallback check for any class inheriting from nn.Module in that file
        for attr_name in dir(model_module):
            attr = getattr(model_module, attr_name)
            if isinstance(attr, type) and issubclass(attr, nn.Module) and attr.__module__ == model_module.__name__:
                ModelClass = attr
                break

    if ModelClass is None:
        print(f"[Error] No valid PyTorch nn.Module class discovered inside models/{args.model}.py")
        return
    
    print(f"[Info] Successfully bound to model class: '{ModelClass.__name__}'")

    # Instantiate Model with your unified parameter dictionary
    try:
        model = ModelClass(
            input_dim=config["imu_feature"],
            time_steps=config["timestamp"],
            n_classes=n_classes
        ).to(DEVICE)
    except Exception as e:
        print(f"[Error] Failed to initialize {ModelClass.__name__}. Please check your __init__ parameters. Error: {e}")
        return
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Info] Model loaded successfully. Total params: {total_params} ({total_params*4/1024:.1f} KB)")

    # Optimization Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-5
    )

    best_val_acc = 0.0
    best_val_loss = float('inf') 
    patience, patience_counter = 40, 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    print("[Info] Start training...")

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, valid_loader, criterion)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        is_improved = False

        # Save Best Acc Model
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best.pt"))
            is_improved = True
            print(f" => Saved best Acc model ({best_val_acc:.4f})")

        # Save Best Loss Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_loss.pt"))
            is_improved = True
            print(f" => Saved best Loss model ({best_val_loss:.4f})")

        # Early stopping logic
        if is_improved:
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "last.pt"))
            print("[Info] Early stopping triggered.")
            break

        if epoch == EPOCHS - 1:
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "last.pt"))

    # =========================================================================
    # ===== Evaluation & Plotting =====
    # =========================================================================
    class_names = [str(i) for i in range(1, n_classes + 1)]
    
    def evaluate_and_plot(model_filename, title_prefix, colormap, output_filename):
        print(f"\n{'='*40}\nEvaluating [{title_prefix}] Model...\n{'='*40}")
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, model_filename)))
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        print(f"{title_prefix} Model - Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc*100:.2f}%")

        y_true, y_pred = get_predictions(model, test_loader)
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap=colormap, values_format='d')
        plt.title(f"Confusion Matrix - {title_prefix} Model (Test Acc: {test_acc*100:.2f}%)")
        plt.savefig(os.path.join(MODEL_DIR, output_filename))
        plt.close()
        print(f"Saved confusion matrix plot for {title_prefix} model.")

    evaluate_and_plot("best.pt", "Best Acc", plt.cm.Blues, "confusion_matrix_best_acc.png")
    evaluate_and_plot("best_loss.pt", "Best Loss", plt.cm.Oranges, "confusion_matrix_best_loss.png")
    evaluate_and_plot("last.pt", "Last", plt.cm.Greens, "confusion_matrix_last.png")

    # Plot Training History
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.title("Loss")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(train_accs, label='train')
    plt.plot(val_accs, label='val')
    plt.title("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(MODEL_DIR, "training_history.png"))
    plt.close()
    print("\n[Info] Saved training history plot.")


if __name__ == "__main__":
    main()