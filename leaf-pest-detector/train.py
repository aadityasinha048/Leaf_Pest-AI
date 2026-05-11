"""
🌿 Leaf Pest Detection — Model Training Script (PyTorch)
=========================================================
Train a MobileNetV2-based classifier for leaf pest detection.

Usage:
  1. Real training:  Place images in data/<class_name>/ folders, then run:
     python train.py

  2. Demo mode:  Creates a pre-configured model without real data:
     python train.py --demo

Classes: healthy, aphid, caterpillar
"""

import os
import sys
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

# ── Configuration ────────────────────────────────────────────────────────────
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
NUM_CLASSES = 3
CLASS_NAMES = ['aphid', 'caterpillar', 'healthy']
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'leaf_pest_model.pth')
CONFIG_PATH = os.path.join(MODEL_DIR, 'model_config.json')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_model(num_classes=NUM_CLASSES):
    """Build MobileNetV2-based transfer learning model."""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # Freeze all pretrained layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, num_classes)
    )

    return model.to(DEVICE)


def get_transforms():
    """Get data transforms for training and validation."""
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def train_real(model):
    """Train with real images from data/ directory."""
    if not os.path.exists(DATA_DIR):
        print(f"❌ Data directory not found: {DATA_DIR}")
        print("   Create folders: data/healthy/, data/aphid/, data/caterpillar/")
        print("   Place 100-200 images per class, then re-run.")
        print("\n   Or use:  python train.py --demo")
        sys.exit(1)

    # Check class folders
    found_classes = sorted([
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    ])

    if len(found_classes) < 2:
        print(f"❌ Need at least 2 class folders in data/. Found: {found_classes}")
        sys.exit(1)

    print(f"📂 Found classes: {found_classes}")

    train_transform, val_transform = get_transforms()

    # Load full dataset with train transform
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)

    # Split 80/20
    total = len(full_dataset)
    train_size = int(0.8 * total)
    val_size = total - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)

    actual_classes = full_dataset.classes
    print(f"   Training samples: {train_size}")
    print(f"   Validation samples: {val_size}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    # Training loop
    best_val_acc = 0.0
    print(f"\n🏋️ Training for {EPOCHS} epochs on {DEVICE}...\n")

    for epoch in range(EPOCHS):
        # ── Train ──
        model.train()
        running_loss = 0.0
        correct = 0
        total_samples = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total_samples += labels.size(0)

        train_loss = running_loss / total_samples
        train_acc = correct / total_samples

        # ── Validate ──
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0

        print(f"  Epoch {epoch+1}/{EPOCHS}  │  "
              f"Train Loss: {train_loss:.4f}  │  "
              f"Train Acc: {train_acc:.1%}  │  "
              f"Val Acc: {val_acc:.1%}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    return best_val_acc, actual_classes


def create_demo_model(model):
    """Create a demo model (saves architecture with initial weights)."""
    print("🎭 Creating demo model...")
    print("   This model uses image color analysis for predictions.")
    print("   For real accuracy, train with actual leaf images.\n")
    return None, CLASS_NAMES


def save_model(model, class_names, is_demo=False):
    """Save model and configuration."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save model state dict
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"💾 Model saved to: {MODEL_PATH}")

    # Save config
    config = {
        'class_names': class_names,
        'img_size': IMG_SIZE,
        'is_demo': is_demo,
        'num_classes': len(class_names),
        'framework': 'pytorch',
        'descriptions': {
            'healthy': {
                'name': 'Healthy Leaf',
                'emoji': '✅',
                'description': 'The leaf appears healthy with no signs of pest damage. Good plant health indicators observed.',
                'recommendation': 'Continue regular care and monitoring. No treatment needed.'
            },
            'aphid': {
                'name': 'Aphid Infestation',
                'emoji': '🐛',
                'description': 'Aphids detected on the leaf surface. These small insects suck plant sap, causing leaf curling and yellowing.',
                'recommendation': 'Apply neem oil spray or introduce ladybugs as natural predators. Remove heavily infested leaves.'
            },
            'caterpillar': {
                'name': 'Caterpillar Damage',
                'emoji': '🦋',
                'description': 'Signs of caterpillar feeding detected. Irregular holes and chewed leaf edges indicate active caterpillar presence.',
                'recommendation': 'Hand-pick caterpillars if few. Apply Bt (Bacillus thuringiensis) spray for larger infestations.'
            }
        }
    }

    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"📋 Config saved to: {CONFIG_PATH}")


def main():
    is_demo = '--demo' in sys.argv

    print("=" * 55)
    print("  🌿 Leaf Pest Detection — Model Training (PyTorch)")
    print("=" * 55)
    print()

    if is_demo:
        print("Mode: 🎭 DEMO (no real training data needed)\n")
    else:
        print("Mode: 🏋️ REAL TRAINING\n")

    print(f"Device: {DEVICE}\n")

    # Build model
    print("🔧 Building MobileNetV2 model...")
    model = build_model()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print()

    if is_demo:
        _, class_names = create_demo_model(model)
        save_model(model, class_names, is_demo=True)
    else:
        best_acc, class_names = train_real(model)
        save_model(model, class_names, is_demo=False)
        print(f"\n📊 Best Validation Accuracy: {best_acc:.1%}")

    print(f"\n✅ Done! Run the app:  python app.py")


if __name__ == '__main__':
    main()
