"""
Stage 1 & 2: CNN Pre-training and Fine-tuning
Section 2.3: "Supervised pre-training" and "Domain-specific fine-tuning"
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import argparse

from models.cnn import AlexNetFeatureExtractor
from config.config import RCNNConfig


def finetune_cnn(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: RCNNConfig,
    device: str = "cuda"
):
    model = model.to(device)
    model.train()

    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.finetune_learning_rate,
        momentum=0.9,
        weight_decay=0.0005
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()

    print("Starting fine-tuning...")
    print(f"Learning rate: {config.finetune_learning_rate}")
    print(f"Batch size: {config.finetune_batch_size}")

    for epoch in range(config.finetune_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            logits = model(images, extract_features=False)
            loss = criterion(logits, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{config.finetune_epochs}] "
                      f"Batch [{batch_idx+1}] "
                      f"Loss: {running_loss/(batch_idx+1):.4f} "
                      f"Acc: {100.*correct/total:.2f}%")

        # Validation
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch [{epoch+1}/{config.finetune_epochs}] "
              f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            save_path = config.output_dir / f"finetuned_cnn_epoch{epoch+1}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint to {save_path}")

    final_path = config.output_dir / "finetuned_cnn_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"Fine-tuning complete. Saved to {final_path}")


def evaluate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            logits = model(images, extract_features=False)
            loss = criterion(logits, labels)

            val_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return val_loss / len(loader), 100. * correct / total


def main():
    parser = argparse.ArgumentParser(description='R-CNN CNN Training')
    parser.add_argument('--stage', type=str, default='finetune',
                       choices=['pretrain', 'finetune'],
                       help='Training stage')
    parser.add_argument('--dataset', type=str, default='airbus',
                       help='Dataset name')
    args = parser.parse_args()

    config = RCNNConfig()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    if args.stage == 'pretrain':
        print("Pre-training on ImageNet...")
        raise NotImplementedError("Use ImageNet pre-trained weights instead")

    elif args.stage == 'finetune':
        print("Fine-tuning on detection data...")

        # Dataset paths (Airbus dataset)
        BASE_DIR = Path("dataset/airbus-aircrafts-sample-dataset")
        ANNOTATIONS_CSV = BASE_DIR / "annotations.csv"
        IMAGES_DIR = BASE_DIR / "images"

        if not ANNOTATIONS_CSV.exists():
            print(f"Error: Annotations not found at {ANNOTATIONS_CSV}")
            print("Please ensure the Airbus dataset is available.")
            exit(1)
        
        model = AlexNetFeatureExtractor(
            pretrained=True,
            feature_layer=config.feature_layer,
            num_classes=config.num_classes
        )

        from rcnn.data import create_rcnn_dataloaders

        print("\nCreating dataloaders...")
        print(f"  Dataset: {args.dataset}")
        print(f"  Annotations: {ANNOTATIONS_CSV}")
        print(f"  Images: {IMAGES_DIR}")

        train_loader, val_loader = create_rcnn_dataloaders(
            annotations_csv=ANNOTATIONS_CSV,
            images_dir=IMAGES_DIR,
            stage="finetune",
            batch_size=config.finetune_batch_size,
            num_positive_per_batch=32,
            num_workers=4,
            train_val_split=0.8,
        )

        print(f"\nTrain batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")

        print("\nStarting fine-tuning...")
        finetune_cnn(model, train_loader, val_loader, config, device)


if __name__ == "__main__":
    main()
