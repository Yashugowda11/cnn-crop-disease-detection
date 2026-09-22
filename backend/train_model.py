import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms


DEFAULT_DATASET = Path(__file__).parent.parent / "dataset" / "dataset" / "plantvillage dataset" / "color"
DEFAULT_OUTPUT = Path(__file__).parent / "model"
IMAGE_SIZE = 224


def build_model(class_count: int) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, class_count)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PlantVillage crop disease CNN.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--max-images", type=int, default=0, help="Optional limit for quick experiments; 0 uses all images.")
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Dataset folder was not found: {args.data}")

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    validation_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(args.data, transform=train_transform)
    validation_dataset = datasets.ImageFolder(args.data, transform=validation_transform)
    classes = train_dataset.classes
    class_counts = {
        class_name: train_dataset.targets.count(class_index)
        for class_index, class_name in enumerate(classes)
    }
    if args.max_images and args.max_images < len(train_dataset):
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(train_dataset), generator=generator)[:args.max_images].tolist()
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        validation_dataset = torch.utils.data.Subset(validation_dataset, indices)

    validation_size = max(1, int(len(train_dataset) * args.validation_split))
    training_size = len(train_dataset) - validation_size
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(train_dataset), generator=generator).tolist()
    training_indices = indices[:training_size]
    validation_indices = indices[training_size:]
    training_set = torch.utils.data.Subset(train_dataset, training_indices)
    validation_set = torch.utils.data.Subset(validation_dataset, validation_indices)

    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)

    best_accuracy = 0.0
    args.output.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in validation_loader:
                outputs = model(images.to(device))
                correct += (outputs.argmax(1).cpu() == labels).sum().item()
                total += labels.size(0)
        accuracy = correct / max(total, 1)
        print(f"Epoch {epoch + 1}/{args.epochs} - loss: {running_loss / max(training_size, 1):.4f} - validation accuracy: {accuracy:.4f}")
        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), args.output / "plant_disease_resnet18.pth")

    metadata = {
        "classes": classes,
        "class_counts": class_counts,
        "image_size": IMAGE_SIZE,
        "validation_accuracy": best_accuracy,
        "architecture": "resnet18",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved model to {args.output}")
    print(f"Classes: {len(classes)} | Best validation accuracy: {best_accuracy:.4f} | Device: {device}")


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    main()
