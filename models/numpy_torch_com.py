import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.sgd import SGD
from torchvision import models
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from resnet18_numpy import ResNet18, cross_entropy_loss, accuracy as np_accuracy

# --- Torch ResNet18 wrapper ---
class TorchResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    def forward(self, x):
        return self.model(x)

def preprocess_numpy(X):
    # Normalize to [0, 1]
    return X.astype(np.float32) / 255.0

def preprocess_torch(X):
    # Normalize to [0, 1] and convert to torch tensor
    X = torch.from_numpy(X).float() / 255.0
    # Normalize with CIFAR-10 mean/std
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1,3,1,1)
    X = (X - mean) / std
    return X

def main():
    # --- Load and prepare data ---
    DATASET_SIZE = 2000
    IMAGES_PER_CLASS = DATASET_SIZE // 10
    BATCH_SIZE = 20
    EPOCHS = 10
    LR = 0.2

    print("Loading CIFAR-10 from OpenML...")
    cifar10 = fetch_openml('CIFAR_10_small', version=1)
    X = cifar10.data
    y = cifar10.target.astype(int)
    X = X.to_numpy().reshape(-1, 3, 32, 32)
    y = y.to_numpy()

    # Balanced sampling
    selected_indices = []
    for class_idx in range(10):
        class_indices = np.where(y == class_idx)[0]
        np.random.shuffle(class_indices)
        selected_indices.extend(class_indices[:IMAGES_PER_CLASS])
    selected_indices = np.array(selected_indices)
    np.random.shuffle(selected_indices)
    X, y = X[selected_indices], y[selected_indices]

    # Split train/val
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

    # --- Numpy ResNet18 ---
    model_np = ResNet18(num_classes=10)

    # --- Torch ResNet18 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_torch = TorchResNet18(num_classes=10).to(device)
    optimizer = SGD(model_torch.parameters(), lr=LR, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Preprocess for torch
    X_train_torch = preprocess_torch(X_train)
    X_val_torch = preprocess_torch(X_val)
    y_train_torch = torch.from_numpy(y_train).long()
    y_val_torch = torch.from_numpy(y_val).long()

    # --- Training loop ---
    for epoch in range(EPOCHS):
        # --- Numpy training ---
        idx = np.random.permutation(len(X_train))
        X_train_np, y_train_np = X_train[idx], y_train[idx]
        for i in range(0, len(X_train_np), BATCH_SIZE):
            X_batch = preprocess_numpy(X_train_np[i:i+BATCH_SIZE])
            y_batch = y_train_np[i:i+BATCH_SIZE]
            logits = model_np.forward(X_batch, train=True)
            loss, d_logits = cross_entropy_loss(logits, y_batch)
            model_np.backward(d_logits)
            model_np.step(LR)
        logits_val_np = model_np.forward(preprocess_numpy(X_val), train=False)
        acc_val_np = np_accuracy(logits_val_np, y_val)

        # --- Torch training ---
        model_torch.train()
        perm = torch.randperm(len(X_train_torch))
        X_train_torch_shuf = X_train_torch[perm]
        y_train_torch_shuf = y_train_torch[perm]
        for i in range(0, len(X_train_torch_shuf), BATCH_SIZE):
            xb = X_train_torch_shuf[i:i+BATCH_SIZE].to(device)
            yb = y_train_torch_shuf[i:i+BATCH_SIZE].to(device)
            optimizer.zero_grad()
            out = model_torch(xb)
            loss_torch = criterion(out, yb)
            loss_torch.backward()
            optimizer.step()
        model_torch.eval()
        with torch.no_grad():
            out_val = model_torch(X_val_torch.to(device))
            preds = out_val.argmax(dim=1).cpu().numpy()
            acc_val_torch = np.mean(preds == y_val)

        print(f"Epoch {epoch+1}/{EPOCHS} | Numpy ResNet18 Val Acc: {acc_val_np:.4f} | Torch ResNet18 Val Acc: {acc_val_torch:.4f}")

if __name__ == "__main__":
    main()