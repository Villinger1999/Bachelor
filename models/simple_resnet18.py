import numpy as np
import sys as sys
import cv2
import skimage.measure as measure
from sklearn.datasets import fetch_openml


def batchNorm(input: np.ndarray, gamma=1.0, beta=0.0, eps=1e-5):
    """Batch normalization

    Args:
        input (np.ndarray): image/tensor of size (C,H,W)
        gamma (float, optional): The scale parameter of the normalization, using the scale the values after nomalization as a trainable parameter. Defaults to 1.0.
        beta (float, optional): The shift parameter of the normalization, used to shift the values after normaliztion to not lose information due to the activation function. Defaults to 0.0.
        eps (_type_, optional): Small number added to the denominator during the normalization, to avoid zero division. Defaults to 1e-5.
        
    Returns:
        np.ndarray: batch-normalized tensor of same shape as input
    """
    # ensure input is a numpy array with shape (C,H,W)
    if not isinstance(input, np.ndarray):
        input = np.array(input)
    if input.ndim == 3:
        # (C, H, W)
        mean = np.mean(input, axis=(1,2), keepdims=True)
        std = np.std(input, axis=(1,2), keepdims=True)
        normalized = (input - mean) / (std + eps)
        return gamma * normalized + beta
    elif input.ndim == 4:
        # (N, C, H, W)
        mean = np.mean(input, axis=(0,2,3), keepdims=True)
        std = np.std(input, axis=(0,2,3), keepdims=True)
        normalized = (input - mean) / (std + eps)
        return gamma * normalized + beta
    else:
        raise ValueError("input must be a 3D or 4D array with shape (C, H, W) or (N, C, H, W)")
    
def convolution3d(input: np.ndarray, kernel, step_size) -> np.ndarray:
    """
    2D convolution for each channel, supports (N, C, H, W) or (C, H, W)
    kernel: (k, k)
    """
    if input.ndim == 3:
        # (C, H, W)
        c, y, x = input.shape
        kernel_size = kernel.shape[0]
        y_out = (y - kernel_size) // step_size + 1
        x_out = (x - kernel_size) // step_size + 1
        new_image = np.zeros((c, y_out, x_out))
        for chan in range(c):
            for i in range(0, y - kernel_size + 1, step_size):
                for j in range(0, x - kernel_size + 1, step_size):
                    new_image[chan, i // step_size, j // step_size] = np.sum(input[chan, i:i+kernel_size, j:j+kernel_size] * kernel)
        return new_image
    elif input.ndim == 4:
        # (N, C, H, W)
        n, c, y, x = input.shape
        kernel_size = kernel.shape[0]
        y_out = (y - kernel_size) // step_size + 1
        x_out = (x - kernel_size) // step_size + 1
        new_image = np.zeros((n, c, y_out, x_out))
        for batch in range(n):
            for chan in range(c):
                for i in range(0, y - kernel_size + 1, step_size):
                    for j in range(0, x - kernel_size + 1, step_size):
                        new_image[batch, chan, i // step_size, j // step_size] = np.sum(input[batch, chan, i:i+kernel_size, j:j+kernel_size] * kernel)
        return new_image
    else:
        raise ValueError("input must be a 3D or 4D array")
    
def ReLU(x):
    return np.maximum(x, 0)
    
def basicBlock(input: np.ndarray, kernel_1: np.ndarray, kernel_2: np.ndarray, stride1: int = 1, project_shortcut: bool = False) -> np.ndarray:
    """ResNet BasicBlock for (N, C, H, W) or (C, H, W). project_shortcut: if True, applies 1x1 conv to shortcut."""
    # Padding for 3x3 conv
    pad = 1
    if input.ndim == 3:
        padded_input = np.pad(input, ((0,0),(pad,pad),(pad,pad)), mode='constant')
    else:
        padded_input = np.pad(input, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant')
    # First conv
    out = convolution3d(padded_input, kernel_1, step_size=stride1)
    out = batchNorm(out)
    out = ReLU(out)
    # Second conv
    if out.ndim == 3:
        out = np.pad(out, ((0,0),(pad,pad),(pad,pad)), mode='constant')
    else:
        out = np.pad(out, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant')
    out = convolution3d(out, kernel_2, step_size=1)
    out = batchNorm(out)
    # Shortcut connection
    shortcut = input
    if project_shortcut:
        # 1x1 conv to match shape
        k1x1 = np.ones((1,1))
        shortcut = convolution3d(input, k1x1, step_size=stride1)
    # Add
    out = out + shortcut
    return ReLU(out)

def max_pool(input: np.ndarray, kernel_size: int = 2, stride: int = 2, padding: int = 0) -> np.ndarray:
    """Max pooling for (N, C, H, W) or (C, H, W)"""
    if padding > 0:
        if input.ndim == 3:
            input = np.pad(input, ((0,0),(padding,padding),(padding,padding)), mode='constant')
        else:
            input = np.pad(input, ((0,0),(0,0),(padding,padding),(padding,padding)), mode='constant')
    if input.ndim == 3:
        c, h, w = input.shape
        out_h = (h - kernel_size) // stride + 1
        out_w = (w - kernel_size) // stride + 1
        out = np.zeros((c, out_h, out_w))
        for ch in range(c):
            for i in range(0, h - kernel_size + 1, stride):
                for j in range(0, w - kernel_size + 1, stride):
                    out[ch, i//stride, j//stride] = np.max(input[ch, i:i+kernel_size, j:j+kernel_size])
        return out
    elif input.ndim == 4:
        n, c, h, w = input.shape
        out_h = (h - kernel_size) // stride + 1
        out_w = (w - kernel_size) // stride + 1
        out = np.zeros((n, c, out_h, out_w))
        for batch in range(n):
            for ch in range(c):
                for i in range(0, h - kernel_size + 1, stride):
                    for j in range(0, w - kernel_size + 1, stride):
                        out[batch, ch, i//stride, j//stride] = np.max(input[batch, ch, i:i+kernel_size, j:j+kernel_size])
        return out
    else:
        raise ValueError("input must be 3D or 4D")

def global_avg_pool(input: np.ndarray) -> np.ndarray:
    """Global average pooling for (N, C, H, W) or (C, H, W)"""
    if input.ndim == 3:
        return np.mean(input, axis=(1,2), keepdims=True)
    elif input.ndim == 4:
        return np.mean(input, axis=(2,3), keepdims=True)
    else:
        raise ValueError("input must be 3D or 4D")

def flatten(input: np.ndarray) -> np.ndarray:
    return input.reshape(input.shape[0], -1) if input.ndim == 4 else input.reshape(1, -1)


class SimpleResNet18:
    def __init__(self, num_classes=10):
        # Only the FC layer is trainable
        self.num_classes = num_classes
        self.feature_dim = None  # Will be set after first forward
        self.fc_weight = None
        self.fc_bias = None
        # Store random kernels for all convs (fixed, not updated)
        self.kernels = [np.random.randn(7,7)]
        for _ in range(16):
            self.kernels.append(np.random.randn(3,3))

    def forward(self, x):
        # Initial conv: 7x7, stride 2, pad 3
        x = np.pad(x, ((0,0),(0,0),(3,3),(3,3)), mode='constant')
        x = convolution3d(x, self.kernels[0], step_size=2)
        x = batchNorm(x)
        x = ReLU(x)
        x = max_pool(x, kernel_size=3, stride=2, padding=1)
        # Stage 1: 2 blocks
        x = basicBlock(x, self.kernels[1], self.kernels[2], stride1=1, project_shortcut=False)
        x = basicBlock(x, self.kernels[3], self.kernels[4], stride1=1, project_shortcut=False)
        # Stage 2: 2 blocks, first block stride=2
        x = basicBlock(x, self.kernels[5], self.kernels[6], stride1=2, project_shortcut=True)
        x = basicBlock(x, self.kernels[7], self.kernels[8], stride1=1, project_shortcut=False)
        # Stage 3: 2 blocks, first block stride=2
        x = basicBlock(x, self.kernels[9], self.kernels[10], stride1=2, project_shortcut=True)
        x = basicBlock(x, self.kernels[11], self.kernels[12], stride1=1, project_shortcut=False)
        # Stage 4: 2 blocks, first block stride=2
        x = basicBlock(x, self.kernels[13], self.kernels[14], stride1=2, project_shortcut=True)
        x = basicBlock(x, self.kernels[15], self.kernels[16], stride1=1, project_shortcut=False)
        # Global average pooling
        x = global_avg_pool(x)
        x = flatten(x)
        # FC layer: initialize if needed
        if self.fc_weight is None or self.fc_bias is None:
            self.feature_dim = x.shape[1]
            self.fc_weight = np.random.randn(self.feature_dim, self.num_classes) * 0.01
            self.fc_bias = np.zeros(self.num_classes)
        out = x @ self.fc_weight + self.fc_bias
        return out, x  # return features for backward

    def predict(self, x):
        logits, _ = self.forward(x)
        return np.argmax(logits, axis=1)

    def train_fc(self, X, y, X_val, y_val, epochs=5, lr=0.1, batch_size=64):
        N = X.shape[0]
        for epoch in range(epochs):
            idx = np.random.permutation(N)
            X, y = X[idx], y[idx]
            for i in range(0, N, batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size].astype(int)
                logits, feats = self.forward(X_batch)
                probs = softmax(logits)
                # Cross-entropy loss and gradient
                N_batch = X_batch.shape[0]
                dlogits = probs
                dlogits[np.arange(N_batch), y_batch] -= 1
                dlogits /= N_batch
                # Gradients for FC layer
                grad_w = feats.T @ dlogits
                grad_b = np.sum(dlogits, axis=0)
                # Update
                self.fc_weight -= lr * grad_w
                self.fc_bias -= lr * grad_b
            # Validation accuracy
            val_logits, _ = self.forward(X_val)
            val_acc = accuracy(val_logits, y_val)
            print(f"Epoch {epoch+1}, Validation Accuracy: {val_acc:.4f}")



    # Add this for accuracy evaluation
def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

def accuracy(logits, targets):
    preds = np.argmax(logits, axis=1)
    return np.mean(preds == targets.astype(int))

if __name__ == "__main__":
    # Download CIFAR-10 (first time will take a while)
    cifar10 = fetch_openml('CIFAR_10_small', version=1)
    X = cifar10.data  # shape (60000, 3072)
    y = cifar10.target

    # Reshape to (N, 3, 32, 32)
    X = X.to_numpy().reshape(-1, 3, 32, 32).astype(np.float32) / 255.0

    # Split into train/val (80% train, 20% val)
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    # Create and train the model (only FC layer is trained)
    model = SimpleResNet18(num_classes=10)
    model.train_fc(X_train, y_train, X_val, y_val, epochs=5, lr=0.1, batch_size=64)

    # Evaluate on validation set
    val_logits, _ = model.forward(X_val)
    val_acc = accuracy(val_logits, y_val)
    print("Final validation accuracy:", val_acc)
    