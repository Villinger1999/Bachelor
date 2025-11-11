import numpy as np

def im2col(X, kernel_size, stride):
    N, C, H, W = X.shape
    k = kernel_size
    out_h = (H - k) // stride + 1
    out_w = (W - k) // stride + 1
    cols = np.zeros((N, C, k, k, out_h, out_w))
    for y in range(k):
        y_max = y + stride * out_h
        for x in range(k):
            x_max = x + stride * out_w
            cols[:, :, y, x, :, :] = X[:, :, y:y_max:stride, x:x_max:stride]
    return cols.reshape(N, C * k * k, out_h * out_w)

def col2im(cols, X_shape, kernel_size, stride):
    N, C, H, W = X_shape
    k = kernel_size
    out_h = (H - k) // stride + 1
    out_w = (W - k) // stride + 1
    cols = cols.reshape(N, C, k, k, out_h, out_w)
    X = np.zeros((N, C, H, W))
    for y in range(k):
        y_max = y + stride * out_h
        for x in range(k):
            x_max = x + stride * out_w
            X[:, :, y:y_max:stride, x:x_max:stride] += cols[:, :, y, x, :, :]
    return X

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        scale = np.sqrt(2. / (in_channels * kernel_size * kernel_size))
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.b = np.zeros(out_channels)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
    def forward(self, X):
        self.X = X
        N, C, H, W = X.shape
        k = self.kernel_size
        s = self.stride
        p = self.padding
        if p > 0:
            X_padded = np.pad(X, ((0,0),(0,0),(p,p),(p,p)), mode='constant')
        else:
            X_padded = X
        self.X_padded = X_padded
        self.padded_shape = X_padded.shape
        out_h = (X_padded.shape[2] - k) // s + 1
        out_w = (X_padded.shape[3] - k) // s + 1
        self.cols = im2col(X_padded, k, s)
        W_col = self.W.reshape(self.out_channels, -1)
        out = W_col @ self.cols + self.b[:, None]
        out = out.reshape(N, self.out_channels, out_h, out_w)
        return out
    def backward(self, d_out):
        N, C, H, W = self.X.shape
        k = self.kernel_size
        s = self.stride
        p = self.padding
        # Use the output shape from the forward pass (after padding)
        out_h = d_out.shape[2]
        out_w = d_out.shape[3]
        d_out_flat = d_out.reshape(N, self.out_channels, -1)
        self.dW = np.zeros((self.out_channels, self.in_channels * k * k))
        self.db = np.zeros_like(self.b)
        for n in range(N):
            self.dW += d_out_flat[n] @ self.cols[n].T
        self.dW = (self.dW / N).reshape(self.W.shape)
        self.db = np.sum(d_out, axis=(0,2,3)) / N
        W_col = self.W.reshape(self.out_channels, -1)
        dX_col = np.zeros((N, self.in_channels * k * k, out_h * out_w))
        for n in range(N):
            dX_col[n] = W_col.T @ d_out_flat[n]
        # Reconstruct gradient w.r.t. padded input
        dX_padded = np.zeros(self.padded_shape)
        for n in range(N):
            dX_padded[n] = col2im(dX_col[n][None], (1, self.in_channels, self.padded_shape[2], self.padded_shape[3]), k, s)[0]
        # Remove padding
        if p > 0:
            dX = dX_padded[:, :, p:-p, p:-p]
        else:
            dX = dX_padded
        return dX
    def step(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db

class BatchNorm2D:
    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.eps = eps
        self.momentum = momentum
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
    def forward(self, X, train=True):
        self.X = X
        if train:
            self.mean = X.mean(axis=(0,2,3), keepdims=True)
            self.var = X.var(axis=(0,2,3), keepdims=True)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean.squeeze()
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var.squeeze()
        else:
            self.mean = self.running_mean[None,:,None,None]
            self.var = self.running_var[None,:,None,None]
        self.X_norm = (X - self.mean) / np.sqrt(self.var + self.eps)
        out = self.gamma[None,:,None,None] * self.X_norm + self.beta[None,:,None,None]
        return out
    def backward(self, d_out):
        N, C, H, W = self.X.shape
        X_mu = self.X - self.mean
        std_inv = 1. / np.sqrt(self.var + self.eps)
        dX_norm = d_out * self.gamma[None,:,None,None]
        dvar = np.sum(dX_norm * X_mu * -0.5 * std_inv**3, axis=(0,2,3), keepdims=True)
        dmean = np.sum(dX_norm * -std_inv, axis=(0,2,3), keepdims=True) + dvar * np.mean(-2. * X_mu, axis=(0,2,3), keepdims=True)
        dX = dX_norm * std_inv + dvar * 2 * X_mu / (N*H*W) + dmean / (N*H*W)
        self.dgamma = np.sum(d_out * self.X_norm, axis=(0,2,3))
        self.dbeta = np.sum(d_out, axis=(0,2,3))
        return dX
    def step(self, lr):
        self.gamma -= lr * self.dgamma
        self.beta -= lr * self.dbeta

class ReLU:
    def forward(self, X):
        self.mask = X > 0
        return X * self.mask
    def backward(self, d_out):
        return d_out * self.mask
    def step(self, lr):
        pass

class GlobalAvgPool:
    def forward(self, X):
        self.X_shape = X.shape
        return X.mean(axis=(2,3), keepdims=True)
    def backward(self, d_out):
        N, C, _, _ = self.X_shape
        H, W = self.X_shape[2], self.X_shape[3]
        return d_out * np.ones((N, C, H, W)) / (H * W)
    def step(self, lr):
        pass

class Flatten:
    def forward(self, X):
        self.orig_shape = X.shape
        return X.reshape(X.shape[0], -1)
    def backward(self, d_out):
        return d_out.reshape(self.orig_shape)
    def step(self, lr):
        pass

class Linear:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2. / in_features)
        self.b = np.zeros(out_features)
    def forward(self, X):
        self.X = X
        return X @ self.W + self.b
    def backward(self, d_out):
        self.dW = self.X.T @ d_out / self.X.shape[0]
        self.db = np.sum(d_out, axis=0) / self.X.shape[0]
        dX = d_out @ self.W.T
        return dX
    def step(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db
# --- Residual Block ---
class BasicBlock:
    def __init__(self, in_channels, out_channels, stride=1, use_projection=False):
        self.conv1 = Conv2D(in_channels, out_channels, 3, stride, padding=1)
        self.bn1 = BatchNorm2D(out_channels)
        self.relu1 = ReLU()
        self.conv2 = Conv2D(out_channels, out_channels, 3, 1, padding=1)
        self.bn2 = BatchNorm2D(out_channels)
        self.relu2 = ReLU()
        self.use_projection = use_projection
        if use_projection or in_channels != out_channels or stride != 1:
            self.proj = Conv2D(in_channels, out_channels, 1, stride, padding=0)
            self.bn_proj = BatchNorm2D(out_channels)
        else:
            self.proj = None
    def forward(self, X, train=True):
        self.X = X
        out = self.conv1.forward(X)
        out = self.bn1.forward(out, train)
        out = self.relu1.forward(out)
        out = self.conv2.forward(out)
        out = self.bn2.forward(out, train)
        if self.proj is not None:
            shortcut = self.proj.forward(X)
            shortcut = self.bn_proj.forward(shortcut, train)
        else:
            shortcut = X
        self.shortcut = shortcut
        out += shortcut
        out = self.relu2.forward(out)
        self.out = out
        return out
    def backward(self, d_out):
        d_out = self.relu2.backward(d_out)
        d_shortcut = d_out.copy()
        d_main = d_out.copy()
        # Main path
        d_main = self.bn2.backward(d_main)
        d_main = self.conv2.backward(d_main)
        d_main = self.relu1.backward(d_main)
        d_main = self.bn1.backward(d_main)
        d_main = self.conv1.backward(d_main)
        # Shortcut path
        if self.proj is not None:
            d_shortcut = self.bn_proj.backward(d_shortcut)
            d_shortcut = self.proj.backward(d_shortcut)
        # Add gradients from both paths
        dX = d_main + (d_shortcut if self.proj is not None else d_shortcut)
        return dX
    def step(self, lr):
        self.conv1.step(lr)
        self.bn1.step(lr)
        self.conv2.step(lr)
        self.bn2.step(lr)
        self.relu1.step(lr)
        self.relu2.step(lr)
        if self.proj is not None:
            self.proj.step(lr)
            self.bn_proj.step(lr)
# --- ResNet18 Model ---
class ResNet18:
    def __init__(self, num_classes=10):
        # Initial conv (CIFAR-10: 3x3, stride 1, padding=1, no maxpool)
        self.conv1 = Conv2D(3, 64, 3, stride=1, padding=1)
        self.bn1 = BatchNorm2D(64)
        self.relu1 = ReLU()
        # Stages: (channels, num_blocks, stride for first block)
        self.stage1 = [BasicBlock(64, 64, stride=1), BasicBlock(64, 64, stride=1)]
        self.stage2 = [BasicBlock(64, 128, stride=2, use_projection=True), BasicBlock(128, 128, stride=1)]
        self.stage3 = [BasicBlock(128, 256, stride=2, use_projection=True), BasicBlock(256, 256, stride=1)]
        self.stage4 = [BasicBlock(256, 512, stride=2, use_projection=True), BasicBlock(512, 512, stride=1)]
        self.global_pool = GlobalAvgPool()
        self.flatten = Flatten()
        self.fc = Linear(512, num_classes)

    def forward(self, X, train=True):
        out = self.conv1.forward(X)
        out = self.bn1.forward(out, train)
        out = self.relu1.forward(out)
        for block in self.stage1:
            out = block.forward(out, train)
        for block in self.stage2:
            out = block.forward(out, train)
        for block in self.stage3:
            out = block.forward(out, train)
        for block in self.stage4:
            out = block.forward(out, train)
        out = self.global_pool.forward(out)
        out = self.flatten.forward(out)
        out = self.fc.forward(out)
        self.out = out
        return out

    def backward(self, d_out):
        d_out = self.fc.backward(d_out)
        d_out = self.flatten.backward(d_out)
        d_out = self.global_pool.backward(d_out)
        for block in reversed(self.stage4):
            d_out = block.backward(d_out)
        for block in reversed(self.stage3):
            d_out = block.backward(d_out)
        for block in reversed(self.stage2):
            d_out = block.backward(d_out)
        for block in reversed(self.stage1):
            d_out = block.backward(d_out)
        d_out = self.relu1.backward(d_out)
        d_out = self.bn1.backward(d_out)
        d_out = self.conv1.backward(d_out)
        return d_out

    def step(self, lr):
        self.conv1.step(lr)
        self.bn1.step(lr)
        self.relu1.step(lr)
        for block in self.stage1:
            block.step(lr)
        for block in self.stage2:
            block.step(lr)
        for block in self.stage3:
            block.step(lr)
        for block in self.stage4:
            block.step(lr)
        self.global_pool.step(lr)
        self.flatten.step(lr)
        self.fc.step(lr)

# --- Training Utilities ---
def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

def cross_entropy_loss(logits, y):
    N = logits.shape[0]
    probs = softmax(logits)
    log_likelihood = -np.log(probs[np.arange(N), y] + 1e-8)
    loss = np.sum(log_likelihood) / N
    d_logits = probs
    d_logits[np.arange(N), y] -= 1
    d_logits /= N
    return loss, d_logits

def accuracy(logits, y):
    preds = np.argmax(logits, axis=1)
    return np.mean(preds == y)

# --- Main Training Loop (CIFAR-10) ---
if __name__ == "__main__":
    import time
    from sklearn.datasets import fetch_openml
    np.random.seed(42)
    # User-adjustable parameters
    DATASET_SIZE = 2000  # Set to None for full CIFAR-10 (slow!)
    BATCH_SIZE = 20
    EPOCHS = 10  # Increase for better results, but will be slow
    LEARNING_RATE = 0.01

    print("Loading CIFAR-10 from OpenML...")
    cifar10 = fetch_openml('CIFAR_10_small', version=1)
    X = cifar10.data  # shape (60000, 3072)
    y = cifar10.target.astype(int)
    X = X.to_numpy().reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    if DATASET_SIZE is not None:
        X, y = X[:DATASET_SIZE], y[:DATASET_SIZE]
    split = int(0.9 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    print(f"Training ResNet-18 on CIFAR-10 subset: {len(X_train)} train, {len(X_val)} val, {EPOCHS} epochs")
    model = ResNet18(num_classes=10)
    for epoch in range(EPOCHS):
        t0 = time.time()
        idx = np.random.permutation(len(X_train))
        X_train, y_train = X_train[idx], y_train[idx]
        for i in range(0, len(X_train), BATCH_SIZE):
            X_batch = X_train[i:i+BATCH_SIZE]
            y_batch = y_train[i:i+BATCH_SIZE]
            logits = model.forward(X_batch, train=True)
            loss, d_logits = cross_entropy_loss(logits, y_batch)
            model.backward(d_logits)
            model.step(LEARNING_RATE)
            if (i // BATCH_SIZE) % 10 == 0:
                print(f"  Batch {i//BATCH_SIZE+1}/{len(X_train)//BATCH_SIZE}: Loss {loss:.4f}", end='\r')
        logits_val = model.forward(X_val, train=False)
        acc_val = accuracy(logits_val, y_val)
        print(f"\nEpoch {epoch+1}/{EPOCHS}, Loss: {loss:.4f}, Val Accuracy: {acc_val:.4f}, Time: {time.time()-t0:.1f}s")
    # Final evaluation
    logits_val = model.forward(X_val, train=False)
    acc_val = accuracy(logits_val, y_val)
    print(f"\nFinal validation accuracy: {acc_val:.4f}")
