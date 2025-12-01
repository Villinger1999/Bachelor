from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.optim.lbfgs import LBFGS
from PIL import Image
from numpy import asarray
import numpy as np

class iDLG:
    def __init__(
        self,
        model,
        orig_img,
        label,
        device,
        *,
        seed: int | None = None,
        clamp: tuple[float, float] | None = (0.0, 1.0),
    ) -> None:
        # Respect provided device and keep original dtype of the model/weights
        self.device = device if isinstance(device, str) else (device.type if hasattr(device, "type") else "cpu")
        self.model = model.to(self.device)
        self.orig_img = orig_img.to(self.device)
        self.criterion = nn.CrossEntropyLoss(reduction='sum').to(self.device)
        self.label = label.to(self.device)
        self.tt = transforms.ToPILImage()
        self.clamp = clamp

        # Align image dtype to model parameter dtype (usually float32)
        self.param_dtype = next(self.model.parameters()).dtype
        if self.orig_img.dtype != self.param_dtype:
            self.orig_img = self.orig_img.to(self.param_dtype)

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def _infer_label_from_grads(self, orig_grads):
        # Map grads to names
        named_grads = {name: g for (name, _), g in zip(self.model.named_parameters(), orig_grads)}
        last_bias_name = None
        for name, param in self.model.named_parameters():
            if name.endswith(".bias") and param.ndim == 1:
                last_bias_name = name  # keep overwriting → last bias

        bias_grad = named_grads[last_bias_name]
        return torch.argmin(bias_grad).detach().reshape((1,))

    def attack(self, iterations=200):
        # iDLG training image reconstruction:
        self.model.eval()
        
        # compute original gradients
        predicted = self.model(self.orig_img)
        loss = self.criterion(predicted, self.label)
        orig_grads = torch.autograd.grad(loss, self.model.parameters())
        orig_grads = list((_.detach().clone() for _ in orig_grads))

        # initialize dummy in the correct iteration, respecting the random seed
        # dummy_data = (torch.randn(self.orig_img.size(), dtype=self.param_dtype, device=self.device).requires_grad_(True))
        
        # initialize dummy as the original image
        # dummy_data = torch.as_tensor(self.orig_img)
        
        # initialize dummy as the original image with noise
        dummy_data = asarray(self.orig_img)
        dummy_shape = dummy_data.shape
        noise = np.random.standard_normal(dummy_shape)
        dummy_data = dummy_data + noise
        dummy_data = torch.as_tensor(dummy_data, dtype=self.param_dtype, device=self.device)
        dummy_noisy = torch.as_tensor(dummy_data, dtype=self.param_dtype, device=self.device)
        

        # init with ground truth:
        label_pred = self._infer_label_from_grads(orig_grads).requires_grad_(False)
        optimizer = LBFGS(
            [dummy_data], lr=.1, max_iter=50,
            tolerance_grad=1e-09, tolerance_change=1e-11,
            history_size=100, line_search_fn='strong_wolfe'
        )

        history = []
        losses = []

        for iters in tqdm(range(iterations)):
            def closure():
                optimizer.zero_grad()
                dummy_pred = self.model(dummy_data)
                dummy_loss = self.criterion(dummy_pred, label_pred)
                dummy_dy_dx = torch.autograd.grad(dummy_loss, self.model.parameters(), create_graph=True)
                grad_diff = 0
                for gx, gy in zip(dummy_dy_dx, orig_grads):
                    grad_diff += ((gx - gy) ** 2).sum()
                grad_diff.backward()
                return grad_diff

            optimizer.step(closure)

            # Optional: keep dummy within valid input range
            if self.clamp is not None:
                with torch.no_grad():
                    dummy_data.clamp_(self.clamp[0], self.clamp[1])

            if iters % 1 == 0:
                current_loss = closure()
                losses.append(current_loss.item())
                history.append(self.tt(dummy_data[0].cpu()))

        return dummy_data.detach().numpy().squeeze(), label_pred, history, losses, dummy_noisy
    
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from os import getcwd
    from os.path import join
    def infer_labels_from_bias_grad(leaked_grads:dict[str, torch.Tensor], model: torch.nn.Module) -> int:
        """
        iDLG uses the fact that when cross-entropy is used with softmax, the correct label will always have a negative sign.
        iDLG: label = argmin(grad w.r.t. last-layer bias) because that bias grad equals (p - one_hot).
        
        Note: This method can only infer a SINGLE label, even for batches. The assumption is that all images 
        in the batch belong to the same class (homogeneous batch).

        Args:
            leaked_grads (dict[str, torch.Tensor]): the leaked gradients
            model (torch.nn.Module): the model used

        Returns:
            int: index of the inferred label (same for all images in batch)
        """
        
        if isinstance(leaked_grads, dict) and 'grads' in leaked_grads:
            leaked_grads = leaked_grads['grads']
            
            # find the name of the last bias
            for name, parameter in model.named_parameters(): #loop through the names and parameters in the model 
                if name.endswith(".bias") and parameter.ndim == 1: # if it's a bias parameter and it's 1 dimentional. 
                    last_bias_name = name # set is as the name of the last bias term
            
            # bias_grad is the Gradient of loss w.r.t. logits (g_i in equation 3 in iDLG paper)
            bias_grad = leaked_grads[last_bias_name] # Bias gradient equals g_i
            true_label = int(torch.argmin(bias_grad).item())
            
            return true_label # True label = index of minimum gradient
        
    
    class LeNet(nn.Module):
        def __init__(self, channel: int = 3, hidden: int = 768, num_classes: int = 10):
            super(LeNet, self).__init__()
            act = nn.Sigmoid
            self.body = nn.Sequential(
                nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
                act(),
                nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
                act(),
                nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
                act(),
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden, num_classes)
            )

        def forward(self, x):
            out = self.body(x)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            return out
    class LeNet5(nn.Module):
        def __init__(self,channel:int = 3, hidden:int = 400, num_classes:int = 10):
            super(LeNet5, self).__init__()
            
            self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            # nn.BatchNorm2d(6),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
            
            self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            # nn.BatchNorm2d(16),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
            
            # self.layer3 = nn.Sequential(
            # nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0),
            # nn.ReLU()
            # )
            
            self.fc = nn.Linear(hidden, 120)
            self.relu = nn.ReLU()
            self.fc1 = nn.Linear(120, 84)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(84, num_classes)

        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            # out = self.layer3(out)
            out = out.reshape(out.size(0), -1)
            out = self.fc(out)
            out = self.relu(out)
            out = self.fc1(out)
            out = self.relu1(out)
            out = self.fc2(out)
            return out

    # Define relevant variables for the ML task
    batch_size = 1000
    num_classes = 10
    learning_rate = 0.01
    num_epochs = 10
    modeltype = LeNet
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = getcwd()

    transform = transforms.Compose([transforms.ToTensor(),])
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    trainloader = DataLoader(trainset,batch_size,shuffle=True)
    testloader = DataLoader(testset,batch_size,shuffle=True)
    
    model = modeltype().to(device)
    cost = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_step = len(trainset)
    
    save_dir = join(path, "reconstructions")
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(trainloader):  
            images = images.to(device)
            labels = labels.to(device)
                
            #Forward pass
            outputs = model(images)
            loss = cost(outputs, labels)
            #Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 400 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            
        if epoch == (num_epochs-1):
                torch.save(model.state_dict(), "grads_LeNet.pt")
          
    attaloader = DataLoader(trainset,1,True)
    
    image, label = next(iter(attaloader))
    img_tensor = image[0] if image.dim() == 4 else image  # remove batch dimension if present
    to_pil = transforms.ToPILImage()
    pil_img = to_pil(img_tensor.cpu())
    pil_img.save(join(save_dir, "original_image.png"))
    pil_img.show()
    
    img, label_pred, history, losses, noisy  = iDLG(model, image, label, device).attack(iterations=200)

    pil_img.save(join(save_dir, "noisy_image.png"))
    pil_img.show()
    model.eval()   
    output = model(image)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, label)
    loss.backward()  # Compute gradients

    # Build the dict of gradients
    leaked_grads = {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None}

    # Now you can call:
    inferred_label = infer_labels_from_bias_grad(leaked_grads, model)
    
    
    arr = img.copy()
    arr = (arr * 255).clip(0, 255).astype('uint8')
    
    if arr.ndim == 3 and arr.shape[0] in [1, 3]:
        arr = arr.transpose(1, 2, 0)
    if arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
        
    Image.fromarray(arr).save(join(save_dir, "reconstrction.jpg"))
    # Or, for the last image in history:
    history[-1].save(join(save_dir, "final_reconstruction.png"))
    
    print("True label:", label.item())
    print("Inferred label_pred:", label_pred.item())
    print("First loss:", losses[0])
    print("Last loss:", losses[-1])