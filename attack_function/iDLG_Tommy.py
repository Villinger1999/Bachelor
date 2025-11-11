from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
from torch.optim.lbfgs import LBFGS

class iDLG:
    def __init__(
        self,
        model,
        label,
        device,
        *,
        seed: int | None = None,
        clamp: tuple[float, float] | None = (0.0, 1.0),
    ) -> None:
        # Respect provided device and keep original dtype of the model/weights
        self.device = device if isinstance(device, str) else (device.type if hasattr(device, "type") else "cpu")
        self.model = model.to(self.device)
        # self.orig_img = orig_img.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.label = label.to(self.device)
        self.tt = transforms.ToPILImage()
        self.clamp = clamp

        # Align image dtype to model parameter dtype (usually float32)
        self.param_dtype = next(self.model.parameters()).dtype
        # if self.orig_img.dtype != self.param_dtype:
        #     self.orig_img = self.orig_img.to(self.param_dtype)

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def attack(self, iterations=200):
        # iDLG training image reconstruction:
        self.model.eval()

        # compute original gradients
        predicted = self.model(self.orig_img)
        loss = self.criterion(predicted, self.label)
        orig_grads = torch.autograd.grad(loss, self.model.parameters())
        orig_grads = list((_.detach().clone() for _ in orig_grads))

        # initialize dummy in the correct iteration, respecting the random seed
        dummy_data = (torch.randn(self.orig_img.size(), dtype=self.param_dtype, device=self.device).requires_grad_(True))

        # init with ground truth:
        label_pred = torch.argmin(torch.sum(orig_grads[-2], dim=-1), dim=-1).detach().reshape((1,)).requires_grad_(False)
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

        return dummy_data.detach().numpy().squeeze(), label_pred, history, losses