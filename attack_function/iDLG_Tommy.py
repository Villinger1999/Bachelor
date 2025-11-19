from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
from torch.optim.lbfgs import LBFGS
import sys

class iDLG:
    def __init__(
        self,
        model,
        orig_img,
        label,
        device,
        grads,
        *,
        seed: int,
        clamp: tuple[float, float]
    ) -> None:
        # Respect provided device and keep original dtype of the model/weights
        self.device = device if isinstance(device, str) else (device.type if hasattr(device, "type") else "cpu")
        self.model = model.to(self.device)
        self.orig_img = orig_img.to(self.device)
        self.criterion = nn.CrossEntropyLoss(reduction='sum').to(self.device)
        self.label = label.to(self.device)
        self.grads = grads
        self.tt = transforms.ToPILImage()
        self.clamp = clamp

        # Align image dtype to model parameter dtype (usually float32)
        self.param_dtype = next(self.model.parameters()).dtype
        if self.orig_img.dtype != self.param_dtype:
            self.orig_img = self.orig_img.to(self.param_dtype)

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def attack(self, iterations=200):
        # iDLG training image reconstruction:
        self.model.eval()

        # compute original gradients
        orig_grads = self.grads
        
        # B = 2
        # dummy_data = torch.randn(
        #     (B, *self.orig_img.shape[1:]),
        #     dtype=self.param_dtype, device=self.device, requires_grad=True
        # )

        # initialize dummy in the correct iteration, respecting the random seed
        dummy_data = (torch.randn(self.orig_img.size(), dtype=self.param_dtype, device=self.device).requires_grad_(True))

        # init with ground truth:
        label_pred = torch.argmin(torch.sum(orig_grads[-2], dim=-1), dim=-1).detach().reshape((1,)).requires_grad_(False)
        # label_single = torch.argmin(torch.sum(orig_grads[-2], dim=-1), dim=-1).detach()
        # label_pred = label_single.repeat(B).requires_grad_(False)  
        
        # this is number 9 with pretrained before FL
        # optimizer = LBFGS( 
        #     [dummy_data], lr=.5, max_iter=50,
        #     tolerance_grad=1e-09, tolerance_change=1e-11,
        #     history_size=100, line_search_fn='strong_wolfe'
        # )
        
        # This is number 8 with pretrained before FL
        # This is number 11 with pretrained after FL
        # this is 17 with pretrained after FL and 1 round / 1 epoch (-1 clamp)
        # this is 20 with pretrained after grads in FL and 1 round / 1 epoch
        # this is 24 no pretrained model - but idlg model updated from FL after grads and 1 round / 1 epoch (-1 clamp)
        # optimizer = LBFGS( 
        #     [dummy_data],
        #     lr=0.4,
        #     max_iter=200,
        #     tolerance_grad=1e-12,
        #     tolerance_change=1e-8,
        #     history_size=100,
        #     line_search_fn='strong_wolfe'
        # )
        
        # this is 21 with pretrained after FL and 1 round / 1 epoch
        # this is 22 no pretrained model - but idlg model updated from FL after grads and 1 round / 1 epoch
        # this is 23 with pretrained after grads and 1 round / 1 epoch
        # optimizer = LBFGS(
        #     [dummy_data],
        #     lr=.2,
        #     max_iter=200,
        #     tolerance_grad=1e-12,
        #     tolerance_change=1e-11,
        #     history_size=100,
        #     line_search_fn='strong_wolfe'
        # )
        
        # this is 30 no pretrained model - but idlg model updated from FL after grads and 1 round / 1 epoch
        # optimizer = LBFGS(
        #     [dummy_data],
        #     lr=.2,
        #     max_iter=200,
        #     tolerance_grad=1e-10,
        #     tolerance_change=1e-10,
        #     history_size=100,
        #     line_search_fn='strong_wolfe'
        # )
        
        
        dummy_data = dummy_data * 0.1 + 0.5   # or just *0.1 if you want

        # detach to make it a new leaf, then enable grads
        dummy_data = dummy_data.detach().requires_grad_(True)
        optimizer = torch.optim.Adam([dummy_data], lr=0.1)
        # optimizer = torch.optim.SGD([dummy_data], lr=0.01, momentum=0.9)

        # this is number 10 with pretrained before FL
        # this is number 12 with pretrained after FL
        # this is 18 with pretrained after FL and 1 round / 1 epoch (-1 clamp)
        # this is 19 with pretrained after FL and 1 round / 1 epoch
        # this is 25 no pretrained model - but idlg model updated from FL after grads and 1 round / 1 epoch (-1 clamp)
        # this is 26 no pretrained model - but idlg model updated from FL after grads and 1 round / 1 epoch 
        # optimizer = LBFGS(
        #     [dummy_data],
        #     lr=.1,
        #     max_iter=50,
        #     tolerance_grad=1e-09,
        #     tolerance_change=1e-11,
        #     history_size=100,
        #     line_search_fn='strong_wolfe'
        # )
        
        # this is 28 no pretrained model - but idlg model updated from FL after grads and 1 round / 1 epoch 
        # this is 29 no pretrained model - but idlg model updated from FL after grads and 1 round / 1 epoch 
        # optimizer = LBFGS(
        #     [dummy_data],
        #     lr=.01,
        #     max_iter=50,
        #     tolerance_grad=1e-09,
        #     tolerance_change=1e-11,
        #     history_size=100,
        #     line_search_fn='strong_wolfe'
        # )
        
        # this is 27 no pretrained model - but idlg model updated from FL after grads and 1 round / 1 epoch 
        # optimizer = LBFGS(
        #     [dummy_data],
        #     lr=.1,
        #     max_iter=200,
        #     tolerance_grad=1e-09,
        #     tolerance_change=1e-11,
        #     history_size=100,
        #     line_search_fn='strong_wolfe'
        # )
        
        # this is 13 with pretrained before FL
        # optimizer = LBFGS( 
        #     [dummy_data],
        #     lr=0.6,
        #     max_iter=200,
        #     tolerance_grad=1e-9,
        #     tolerance_change=1e-11,
        #     history_size=100,
        #     line_search_fn='strong_wolfe'
        # )
        
        # this is 14 with pretrained after FL
        # this is 15 with pretrained after FL and 1 round / 1 epoch
        # this is 16 with pretrained after FL and 1 round / 1 epoch (-1 clamp)
        # optimizer = LBFGS( 
        #     [dummy_data],
        #     lr=0.2,
        #     max_iter=50,
        #     tolerance_grad=1e-7,
        #     tolerance_change=1e-9,
        #     history_size=100,
        #     line_search_fn='strong_wolfe'
        # )

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
                history.append(self.tt(dummy_data[0].detach().cpu()))
                
        return dummy_data.detach().numpy().squeeze(), label_pred, history, losses