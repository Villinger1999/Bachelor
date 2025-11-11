from attack_function.iDLG import iDLG, infer_labels_from_bias_grad
from models.central_model import get_model
import torch
from test_train_noiseadd import x_train
from torchvision.utils import save_image
import numpy as np

model = get_model()

leaked_grads = torch.load("state_dicts/local_grads_client15c_10r_batch2.pt", map_location=torch.device('cpu'), weights_only=False)

infered_label = infer_labels_from_bias_grad(leaked_grads, model)

x_shape = (2,3,32,32)

image_tensor = iDLG(model, leaked_grads, infered_label, x_shape)

save_image(image_tensor[0], "reconstructions/Infered_image_client15c_10r_batch2.png")
save_image(image_tensor[1], "reconstructions/Infered_image_client15c_10r_batch2_2.png")
print(infered_label)