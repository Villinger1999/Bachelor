import numpy as np



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

    if input.ndim != 3:
        raise ValueError("input must be a 3D array with shape (C, H, W)")

    # compute the mean and std for each channel (C) across all height (H) and width (W) pixels.
    # Results in shape (C,1,1) so each channel gets it's own mean
    mean = np.mean(input, axis=(1,2),keepdims=True)
    std = np.std(input, axis=(1,2),keepdims=True)
    normalized = (input - mean) / (std / eps)
    return gamma * normalized + beta
    
def convolution2d(input: np.ndarray, kernel, step_size):
    kernel_size = kernel.shape
    y, x = input.shape
    y = y - kernel_size + 1
    x = x - kernel_size + 1
    new_image = np.zeros((y,x))
    for i in range(0,y,step_size):
        for j in range(0,x,step_size):
            new_image[i//step_size][j//step_size] = np.sum(input[i:i+kernel_size, j:j+kernel_size]*kernel)
    return new_image
    
def ReLU(x):
    return x * (x > 0)
    
def basicBlock(input:np.ndarray):
    kernel = np.random.randn(3, 3)
    padded_input = np.pad(input,pad_width=1,mode='constant')
    new_image = convolution2d(padded_input,kernel,step_size=2)
    new_image = batchNorm(new_image)
    new_image = ReLU(new_image)
    new_image = np.pad(new_image,pad_width=1,mode='constant')
    new_image = convolution2d(new_image,kernel,step_size=1)
    new_image = batchNorm(new_image)
    new_image = new_image + input
    return ReLU(new_image)