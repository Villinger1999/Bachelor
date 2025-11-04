import numpy as np
import sys as sys
import cv2
import skimage.measure as measure

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
    
def convolution3d(input: np.ndarray, kernel, step_size) -> np.ndarray:
    kernel_size = kernel.shape[0] #since we use a square kernel
    # batchsize, channels, height, width
    n, c, y, x = input.shape 
    y_out = (y - kernel_size) // step_size + 1
    x_out = (x - kernel_size) // step_size + 1
    new_image = np.zeros((c, y_out, x_out))
    for chan in range(c):
        for i in range(0, y - kernel_size + 1, step_size):
            for j in range(0, x - kernel_size + 1, step_size):
                new_image[chan, i // step_size, j // step_size] = np.sum(input[chan, i:i+kernel_size, j:j+kernel_size] * kernel)
    return new_image
    
def ReLU(x):
    return x * (x > 0)
    
def basicBlock(input:np.ndarray, kernel_1:np.ndarray, kernel_2:np.ndarray, stride1:int = 2) -> np.ndarray:
    """This is a simplified NumPy implementation of a ResNet basic block. 

    Args:
        input (np.ndarray): the image used in the convolution
        kernel_size (int, optional): defines the size of a square kernel
        stride1 (int, optional): defines the stride/step size during the first convolution. Defaults to 2.

    Returns:
        np.ndarray: return an image
    """
    # Create the kernel used in the first convolution.
    # Adds zero padding to the input
    padded_input = np.pad(input,pad_width=1,mode='constant')
    # First convolution on padded input
    new_image = convolution3d(padded_input,kernel_1,step_size=stride1)
    # Batch normalization
    new_image = batchNorm(new_image)
    # ReLU activation function
    new_image = ReLU(new_image)
    #create the kernel used in the second convolution.
    # Adds zero padding to the input
    new_image = np.pad(new_image,pad_width=1,mode='constant')
    # econd convolution after activation and padding
    new_image = convolution3d(new_image,kernel_2,step_size=1)
    # Batchnormalization
    new_image = batchNorm(new_image)
    # Downsamples the original input to match the output shape for residual addition
    resized_input = convolution3d(input, kernel=np.ones((1,1)),step_size=stride1)
    # Adds the residual (downsampled input) to the output of the convolutions 
    new_image = new_image + resized_input
    # returning the image after having used a ReLU activation function
    return ReLU(new_image)



if __name__ == "main":
    image_path = "data/imagenetSub"
    image_array = []
    #resnet18
    # for image in image_path:
    #     im = cv2.imread("abc.tiff",mode='RGB')
    
    kernel7x7 = np.random.randn(7, 7)
    inp = np.array([])
    padded_input = np.pad(inp,pad_width=3,mode='constant')
    firstconv = convolution3d(padded_input,kernel7x7,step_size=2)
    measure.block_reduce(inp,2,np.max)
    
    