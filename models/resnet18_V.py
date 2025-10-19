import numpy as np


def batchNorm(input: np.ndarray, step_size:int=1, padding_size:int = 1, kernel_size:int=3):
    """Batch normalization

    Args:
        input (np.ndarray): image/tensor of size (C,H,W)
        step_size (int, optional): stepsize for the kernel. Defaults to 1.
        padding_size (int, optional): adds zero-padding to the image. Defaults to 1.
        kernel_size (int, optional): kernel size k, means the size of the kernel will be k x k. Defaults to 3.
    Returns:
        np.ndarray: batch-normalized tensor of same shape as input
    """
    # ensure input is a numpy array with shape (C,H,W)
    if not isinstance(input, np.ndarray):
        input = np.array(input)

    if input.ndim != 3:
        raise ValueError("input must be a 3D array with shape (C, H, W)")
    
    padded_input = np.pad(input,pad_width=padding_size,mode='constant')
    
    for c in input[0]:
        for w in input[2]:
            for h in input[1]:
                h=0
            
    return
    
def basicBlock(input):
    
    
    
    return