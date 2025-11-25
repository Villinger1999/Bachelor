def clipping(grads, threshold: float):
    """
    clipping:
    Set gradients with magnitude >= threshold to threshold.

    Args:
        model (nn.Module): model with gradients
        threshold (float): threshold for cutoff
    """
    pruned_grads = []
    for grad in grads:      
        if grad.abs() >= threshold:
            pruned_grads.append(threshold)  # scale down large grads
        else:
            pruned_grads.append(grad)
    return pruned_grads