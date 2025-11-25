def plgp_gradients(grads, threshold: float, alpha: float):
    """
    Proportional Large Gradient Pruning (PLGP):
    Scales down gradients with magnitude > threshold by factor alpha.

    Args:
        model (nn.Module): model with gradients
        threshold (float): threshold for cutoff
        alpha (float): scaling factor for large gradients (0 < alpha < 1)
    """
    pruned_grads = []
    for grad in grads:      
        if grad.abs() > threshold:
            pruned_grads.append(alpha * grad)  # scale down large grads
        else:
            pruned_grads.append(grad)
    return pruned_grads


def sgp_gradients(grads, threshold: float):
    """
    Small Gradient Pruning (sgp):
    Set gradients with magnitude <= threshold to 0.

    Args:
        model (nn.Module): model with gradients
        threshold (float): threshold for cutoff
    """   
    pruned_grads = []
    for grad in grads:      
        if grad.abs() <= threshold:
            pruned_grads.append(0.0)  # scale down large grads
        else:
            pruned_grads.append(grad)
    return pruned_grads