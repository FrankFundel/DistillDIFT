import torch
import torch.nn.functional as F

def softmax_with_temperature(input, temperature=1.0):
    """
    Apply the softmax function with temperature on the input tensor.

    Parameters:
        input (torch.Tensor): The input tensor for which the softmax is to be computed.
        temperature (float, optional): The temperature parameter to adjust the smoothness of the output. Default is 1.0.
    
    Returns:
        torch.Tensor: The softmax probabilities with applied temperature.
    """
    scaled_input = input / temperature # Adjust the input based on the temperature
    return F.softmax(scaled_input, dim=-1) # Compute softmax on the scaled input

def softargmax2d(input, beta=100):
    """
    Apply the soft-argmax function on the input tensor.

    Parameters:
        input (torch.Tensor): The input tensor for which the soft-argmax is to be computed.
        beta (float, optional): The beta parameter to adjust the smoothness of the output. Default is 100.
    """

    *_, h, w = input.shape

    input = input.reshape(*_, h * w)
    input = F.softmax(beta * input, dim=-1)

    indices_c, indices_r = torch.meshgrid(
        torch.linspace(0, 1, w),
        torch.linspace(0, 1, h),
        indexing='xy'
    )

    indices_r = indices_r.reshape(-1, h * w).to(input.device)
    indices_c = indices_c.reshape(-1, h * w).to(input.device)

    result_r = torch.sum((h - 1) * input * indices_r, dim=-1)
    result_c = torch.sum((w - 1) * input * indices_c, dim=-1)

    return torch.stack([result_r, result_c], dim=-1)

def softargmax1d(input, beta=100):
    """
    Apply the soft-argmax function on the input tensor.

    Parameters:
        input (torch.Tensor): The input tensor for which the soft-argmax is to be computed.
        beta (float, optional): The beta parameter to adjust the smoothness of the output. Default is 100.
    """
    *_, n = input.shape
    input = F.softmax(beta * input, dim=-1)
    indices = torch.linspace(0, 1, n, device=input.device)
    return torch.sum((n - 1) * input * indices, dim=-1)
