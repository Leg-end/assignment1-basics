import torch
from collections.abc import Iterable

def register_activation_hooks(model):
    """
    注册前向钩子来捕获各层的激活值
    """
    activation_norms = {}
    hooks = []
    
    def hook_fn(name):
        def hook(module, input, output):
            # 计算输出的L2范数
            if isinstance(output, torch.Tensor):
                activation_norms[name] = torch.norm(output, p=2).item()
            elif isinstance(output, tuple):
                # 对于多输出的层，计算所有输出的范数
                total_norm = 0
                for out in output:
                    if isinstance(out, torch.Tensor):
                        total_norm += torch.norm(out, p=2).item() ** 2
                activation_norms[name] = total_norm ** 0.5
        return hook
    
    # 为所有层注册钩子
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 只注册叶子模块
            hook = module.register_forward_hook(hook_fn(name))
            hooks.append(hook)
    
    return activation_norms, hooks

def clip_grad_norm(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> torch.Tensor:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    parameters = [p for p in parameters if p.grad is not None]
    if len(parameters) == 0:
        return
    # Calculate total L2 norm of all gradients, take all gradients as a single vector
    # and compute its L2 norm by summing the squares of all gradients and then taking the square root.
    total_norm = torch.sqrt(sum(torch.sum(p.grad.pow(2)) for p in parameters))
    
    # Calculate clipping coefficient
    clip_coef = max_l2_norm / (total_norm + 1e-6)  # Add small value to avoid division by zero
    
    # If total norm exceeds max_norm, scale down all gradients
    if clip_coef < 1.0:
        for p in parameters:
            p.grad.mul_(clip_coef)
    return total_norm


def compute_weights_norms(model: torch.nn.Module,
                          norm_type: float | str | None = 2) -> tuple[dict[str, float], float]:
    """
    计算模型中所有权重参数的范数
    
    Args:
        model: PyTorch模型
        norm_type: 范数类型 (1: L1, 2: L2, 'fro': Frobenius范数)
    
    Returns:
        norms_dict: 各层权重范数字典
        total_norm: 所有权重参数的总范数
    """
    norms_dict = {}
    total_norm = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad and 'weight' in name:
            norm = torch.norm(param, p=norm_type)
            
            norms_dict[name] = norm.item()
            total_norm += norm.item() ** 2
    
    total_norm = total_norm ** 0.5
    return norms_dict, total_norm

    