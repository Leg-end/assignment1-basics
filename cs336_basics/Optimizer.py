import torch
from torch.optim import Optimizer
from torch import nn
from typing import Callable


class AdamW(Optimizer):
    """
    AdamW is an advanced Adam Optimizer combined with L2 normalization (weight decay),
    proposed by Ilya Loshchilov and Frank Hutter in paper `Decoupled Weight Decay Regularization`.
    It aims at decopling weight decay from adaptive learning rate, which is a thorny issue in Adam.
    
    Adaptive learning rate mechanism of Adam
        first order moment: use EMA of gradient(momentum) to get steady update direction
            m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_t
        second order moment: use EMA of square of gradient(approximate to variance of gradient) to 
        locate learning-saturated parameters:
            v_t = beta_2 * v_{t-1} + (1 - beta_2) * {g_t}^2
        rectify statistical bias
            m^_t = m_t / (1 - {beta_1}^t)
            v^_t = v_t / (1 - {beta_2}^t)
        update rules:
            theta_{t+1} = theta_t - eta * m^t / (sqrt(v^_t) + epsilon)
            larger v_t, more volatile parameter, larger learning rate
            smaller v_t, more saturated parameter, smaller learning rate
    Issue of Adam: weight decay coupled with EMA
        g_{L2_wd} = g_t + lambda * theta_t
    
    Solution of AdamW
        decouple weight decay from EMA, directly applied in update rules:
            theta_{t+1} = theta_t - eta * ( m^t / (sqrt(v^_t) + epsilon) + lambda * theta_t )

    Relation to other optimizers:
        momentum = SGD + exponential averaging of grad
        AdaGrad = SGD + averaging by grad^2
        RMSProp = AdaGrad + exponentially averaging of grad^2
        Adam = RMSProp + momentum
    """
    def __init__(self,
                 params: nn.Parameter,
                 lr: float = 1e-3,
                 betas: tuple = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 1e-2):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)
        
    def step(self, closure: Callable | None = None) -> float:
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Perform stepweight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])
                
                # Get parameter-specific state
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = torch.zeros(())
                    state['exp_avg'] = torch.zeros_like(p.data)  # EMA of gradient
                    state['exp_avg_sq'] = torch.zeros_like(p.data)  # EMA of squared gradient
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                step_size = group['lr']
                eps = group['eps']
                
                # Update state
                state['step'] += 1
                grad = p.grad.data
                
                # Decay the first and second moment runing average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = exp_avg_sq.sqrt().add_(eps)
                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = step_size * (bias_correction2 ** 0.5) / bias_correction1
                
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                
        return loss