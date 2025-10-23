import torch

def cross_entropy_loss(inputs: torch.Tensor,
                       targets: torch.LongTensor,
                       reduction: str = 'mean') -> torch.Tensor:
    x_max = torch.max(inputs, dim=-1, keepdim=True).values
    x_stable = inputs - x_max
    target_logits = x_stable.gather(1, index=targets.unsqueeze(-1))
    logsumexp = torch.logsumexp(x_stable, -1, keepdim=True)
    loss = -target_logits + logsumexp
    if reduction =='mean':
        loss = torch.mean(loss)
    elif reduction =='sum':
        loss = torch.sum(loss)
    return loss
