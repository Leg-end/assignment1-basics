import torch


def cross_entropy_loss(inputs: torch.Tensor, targets: torch.LongTensor) -> torch.Tensor:
    x_max = torch.max(inputs, dim=-1, keepdim=True).values
    x_stable = inputs - x_max
    target_logits = x_stable.gather(1, index=targets.unsqueeze(-1))
    logsumexp = torch.logsumexp(x_stable, -1, keepdim=True)
    loss = torch.mean(-target_logits + logsumexp)
    return loss  # perplexity = exp(loss)
