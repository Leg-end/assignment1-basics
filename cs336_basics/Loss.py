import torch


def cross_entropy_loss(inputs: torch.Tensor, targets: torch.LongTensor) -> torch.Tensor:
    target_logits = inputs.gather(1, index=targets.unsqueeze(-1))
    logsumexp = torch.logsumexp(inputs, -1, keepdim=True)
    loss = torch.mean(-target_logits + logsumexp)
    return loss