import torch


def sample(logits: torch.Tensor, temperatures: torch.Tensor, top_ps: torch.Tensor) -> torch.Tensor:
    """logits [B, V] -> token ids [B]. Rows with temperature 0 are greedy."""
    greedy = logits.argmax(dim=-1)
    if bool((temperatures == 0).all()):
        return greedy
    probs = torch.softmax(logits.float() / temperatures.clamp(min=1e-5)[:, None], dim=-1)
    # Top-p: keep the smallest set of tokens whose probability mass reaches top_p.
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    mass_before = sorted_probs.cumsum(dim=-1) - sorted_probs
    sorted_probs[mass_before > top_ps[:, None]] = 0.0
    choice = torch.multinomial(sorted_probs, 1).squeeze(1)
    sampled = sorted_idx.gather(1, choice[:, None]).squeeze(1)
    return torch.where(temperatures == 0, greedy, sampled)
