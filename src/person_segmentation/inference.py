import torch
from torch import nn


def collect_logits(
    model: nn.Module, loader, device: torch.device, tta: bool = False
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    model.eval()
    logits, masks, names = [], [], []
    with torch.no_grad():
        for images, targets, batch_names in loader:
            inputs = images.to(device)
            output = model(inputs)
            if tta:
                output = (output + model(inputs.flip(-1)).flip(-1)) / 2.0
            logits.append(output.cpu())
            masks.append(targets)
            names.extend(batch_names)
    return torch.cat(logits), torch.cat(masks), names
