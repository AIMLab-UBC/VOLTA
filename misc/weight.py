import torch.nn


def weight_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
        torch.nn.init.xavier_normal_(m.weight)
    elif isinstance(m, torch.nn.LayerNorm):
        torch.nn.init.constant_(m.bias, 0)
        torch.nn.init.constant_(m.weight, 1.0)
