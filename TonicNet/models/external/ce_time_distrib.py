import torch
import torch.nn as nn

class CrossEntropyTimeDistributedLoss(nn.Module):
    """loss function for multi-timsetep model output"""

    def __init__(self):
        super(CrossEntropyTimeDistributedLoss, self).__init__()

        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, y_hat, y):
        _y_hat = y_hat.squeeze(0)
        _y = y.squeeze(0)

        # Loss from one sequence
        loss = self.loss_func(_y_hat, _y)
        loss = torch.sum(loss)
        return loss
    