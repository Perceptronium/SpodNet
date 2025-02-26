"""Main class of our algorithmic framework.
This performs a forward pass across layers that are specified in its init. """

import torch
from torch import nn

from spodnet.layers import UpdateTheta


class SpodNet(nn.Module):
    """ Main class : unrolled algorithm. """

    def __init__(self,
                 K,
                 p,
                 layer_type,
                 device):
        super().__init__()

        self.K = K
        self.p = p

        self.layer_type = layer_type

        self.device = device

        if layer_type == 'UBG':
            print("Learning in UBG mode.")
            self.forward_stack = UpdateTheta(
                self.p, theta_12_generator='UBG', device=device)

        elif layer_type == 'PNP':
            print("Learning in PNP mode.")
            self.forward_stack = UpdateTheta(
                self.p, theta_12_generator='PNP', device=device)

        if layer_type == 'E2E':
            print("Learning in E2E mode.")
            self.forward_stack = UpdateTheta(
                self.p, theta_12_generator='E2E', device=device)

    def forward(self, S):
        """ Forward pass. """

        Theta = torch.linalg.pinv(
            S + torch.eye(S.shape[-1]).expand_as(S).type_as(S), hermitian=True)
        W = torch.linalg.pinv(Theta, hermitian=True)

        self.forward_stack.W = W.clone().detach()
        self.forward_stack.S = S.clone().detach()

        for _ in range(self.K):
            Theta = self.forward_stack(Theta)

        return Theta
