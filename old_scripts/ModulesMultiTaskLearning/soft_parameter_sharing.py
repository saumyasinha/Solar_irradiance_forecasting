from typing import Tuple

import torch
from torch import nn

"""Soft parameter sharing via l_2 regularization
"""


class FFNN(nn.Module):
    """Simple FF network with multiple outputs.
    """

    def __init__(
            self,
            input_size,
            hidden_size,
            n_hidden,
            n_outputs,
            dropout_rate=.5,
    ):
        """
        :param input_size: input size
        :param hidden_size: common hidden size for all layers
        :param n_hidden: number of hidden layers
        :param n_outputs: number of outputs
        :param dropout_rate: dropout rate
        """
        super().__init__()
        assert 0 <= dropout_rate < 1
        self.input_size = input_size

        h_sizes = [self.input_size] + [hidden_size for _ in range(n_hidden)] + [n_outputs]

        self.hidden = nn.ModuleList()
        for k in range(len(h_sizes) - 1):
            self.hidden.append(
                nn.Linear(
                    h_sizes[k],
                    h_sizes[k + 1]
                )
            )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):

        for layer in self.hidden[:-1]:
            x = layer(x)
            x = self.relu(x)
            x = self.dropout(x)

        return self.hidden[-1](x)


class TaskIndependentNets(nn.Module):
    """Independent FFNN for each task
    """

    def __init__(
            self,
            input_size,
            hidden_size,
            n_hidden,
            n_outputs,
            dropout_rate=.5,
    ):
        super().__init__()

        self.n_outputs = n_outputs
        self.task_nets = nn.ModuleList()
        for _ in range(n_outputs):
            self.task_nets.append(
                FFNN(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    n_hidden=n_hidden,
                    n_outputs=1,
                    dropout_rate=dropout_rate,
                )
            )

    def forward(self, x):
        return torch.cat(
            tuple(task_model(x) for task_model in self.task_nets),
            dim=1
        )


class SoftSharing(nn.Module):
    """FFNN with soft parameter sharing via `l_2` regularization
    """

    def __init__(
            self,
            input_size,
            hidden_size,
            n_hidden,
            n_outputs = 2,
            dropout_rate=.5,
    ):

        super().__init__()

        self.model = TaskIndependentNets(
            input_size=input_size,
            hidden_size=hidden_size,
            n_hidden=n_hidden,
            n_outputs=n_outputs,  # we assume we have two tasks. For more tasks,
            # we must change the soft penalty structure
            dropout_rate=dropout_rate
        )

    def get_param_groups(self):
        """
        :return: returns a list of lists of related params from all tasks specific networks
        """
        param_groups = []
        for out in zip(*[n.named_parameters() for n in self.model.task_nets]):
            if 'weight' in out[0][0]:
                param_groups.append(
                    [
                        out[i][1]
                        for i in range(len(out))
                    ]
                )
        return param_groups

    def soft_loss(self):
        param_groups = self.get_param_groups()

        soft_sharing_loss = torch.tensor(0.)
        for params in param_groups:
            soft_sharing_loss += torch.norm(params[0] - params[1], p='fro') + torch.norm(params[0] - params[2], p='fro') + torch.norm(params[1] - params[2], p='fro')

        return soft_sharing_loss

    def forward(self, x, return_loss=True) -> Tuple:
        """
        :param x: input
        :param return_loss: If True, soft sharing loss will be returned
        :return: always returns a tuple.
        """
        outupts = tuple([self.model(x)], )

        if return_loss:
            soft_loss = self.soft_loss()
            outputs = outupts + (soft_loss,)

        return outputs