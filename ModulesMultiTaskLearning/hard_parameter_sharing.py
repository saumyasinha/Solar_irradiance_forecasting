import torch
from torch import nn


class FFNN(nn.Module):
    """Simple FF network with multiple outputs.
    """
    def __init__(
        self,
        input_size,
        hidden_size,
        n_hidden,
        n_outputs,
        dropout_rate=.1,
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


class TaskIndependentLayers(nn.Module):
    """NN for MTL with hard parameter sharing
    """

    def __init__(
            self,
            input_size,
            hidden_size,
            n_hidden,
            n_outputs,
            dropout_rate=.1,
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


class HardSharing(nn.Module):
    """FFNN with hard parameter sharing
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        n_hidden,
        n_outputs,
        n_task_specific_layers=1,
        task_specific_hidden_size=None,
        dropout_rate=.1,
    ):

        super().__init__()
        if task_specific_hidden_size is None:
            task_specific_hidden_size = hidden_size

        self.model = nn.Sequential()

        self.model.add_module(
            'hard_sharing',
            FFNN(
                input_size=input_size,
                hidden_size=hidden_size,
                n_hidden=n_hidden,
                n_outputs=hidden_size,
                dropout_rate=dropout_rate
            )
        )

        if n_task_specific_layers > 0:
            # if n_task_specific_layers == 0 than the task specific mapping is linear and
            # constructed as the product of last layer in the 'hard_sharing' and the linear layer
            # in 'task_specific', with no activation or dropout
            self.model.add_module('relu', nn.ReLU())
            self.model.add_module('dropout', nn.Dropout(p=dropout_rate))

        self.model.add_module(
            'task_specific',
            TaskIndependentLayers(
                input_size=hidden_size,
                hidden_size=task_specific_hidden_size,
                n_hidden=n_task_specific_layers,
                n_outputs=n_outputs,
                dropout_rate=dropout_rate
            )
        )

    def forward(self, x):
        return self.model(x)