import torch
from torch import nn
from SolarForecasting.ModulesLearning import model as models


class FFNN(nn.Module):
    """Simple FF network with multiple outputs.
    """
    def __init__(
        self,
        input_size,
        hidden_sizes,
        n_outputs = None,
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

        if n_outputs is None:
            n_outputs = []

        h_sizes = [self.input_size] + hidden_sizes + n_outputs

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
            hidden_sizes,
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
                    hidden_sizes=hidden_sizes,
                    n_outputs=[1],
                    dropout_rate=dropout_rate,
                )
            )

    def forward(self, x):

        return torch.cat(
            tuple(task_model(x) for task_model in self.task_nets),
            dim=1
        )


# class HardSharing(nn.Module):
#     """FFNN with hard parameter sharing
#     """
#
#         def __init__(
#             self,
#             input_size,
#             hidden_sizes,
#             n_outputs,
#             pretrained_path,
#             task_specific_hidden_sizes=None,
#             dropout_rate=.5,
#     ):
#
#         super(Custom_HardSharing, self).__init__()
#
#
#         self.model = nn.Sequential()
#
#         self.model.add_module(
#             'hard_sharing',
#             FFNN(
#                 input_size=input_size,
#                 hidden_sizes=hidden_sizes,
#                 # n_outputs=hidden_sizes[-1],
#                 dropout_rate=dropout_rate
#             )
#         )
#
#
#         self.model.add_module('relu', nn.ReLU())
#         self.model.add_module('dropout', nn.Dropout(p=dropout_rate))
#
#         self.model.add_module(
#             'task_specific',
#             TaskIndependentLayers(
#                 input_size=hidden_sizes[-1],
#                 hidden_sizes=task_specific_hidden_sizes,
#                 n_outputs=n_outputs,
#                 dropout_rate=dropout_rate
#             )
#         )
#
#
#
#
#     def forward(self, x):
#
#         return self.model(x)


## Model used for transfer learning
class Custom_HardSharing(nn.Module):

    def __init__(
            self,
            input_size,
            hidden_sizes,
            n_outputs,
            pretrained_path,
            task_specific_hidden_sizes=None,
            dropout_rate=.5,
    ):

        super(Custom_HardSharing, self).__init__()


        self.model = nn.Sequential()

        self.model.add_module(
            'hard_sharing',
            FFNN(
                input_size=input_size,
                hidden_sizes=hidden_sizes,
                # n_outputs=hidden_sizes[-1],
                dropout_rate=dropout_rate
            )
        )

        # pretrained_model = models.Network(input_size=input_size,
        #         hidden_sizes=hidden_sizes)

        net = models.NeuralNetRegressor(
            models.Network(hidden_sizes=hidden_sizes),
            criterion=nn.MSELoss
        )

        net.initialize()
        net.load_params(f_params=pretrained_path)

        # pretrained_model.load_state_dict(torch.load(pretrained_path))
        pretrained_model = net.module_

        print("Model's state_dict:")
        for param_tensor in pretrained_model.state_dict():
            print(param_tensor, "\t", pretrained_model.state_dict()[param_tensor].size())

        # ## freezing the "features" parameters (this is excluding the fully connected layers)
        # for param in self.model.hard_sharing.parameters():
        #     param.requires_grad = False

        # if task_specific_hidden_sizes is None:
        #     task_specific_hidden_sizes = [hidden_sizes[-1]]

        # if task_specific_hidden_sizes is not None:
            # if n_task_specific_layers == 0 than the task specific mapping is linear and
            # constructed as the product of last layer in the 'hard_sharing' and the linear layer
            # in 'task_specific', with no activation or dropout
        self.model.add_module('relu', nn.ReLU())
        self.model.add_module('dropout', nn.Dropout(p=dropout_rate))

        if task_specific_hidden_sizes is None:
            task_specific_hidden_sizes = []

        self.model.add_module(
            'task_specific',
            TaskIndependentLayers(
                input_size=hidden_sizes[-1],
                hidden_sizes=task_specific_hidden_sizes,
                n_outputs=n_outputs,
                dropout_rate=dropout_rate
            )
        )

        ## Use pretrained "features" in your model
        self.model.hard_sharing = pretrained_model.model.hard_sharing


    def forward(self, x):

        return self.model(x)

