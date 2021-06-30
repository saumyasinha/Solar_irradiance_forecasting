
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from sklearn.metrics import mean_squared_error
from torch.autograd import Variable
# import torch.optim.lr_scheduler.StepLR
from torch.nn.utils import weight_norm
from SolarForecasting.ModulesLearning.ModulesCNN.tcn import TemporalConvNet


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, saving_path, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.saving_path = saving_path

    def __call__(self, val_loss, model, epoch):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif (score < self.best_score + self.delta):
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        elif epoch>20:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print('Validation loss decreased ({} --> {}).  Saving model ...'.format(self.val_loss_min, val_loss))
        torch.save(model.state_dict(), self.saving_path)
        self.val_loss_min = val_loss



class ConvForecasterDilationLowRes(nn.Module):
    def __init__(self, input_dim, timesteps, folder_saving, model, quantile, alphas=None, outputs=None, valid=False):
        super(ConvForecasterDilationLowRes, self).__init__()
        self.quantile = quantile

        def weights_init(m):
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data)
                nn.init.zeros_(m.bias.data)

        if self.quantile:
            assert outputs == len(alphas), "The outputs and the quantiles should be of the same dimension"

        self.input_dim = input_dim
        self.timesteps = timesteps
        self.alphas = alphas
        self.outputs = outputs
        self.valid = valid
        self.train_mode = False
        self.saving_path = folder_saving+model

        # self.conv1 = nn.Conv1d(self.input_dim, 40, 2, stride=1)
        # self.conv1_fn = nn.ReLU()
        # self.avgpool1 = nn.AvgPool1d(kernel_size=2, stride=1)
        # self.conv1.apply(weights_init)
        #
        # self.conv2 = nn.Conv1d(40, 80, 3, stride=1, dilation=2)
        # self.conv2_fn = nn.ReLU()
        # self.avgpool2 = nn.AvgPool1d(kernel_size=2, stride=2)
        # self.conv2.apply(weights_init)
        #
        # self.conv3 = nn.Conv1d(80, 128, 3, stride=1, dilation=4)
        # self.conv3_fn = nn.ReLU()
        # self.avgpool3 = nn.AvgPool1d(kernel_size=2, stride=1)
        # # self.avgpool3 = nn.AvgPool1d(kernel_size=2, stride=2)
        # self.conv3.apply(weights_init)
        #
        # conv_layers = [ self.conv1,self.conv1_fn,self.avgpool1,self.conv2,self.conv2_fn,self.avgpool2,self.conv3,self.conv3_fn,self.avgpool3]
        # # conv_layers = [self.conv1, self.conv1_fn, self.conv2, self.conv2_fn, self.conv3,
        # #                self.conv3_fn]
        #
        # # self.conv4 = nn.Conv1d(100, 150, 3, stride=1, dilation=6)
        # # self.conv4_fn = nn.ReLU()
        # # self.avgpool4 = nn.AvgPool1d(kernel_size=2, stride=1)
        #
        # conv_module = nn.Sequential(*conv_layers)
        #
        # test_ipt = Variable(torch.zeros(1, self.input_dim, self.timesteps))
        # test_out = conv_module(test_ipt)
        #
        # self.conv_output_size = test_out.size(1) * test_out.size(2)
        # fc1_dim = self.conv_output_size+(self.input_dim*self.timesteps)
        #
        # self.dropout = nn.Dropout(0.5)
        #
        # #self.fc1 = nn.Linear(fc1_dim, int(fc1_dim/2))#int(fc1_dim/4))
        # #self.fc1_fn = nn.Tanh()
        #
        # # self.fc2 = nn.Linear(int(fc1_dim/2), int(fc1_dim/4))
        # # self.fc2_fn = nn.Tanh()
        #
        #
        # # Attention Layer :
        # self.conv_attn = nn.Conv1d(self.input_dim, 1, 1, stride=1)
        # self.attn_layer = nn.Sequential(
        #     #nn.Linear(self.conv_output_size+self.timesteps, self.timesteps*self.input_dim),
        #     #nn.Tanh(),
        #     # nn.Linear(self.timesteps*self.input_dim, self.timesteps),
        #     nn.Linear(self.conv_output_size+self.timesteps, self.timesteps),
        #     nn.Softmax(dim=1)
        # )
        #
        #
        # #self.fc3 = nn.Linear(int(fc1_dim/2), self.outputs)
        # self.fc3 = nn.Linear(fc1_dim, self.outputs)
        #
        # ## adding self attention (and not the one above)
        # # in_dim = test_out.size(1)
        # # self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        # # self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        # # self.value_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        # # self.gamma = nn.Parameter(torch.zeros(1))
        # # self.softmax = nn.Softmax(dim=-1)
        # # self.fc = nn.Linear(self.conv_output_size,self.outputs)

        ## play around with channel size and kernel size
        num_channels =[25]*6 #24/25 before and 6 num of channels, reduced for multihead

        self.tcn = TemporalConvNet(self.input_dim, num_channels, kernel_size=5, dropout=0.2, attention=True)

        self.linear = nn.Linear(num_channels[-1], self.outputs)



    def forward(self, xx, n_output_length=1):
        # # print(xx.shape)
        # output = self.conv1(xx)
        # # print(output.shape)
        # output = self.conv1_fn(output)
        # if self.train_mode:
        #     output = self.dropout(output)
        # output = self.avgpool1(output)
        # # print(output.shape)
        #
        # output = self.conv2(output)
        # # print(output.shape)
        # output = self.conv2_fn(output)
        # if self.train_mode:
        #     output = self.dropout(output)
        # output = self.avgpool2(output)
        #
        # # print(output.shape)
        #
        # output = self.conv3(output)
        # # print(output.shape)
        # output = self.conv3_fn(output)
        # if self.train_mode:
        #     output = self.dropout(output)
        # output = self.avgpool3(output)
        #
        # # output = self.conv4(output)
        # # output = self.conv4_fn(output)
        # # output = self.avgpool4(output)
        # output = output.reshape(-1, output.shape[1]*output.shape[2])
        # # print("after convolution: ", output.shape)
        # # Compute Context Vector
        # xx_single = self.conv_attn(xx).reshape(-1, self.timesteps)
        # # print("xx_single: ", xx_single.shape)
        # attn_input = torch.cat((output, xx_single), dim=1)
        # # print("attn_input: ", attn_input.shape)
        # attention = self.attn_layer(attn_input).reshape(-1, 1, self.timesteps)
        #
        # # print("attention: ", attention.shape)
        # x_attenuated = (xx * attention)
        # # print("xx attentuated: ",x_attenuated.shape)
        # x_attenuated = x_attenuated.reshape(-1, x_attenuated.shape[1]*x_attenuated.shape[2])
        #
        #
        # output = torch.cat((output, x_attenuated), dim=1)
        # # print("output concat with attenuated: ",output.shape)
        # #output = self.fc1(output)
        # #output = self.fc1_fn(output)
        # #if self.train_mode:
        #  #   output = self.dropout(output)
        # #
        # # output = self.fc2(output)
        # # output = self.fc2_fn(output)
        # # if self.train_mode:
        # #     output = self.dropout(output)
        # output = self.fc3(output)
        #

        if n_output_length>1:
            # print(xx.shape)
            output = self.tcn(xx).transpose(1,2)
            # print(output.shape)
            output = self.linear(output).transpose(1,2)[:,:,:n_output_length] #unsure here if it should be [:,:,-n_output_length:]
            # print(output.shape)
        else:
            output = self.tcn(xx)
            output = self.linear(output[:,:,-1])


        return output



        ## forward function for working with dilated kernels and self-attention
    # def forward(self,xx):
    #     # print(xx.shape)
    #     output = self.conv1(xx)
    #     # print(output.shape)
    #     output = self.conv1_fn(output)
    #     if self.train_mode:
    #         output = self.dropout(output)
    #     output = self.avgpool1(output)
    #     # print(output.shape)
    #
    #     output = self.conv2(output)
    #     # print(output.shape)
    #     output = self.conv2_fn(output)
    #     if self.train_mode:
    #         output = self.dropout(output)
    #     output = self.avgpool2(output)
    #     # print(output.shape)
    #
    #     output = self.conv3(output)
    #     # print(output.shape)
    #     output = self.conv3_fn(output)
    #     if self.train_mode:
    #         output = self.dropout(output)
    #     output = self.avgpool3(output)
    #     # print(output.shape)
    #
    #     # output = output.reshape(-1, output.shape[1] * output.shape[2])
    #     # print("batch, channels, timestep: ",m_batchsize, n_channel, timestep)
    #
    #     proj_query = self.query_conv(output).permute(0,2,1)
    #     proj_key = self.key_conv(output)
    #     energy = torch.bmm(proj_query, proj_key)
    #     attention = self.softmax(energy)
    #     proj_value = self.value_conv(output)
    #
    #     # print("value projected shape: ", proj_value.shape)
    #     # print("query projected shape: ", proj_query.shape)
    #     # print("key projected shape: ", proj_key.shape)
    #     # print("attention shape: ", attention.shape)
    #
    #     out = torch.bmm(proj_value, attention.permute(0,2,1))
    #     # print(out.shape)
    #
    #     output = self.gamma * out + output
    #
    #     output = output.reshape(-1, output.shape[1] * output.shape[2])
    #     output = self.fc(output)
    #
    #     return output

    def trainBatchwise(self, trainX, trainY, epochs, batch_size, lr=0.0001, validX=None,
                       validY=None, n_output_length = 1, patience=None, verbose=None, reg_lamdba = 0): #0.0001):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # scheduler = StepLR(optimizer, step_size=25, gamma=0.1)
        criterion = torch.nn.MSELoss()
        # criterion = nn.L1Loss()
        samples = trainX.size()[0]
        losses = []
        valid_losses = []

        early_stopping = EarlyStopping(self.saving_path, patience=patience, verbose=True)

        for epoch in range(epochs):
            if self.train_mode is not True:
                self.train()
                self.train_mode = True

            indices = torch.randperm(samples)
            trainX, trainY = trainX[indices, :, :], trainY[indices]
            per_epoch_loss = 0
            count_train = 0
            for i in range(0, samples, batch_size):
                xx = trainX[i: i + batch_size, :, :]
                yy = trainY[i: i + batch_size]

                if torch.cuda.is_available():
                    xx, yy = xx.cuda(), yy.cuda()

                outputs = self.forward(xx, n_output_length)
                optimizer.zero_grad()
                if self.quantile:
                    if n_output_length==1:
                        loss = self.quantile_loss(outputs, yy)
                    else:
                        # train loss for multiple outputs or multi-task learning
                        total_loss = []
                        for n in range(n_output_length):
                            y_pred = outputs[:,:, n]
                            # calculate the batch loss
                            loss = self.quantile_loss(y_pred, yy[:, n])
                            total_loss.append(loss)

                        loss = sum(total_loss)

                else:
                    loss = criterion(outputs, yy)


                reg_loss = np.sum([weights.norm(2) for weights in self.parameters()])
                total_loss = loss + reg_lamdba/ 2 * reg_loss
                # backward pass: compute gradient of the loss with respect to model parameters
                total_loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()
                # scheduler.step()
                per_epoch_loss+=loss.item()
                count_train+=1


            train_loss_this_epoch = per_epoch_loss/count_train
            losses.append(train_loss_this_epoch)

            if self.valid:
                self.train_mode = False
                self.eval()
                if torch.cuda.is_available():

                    validX,validY = validX.cuda(), validY.cuda()


                if self.quantile:
                    validYPred = self.forward(validX,n_output_length)
                    # validYPred = validYPred.cpu().detach().numpy()
                    # validYTrue = validY.cpu().detach().numpy()
                    # valid_loss_this_epoch = self.quantile_loss(validYPred,validY).item()

                    if n_output_length == 1:
                        valid_loss_this_epoch =  self.quantile_loss(validYPred,validY).item()
                    else:
                        # train loss for multiple outputs or multi-task learning
                        total_loss = []
                        for n in range(n_output_length):
                            y_pred = validYPred[:, :, n]
                            # calculate the batch loss
                            loss = self.quantile_loss(y_pred, validY[:, n])
                            total_loss.append(loss)

                        valid_loss_this_epoch = sum(total_loss).item()

                    # valid_loss = elf.crps_score(validYPred, validYTrue, np.arange(0.05, 1.0, 0.05))s
                    valid_losses.append(valid_loss_this_epoch)
                    print("Epoch: %d, loss: %1.5f and valid_loss : %1.5f" % (epoch, train_loss_this_epoch, valid_loss_this_epoch))
                else:
                    validYPred = self.forward(validX,n_output_length)

                    # # valid loss for multiple outputs or multi-task learning
                    # total_loss = []
                    # for n in range(self.outputs):
                    #     y_pred = validYPred[:, n]
                    #     # calculate the batch lossnb6h
                    #     validloss = criterion(y_pred, validY[:, n])
                    #     total_loss.append(validloss)

                    # validloss = sum(total_loss)
                    # valid_loss_this_epoch = validloss.item()
                    valid_loss_this_epoch = criterion(validYPred, validY).item()
                    # validYPred = validYPred.cpu().detach().numpy()
                    # validYTrue = validY.cpu().detach().numpy()
                    # valid_loss = np.sqrt(mean_squared_error(validYPred, validYTrue))
                    valid_losses.append(valid_loss_this_epoch)
                    print("Epoch: %d, train loss: %1.5f and valid loss : %1.5f" % (epoch, train_loss_this_epoch, valid_loss_this_epoch))

                # early_stopping(valid_loss, self)
                early_stopping(valid_loss_this_epoch, self, epoch)

                if early_stopping.early_stop:
                    print("Early stopping")
                    break
            else:
                print("Epoch: %d, loss: %1.5f" % (epoch, train_loss_this_epoch))
                torch.save(self.state_dict(), self.saving_path)
        # load the last checkpoint with the best model
        # self.load_state_dict(torch.load('checkpoint.pt'))
        return losses, valid_losses

    def crps_score(self, outputs, target, alphas):
        loss = []
        for i, alpha in enumerate(alphas):
            output = outputs[:, i].reshape((-1, 1))
            covered_flag = (output <= target).astype(np.float32)
            uncovered_flag = (output > target).astype(np.float32)
            if i == 0:
                loss.append(np.mean(
                    ((target - output) * alpha * covered_flag + (output - target) * (1 - alpha) * uncovered_flag)))
            else:
                loss.append(np.mean(
                    ((target - output) * alpha * covered_flag + (output - target) * (1 - alpha) * uncovered_flag)))

        return 2*np.mean(np.array(loss))

    def quantile_loss(self, outputs, target):
        for i, alpha in zip(range(self.outputs), self.alphas):
            output = outputs[:, i].reshape((-1, 1))
            covered_flag = (output <= target).float()
            uncovered_flag = (output > target).float()
            if i == 0:
                loss = ((target - output) * alpha * covered_flag + (output - target) * (1 - alpha) * uncovered_flag)
            else:
                loss += ((target - output) * alpha * covered_flag + (output - target) * (1 - alpha) * uncovered_flag)

        return torch.mean(loss)



 # ## Wavenet style model architecture

# class DilatedCausalConv1d(nn.Module):
#     def __init__(self, hyperparams: dict, dilation_factor: int, in_channels: int, out_channels: int):
#         super().__init__()
#
#         def weights_init(m):
#             if isinstance(m, nn.Conv1d):
#                 nn.init.kaiming_normal_(m.weight.data)
#                 nn.init.zeros_(m.bias.data)
#
#         self.dilation_factor = dilation_factor
#         self.dilated_causal_conv = nn.Conv1d(in_channels=in_channels,
#                                              out_channels=out_channels,
#                                              kernel_size=hyperparams['kernel_size'],
#                                              dilation=dilation_factor)
#         self.dilated_causal_conv.apply(weights_init)
#
#         self.skip_connection = nn.Conv1d(in_channels=in_channels,
#                                          out_channels=out_channels,
#                                          kernel_size=1)
#         self.skip_connection.apply(weights_init)
#         self.leaky_relu = nn.LeakyReLU(0.1)
#
#     def forward(self, x):
#         x1 = self.leaky_relu(self.dilated_causal_conv(x))
#         x2 = x[:, :, self.dilation_factor:]
#         x2 = self.skip_connection(x2)
#         return x1 + x2
        # hyperparams = {}
        # hyperparams['nb_layers'] = 5 #5
        # # hyperparams['nb_filters'] = 64
        # hyperparams['kernel_size'] = 2
        #
        # receptive_field = 2 ** (hyperparams['nb_layers'] - 1) * hyperparams['kernel_size']
        # self.padding = receptive_field - 1
        #
        # in_channels = self.input_dim
        # self.dilation_factors = [2 ** i for i in range(0, hyperparams['nb_layers'])]
        # self.in_channels = [in_channels] + [32 for _ in range(hyperparams['nb_layers'])] #2**(5+_)
        # self.dilated_causal_convs = nn.ModuleList(
        #     [DilatedCausalConv1d(hyperparams, self.dilation_factors[i], self.in_channels[i], self.in_channels[i+1]) for i in
        #      range(hyperparams['nb_layers'])])
        # for dilated_causal_conv in self.dilated_causal_convs:
        #     dilated_causal_conv.apply(weights_init)
        #
        # #This is part of the original code: if you want to get [batch_size,1,timesteps] as output
        # # self.output_layer = nn.Conv1d(in_channels=self.in_channels[-1],
        # #                               out_channels=1,
        # #                               kernel_size=1)
        #
        # ##This is when you want to use purely wavenet architecture and give n_outputs as output with 2 fc layers
        # self.fc_dim = self.in_channels[-1]*self.timesteps
        # self.output_layer1 = nn.Linear(self.in_channels[-1]*self.timesteps,int(self.fc_dim/2))
        # self.output_layer1.apply(weights_init)
        # self.output_layer2 = nn.Linear(int(self.fc_dim / 2),self.outputs)
        # self.output_layer2.apply(weights_init)
        # self.leaky_relu = nn.LeakyReLU(0.1)
        # self.dropout = nn.Dropout(0.5)
        #
        # ## This is when you want to add self attention to the wavent style architecture
        # # in_dim = self.in_channels[-1]
        # # self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1) #8
        # # self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1) #8
        # # self.value_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        # # self.gamma = nn.Parameter(torch.zeros(1))
        # # self.softmax = nn.Softmax(dim=-1)
        # # self.fc = nn.Linear(self.fc_dim, self.outputs)


    # Forward function for wavenet style model
    # def forward(self, x):
    #
    #
    #     npad = ((0, 0), (0, 0), (self.padding, 0))
    #     x = torch.from_numpy(np.pad(x, pad_width=npad, mode='constant', constant_values=0))
    #
    #     for dilated_causal_conv in self.dilated_causal_convs:
    #         x = dilated_causal_conv(x)
    #
    #
    #
    #     # x = self.leaky_relu(self.output_layer(x))
    #     # # output = x[:,0,-1]
    #     # output = x[:,0,:]
    #
    #     output = x.reshape(-1,x.shape[1] * x.shape[2])
    #     output = self.leaky_relu(self.output_layer1(output))
    #     output = self.dropout(output)
    #
    #     output = self.leaky_relu(self.output_layer2(output))
    #     # output = x
    #     #
    #     # proj_query = self.query_conv(output).permute(0, 2, 1)
    #     # proj_key = self.key_conv(output)
    #     # energy = torch.bmm(proj_query, proj_key)
    #     # attention = self.softmax(energy)
    #     # proj_value = self.value_conv(output)
    #     #
    #     # out = torch.bmm(proj_value, attention.permute(0,2,1))
    #     # # print(out.shape)
    #     #
    #     # output = self.gamma * out + output
    #     #
    #     # output = output.reshape(-1, output.shape[1] * output.shape[2])
    #     # output = self.leaky_relu(self.fc(output))
    #

        # return output


