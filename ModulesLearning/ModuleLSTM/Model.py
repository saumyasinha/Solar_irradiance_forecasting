import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

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


class quantileLSTM(nn.Module):

    # def __init__(self, num_classes, input_size, hidden_size, num_layers)
    def __init__(self, input_dim, timesteps, folder_saving, model, quantile, hidden_size, num_layers=1, alphas=None, outputs=None, valid=False):
        super(quantileLSTM, self).__init__()

        self.quantile = quantile
        if self.quantile:
            assert outputs == len(alphas), "The outputs and the quantiles should be of the same dimension"


        self.outputs = outputs
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.timesteps = timesteps
        self.alphas = alphas
        self.valid = valid
        self.train_mode = False
        self.saving_path = folder_saving + model

        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, self.outputs)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))

        h_out = h_out.view(-1, self.hidden_size)

        out = self.fc(h_out)

        return out


    def trainBatchwise(self, trainX, trainY, epochs, batch_size, lr=0.0001, validX=None,
                       validY=None, patience=None, verbose=None, reg_lamdba = 0.0001):
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

                outputs = self.forward(xx)
                optimizer.zero_grad()
                if self.quantile:
                    loss = self.quantile_loss(outputs, yy)
                else:
                    loss = criterion(outputs, yy)

                    ## train loss for multiple outputs or multi-task learning
                    # total_loss = []
                    # for n in range(self.outputs):
                    #     y_pred = outputs[:, n]
                    #     # calculate the batch loss
                    #     loss = criterion(y_pred, yy[:, n])
                    #     total_loss.append(loss)
                    #
                    # loss = sum(total_loss)

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
                    validYPred = self.forward(validX)
                    # validYPred = validYPred.cpu().detach().numpy()
                    # validYTrue = validY.cpu().detach().numpy()
                    valid_loss_this_epoch = self.quantile_loss(validYPred,validY).item()
                    # valid_loss = elf.crps_score(validYPred, validYTrue, np.arange(0.05, 1.0, 0.05))s
                    valid_losses.append(valid_loss_this_epoch)
                    print("Epoch: %d, loss: %1.5f and valid_loss : %1.5f" % (epoch, train_loss_this_epoch, valid_loss_this_epoch))
                else:
                    validYPred = self.forward(validX)

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



