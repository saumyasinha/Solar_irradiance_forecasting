import torch
import os
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import defaultdict
from SolarForecasting.ModulesMultiTaskLearning import hard_parameter_sharing, soft_parameter_sharing



def loss_plots(train_loss, valid_loss, folder_saving, loss_type=""):
    epochs = range(2, len(train_loss)+1)
    train_loss = train_loss[1:]
    valid_loss = valid_loss[1:]
    plt.figure()
    plt.plot(epochs, train_loss)
    plt.plot(epochs, valid_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['training', 'validation'], loc='lower right')

    plt.savefig(folder_saving+"loss_plots_"+loss_type)
    plt.close()


def random_mini_batches(X, y, batch_size=32, seed=42):
    # Creating the mini-batches
    np.random.seed(seed)
    m = X.shape[0]
    mini_batches = []
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_y = y[permutation]
    num_complete_minibatches = math.floor(m / batch_size)
    for k in range(0, int(num_complete_minibatches)):
        mini_batch_X = shuffled_X[k * batch_size: (k + 1) * batch_size, :]
        mini_batch_y = shuffled_y[k * batch_size: (k + 1) * batch_size]
        mini_batch = (mini_batch_X, mini_batch_y)
        mini_batches.append(mini_batch)
    Lower = int(num_complete_minibatches * batch_size)
    Upper = int(m - (batch_size * math.floor(m / batch_size)))
    if m % batch_size != 0:
        mini_batch_X = shuffled_X[Lower: Lower + Upper, :]
        mini_batch_y = shuffled_y[Lower: Lower + Upper]
        mini_batch = (mini_batch_X, mini_batch_y)
        mini_batches.append(mini_batch)

    return mini_batches


def train(X_train, y_train, X_valid, y_valid, input_size, hidden_size, n_hidden, n_tasks, folder_saving, model_saved, n_epochs, lr, batch_size):

    os.makedirs(folder_saving, exist_ok=True)

    # check if CUDA is available
    train_on_gpu = torch.cuda.is_available()
    print(train_on_gpu) 

    criterion = nn.MSELoss()



    ## Print total parameters and trainable parameters
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f'{total_params:,} total parameters.')
    # total_trainable_params = sum(
    #     p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'{total_trainable_params:,} training parameters.')

    model = hard_parameter_sharing.HardSharing(
        input_size=input_size,
        hidden_size = hidden_size,
        n_hidden = n_hidden,
        n_outputs= n_tasks
    )

    print(model)

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    # optimizer = optim.Adam([
    #     {'params': model.base.parameters()},
    #     {'params': model.task_specific.task_nets.parameters(), 'lr': 1e-3}], lr=lr, betas=(0.9, 0.999), eps=1e-08)

    if train_on_gpu:
        model.cuda()

    valid_loss_min = np.Inf  # track change in validation loss

    train_loss_list = []
    valid_loss_list = []
    task_specific_train_loss_list = defaultdict(list)
    task_specific_valid_loss_list = defaultdict(list)

    count_train, count_valid = X_train.shape[0], X_valid.shape[0]
    print(count_train,count_valid)

    X_train = torch.from_numpy(X_train)
    X_train = X_train.float()
    y_train = torch.tensor(y_train)
    y_train = y_train.float()

    X_valid = torch.from_numpy(X_valid)
    X_valid = X_valid.float()
    y_valid = torch.tensor(y_valid)
    y_valid = y_valid.float()

    for epoch in range(1, n_epochs + 1):

        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        task_specific_train_loss = defaultdict(float)
        task_specific_valid_loss = defaultdict(float)

        ###################
        # train the model #
        ###################
        model.train()
        minibatches = random_mini_batches(X_train, y_train, batch_size)
        for minibatch in minibatches:
            data, target = minibatch
            # move tensors to GPU if CUDA is available
            # target = target.long()
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output= model(data)
            # print(output.shape)
            total_loss = []
            for n in range(n_tasks):
                y_pred = output[:,n]
                # calculate the batch loss
                loss = criterion(y_pred, target[:,n])
                task_specific_train_loss[n]+=loss.item() * data.size(0)
                total_loss.append(loss)

            loss = (sum(total_loss))/len(total_loss)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item() * data.size(0)


        ######################
        # validate the model #
        ######################
        model.eval()

        data, target = X_valid, y_valid
        # target = target.long()
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        total_loss = []
        for n in range(n_tasks):
            y_pred = output[:,n]
            # calculate the batch loss
            loss = criterion(y_pred, target[:,n])
            task_specific_valid_loss[n] = loss.item()
            total_loss.append(loss)

        loss = (sum(total_loss))/len(total_loss)
        # update validation loss
        valid_loss = loss.item()

        # calculate average losses
        train_loss = train_loss / count_train
        # valid_loss = valid_loss / count_valid

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

        for n in range(n_tasks):
            task_specific_train_loss_list[n].append(task_specific_train_loss[n] / count_train)
            task_specific_valid_loss_list[n].append(task_specific_valid_loss[n])

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            torch.save(model.state_dict(), folder_saving+model_saved)
            valid_loss_min = valid_loss

    loss_plots(train_loss_list, valid_loss_list, folder_saving)

    for n in range(n_tasks):
        loss_plots(task_specific_train_loss_list[n],task_specific_valid_loss_list[n], folder_saving, loss_type='for task:'+str(n))


