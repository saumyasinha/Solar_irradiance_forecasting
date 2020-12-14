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

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

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


def train(X_train, y_train, X_valid, y_valid, input_size, hidden_size, n_hidden, n_tasks, folder_saving, model_saved, n_epochs, lr, batch_size, soft_loss_weight = 0):

    os.makedirs(folder_saving, exist_ok=True)

    # check if CUDA is available
    train_on_gpu = torch.cuda.is_available()
    print(train_on_gpu) 

    criterion = nn.MSELoss()
    #
    model = soft_parameter_sharing.SoftSharing(
        input_size=input_size,
        hidden_size = hidden_size,
        n_hidden = n_hidden,
        n_outputs= n_tasks
    )

    ## Print total parameters and trainable parameters
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f'{total_params:,} total parameters.')
    # total_trainable_params = sum(
    #     p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'{total_trainable_params:,} training parameters.')

    # model = hard_parameter_sharing.HardSharing(
    #     input_size=input_size,
    #     hidden_size = hidden_size,
    #     n_hidden = n_hidden,
    #     n_outputs= n_tasks
    # )

    print(model)

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    # optimizer = optim.Adam([
    #     {'params': model.model.task_nets[0].parameters(), 'lr':0.0001},
    #     {'params': model.model.task_nets[1].parameters(), 'lr':0.00001},
    # {'params': model.model.task_nets[2].parameters(), 'lr':0.00001}], betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    #


    if train_on_gpu:
        model.cuda()

    valid_loss_min = np.Inf  # track change in validation loss

    train_loss_list = []
    valid_loss_list = []
    task_specific_train_loss_list = defaultdict(list)
    task_specific_valid_loss_list = defaultdict(list)
    train_soft_loss_list = []
    valid_soft_loss_list = []

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

    # initialize the early_stopping object
    patience = 50
    early_stopping = EarlyStopping(patience=patience, verbose=True, path = folder_saving+model_saved)

    for epoch in range(1, n_epochs + 1):

        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        train_soft_loss = 0.0
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
            # output = model(data)
            output, soft_loss = model(data)
            # print(output.shape)
            total_loss = []
            for n in range(n_tasks):
                y_pred = output[:,n]
                # calculate the batch loss
                loss = criterion(y_pred, target[:,n])
                task_specific_train_loss[n]+=loss.item()
                total_loss.append(loss)

            # loss = (sum(total_loss))/len(total_loss) # in case of soft loss (loss = loss+soft_loss)
            loss = (sum(total_loss))+soft_loss_weight*soft_loss
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item()
            train_soft_loss= soft_loss.item()


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
        # output = model(data)
        output, soft_loss = model(data)
        total_loss = []
        for n in range(n_tasks):
            y_pred = output[:,n]
            # calculate the batch loss
            loss = criterion(y_pred, target[:,n])
            task_specific_valid_loss[n] = loss.item()
            total_loss.append(loss)

        # loss = (sum(total_loss)/len(total_loss))
        loss = (sum(total_loss)) + soft_loss_weight*soft_loss
        # update validation loss
        valid_loss = loss.item()
        valid_soft_loss = soft_loss.item()

        # calculate average losses
        train_loss = np.average(train_loss)
        # valid_loss = valid_loss / count_valid

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        train_soft_loss_list.append(train_soft_loss)
        valid_soft_loss_list.append(valid_soft_loss)

        for n in range(n_tasks):
            task_specific_train_loss_list[n].append(np.average(task_specific_train_loss[n]))
            task_specific_valid_loss_list[n].append(task_specific_valid_loss[n])

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))

        # # save model if validation loss has decreased
        # if valid_loss <= valid_loss_min:
        #     print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        #         valid_loss_min,
        #         valid_loss))
        #     torch.save(model.state_dict(), folder_saving+model_saved)
        #     valid_loss_min = valid_loss

        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    loss_plots(train_loss_list, valid_loss_list, folder_saving)
    loss_plots(train_soft_loss_list, valid_soft_loss_list, folder_saving, loss_type = "for soft loss")

    for n in range(n_tasks):
        loss_plots(task_specific_train_loss_list[n],task_specific_valid_loss_list[n], folder_saving, loss_type='for task:'+str(n))


def train_with_clusters(X_train, y_train, X_valid, y_valid, kmeans, cluster_labels, cluster_labels_valid, n_clusters, input_size, hidden_sizes, task_specific_hidden_sizes, folder_saving, model_saved, n_epochs, lr, batch_size, weight_decay, lead, pretrained_path):

    # check if CUDA is available
    train_on_gpu = torch.cuda.is_available()
    print(train_on_gpu)

    criterion = nn.MSELoss()


    model = hard_parameter_sharing.Custom_HardSharing(
        input_size=input_size,
        hidden_sizes = hidden_sizes,
        n_outputs= n_clusters,
        pretrained_path=pretrained_path,
        task_specific_hidden_sizes = task_specific_hidden_sizes
    )

    # model = hard_parameter_sharing.HardSharing(
    #     input_size=input_size,
    #     hidden_size = hidden_size,
    #     n_hidden = n_hidden,
    #     n_outputs= n_clusters,
    #     task_specific_hidden_size = task_specific_hidden_size
    # )

    print(model)
    ## Print total parameters and trainable parameters
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f'{total_params:,} total parameters.')
    # total_trainable_params = sum(
    #     p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'{total_trainable_params:,} training parameters.')

    #
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
   #  optimizer = optim.Adam([
   #      {'params': model.model.hard_sharing.parameters(), 'lr':0.0001}
   # ], lr = lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
   #  decayRate = 0.96
    # my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

    if train_on_gpu:
        model.cuda()

    valid_loss_min = np.Inf  # track change in validation loss

    train_loss_list = []
    valid_loss_list = []
    task_specific_train_loss_list = defaultdict(list)
    task_specific_valid_loss_list = defaultdict(list)


    count_train, count_valid = X_train.shape[0], X_valid.shape[0]
    print(count_train, count_valid)

    # kmeans, cluster_labels = clustering.clustering(X_train, features_indices_to_cluster_on, n_clusters=n_clusters)
    # cluster_labels_valid = clustering.get_closest_clusters(X_valid, kmeans, features_indices_to_cluster_on)

    X_train = torch.from_numpy(X_train)
    X_train = X_train.float()
    y_train = torch.tensor(y_train)
    y_train = y_train.float()
    X_valid = torch.from_numpy(X_valid)
    X_valid = X_valid.float()
    y_valid = torch.tensor(y_valid)
    y_valid = y_valid.float()
    cluster_labels = torch.from_numpy(cluster_labels)
    cluster_labels_valid = torch.from_numpy(cluster_labels_valid)


    # # initialize the early_stopping object
    # patience = 50
    # early_stopping = EarlyStopping(patience=patience, verbose=True, path=folder_saving + model_saved)

    for epoch in range(1, n_epochs + 1):

        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        train_soft_loss = 0.0
        task_specific_train_loss = defaultdict(float)
        task_specific_valid_loss = defaultdict(float)

        ###################
        # train the model #
        ###################
        # my_lr_scheduler.step()
        model.train()

        minibatches_for_all_clusters = {}
        for i in range(n_clusters):
            X_train_task = X_train[cluster_labels == i]
            y_train_task = y_train[cluster_labels == i]
            # print("\ntask: ",i," ")
            # print(X_train_task.shape)
            minibatches_for_all_clusters[i] = random_mini_batches(X_train_task, y_train_task, batch_size)

        common_batches = len(min(minibatches_for_all_clusters.values(), key=len))
        # print(common_batches)

        for iter in range(common_batches):
            total_minibatch_loss = []
            for i in range(n_clusters):
                minibatch = minibatches_for_all_clusters[i][iter]
                data, target = minibatch
                # print(data.shape, target.shape, type(data), type(target))
                # move tensors to GPU if CUDA is available
                # target = target.long()
                if train_on_gpu:
                    data, target = data.cuda(), target.cuda()

                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data)
                # print(output.shape)
                y_pred = output[:, i]
                # calculate the batch loss
                loss = criterion(y_pred, target)
                task_specific_train_loss[i] += loss.item()
                total_minibatch_loss.append(loss)

            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            loss = sum(total_minibatch_loss)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item()


        ######################
        # validate the model #
        ######################
        model.eval()
        total_valid_loss = []
        for i in range(n_clusters):
            data = X_valid[cluster_labels_valid == i]
            target = y_valid[cluster_labels_valid==i]

            # data, target = X_valid[cluster_labels_valid == i], y_valid[cluster_labels_valid==i]
            # move tensors to GPU if CUDA is available
            # target = target.long()
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # print(output.shape)
            y_pred = output[:, i]
            # calculate the batch loss
            loss = criterion(y_pred, target)
            task_specific_valid_loss[i] += loss.item()
            total_valid_loss.append(loss)

        valid_loss = (sum(total_valid_loss)).item()

        # calculate average losses
        train_loss = train_loss/common_batches
        # valid_loss = valid_loss / count_valid

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

        for n in range(n_clusters):
            task_specific_train_loss_list[n].append(task_specific_train_loss[n]/common_batches)
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



    loss_plots(train_loss_list, valid_loss_list, folder_saving, loss_type = "for_lead_"+str(lead))
    # loss_plots(train_soft_loss_list, valid_soft_loss_list, folder_saving, loss_type="for soft loss")

    for n in range(n_clusters):
        loss_plots(task_specific_train_loss_list[n], task_specific_valid_loss_list[n], folder_saving,
                   loss_type='for_task_' + str(n)+"_and_lead_"+str(lead))

    return kmeans


# def train_with_clusters_after_finetuning():

