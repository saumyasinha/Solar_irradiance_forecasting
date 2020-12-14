import torch
import torch.nn as nn
import numpy as np
from SolarForecasting.ModulesMultiTaskLearning import hard_parameter_sharing,soft_parameter_sharing


def get_predictions_on_test(PATH, X_test,y_test, input_size, hidden_size, n_hidden,n_tasks, folder_saving, soft_loss_weight=0):

    train_on_gpu = torch.cuda.is_available()
    print(train_on_gpu)

    model = soft_parameter_sharing.SoftSharing(
        input_size=input_size,
        hidden_size = hidden_size,
        n_hidden = n_hidden,
        n_outputs= n_tasks
    )

    if train_on_gpu:
        model.cuda()

    model.load_state_dict(torch.load(folder_saving+PATH))

    criterion = nn.MSELoss()

    model.eval()

    predictions = {}
    task_specific_test_loss = {}
    count_test = X_test.shape[0]

    X_test = torch.from_numpy(X_test)
    X_test = X_test.float()
    y_test = torch.from_numpy(y_test)
    y_test = y_test.float()

    data, target = X_test, y_test
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

        ##check shapes before concatenating
        predictions[n] = y_pred.detach().cpu().numpy()
        # task_specific_test_loss[n] = loss.item()
        total_loss.append(loss)

    # loss = (sum(total_loss) / len(total_loss))
    loss = (sum(total_loss)) + soft_loss_weight*(soft_loss)
    test_loss = loss.item()


    print("test total loss is: ", str(test_loss))
    print("test soft loss is: ", str(soft_loss))

    return predictions



def get_predictions_with_clustering_on_test(PATH, X_test,y_test, input_size, hidden_sizes,task_specific_hidden_sizes,n_clusters,cluster_labels_test, folder_saving, pretrained_path):

    train_on_gpu = torch.cuda.is_available()
    print(train_on_gpu)
    #
    model = hard_parameter_sharing.Custom_HardSharing(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        n_outputs=n_clusters,
        pretrained_path=pretrained_path,
        task_specific_hidden_sizes = task_specific_hidden_sizes
    )


    if train_on_gpu:
        model.cuda()

    model.load_state_dict(torch.load(folder_saving+PATH))

    # for k, v in list(model.model.hard_sharing.state_dict().items()):
    #     print("Layer {}".format(k))
    #     print(v)

    model.eval()

    count_test = X_test.shape[0]

    X_test = torch.from_numpy(X_test)
    X_test = X_test.float()
    y_test = torch.from_numpy(y_test)
    y_test = y_test.float()

    data, target = X_test, y_test
    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()

    # predicted_clusters = clustering.get_closest_clusters(X_test, kmeans, features_indices_to_cluster_on)
    y_pred = []
    for i in range(count_test):
        cluster_label = cluster_labels_test[i]
        pred = model(data[i].reshape(1, -1))
        y_pred.append(pred[:,cluster_label].item())

    return y_pred





