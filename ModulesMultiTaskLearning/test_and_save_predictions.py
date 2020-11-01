import torch
import torch.nn as nn
import numpy as np
from SolarForecasting.ModulesMultiTaskLearning import hard_parameter_sharing



def get_predictions_on_test(PATH, X_test,y_test, input_size, hidden_size, n_hidden,n_tasks, folder_saving):

    train_on_gpu = torch.cuda.is_available()
    print(train_on_gpu)

    model = hard_parameter_sharing.HardSharing(
        input_size=input_size,
        hidden_size=hidden_size,
        n_hidden=n_hidden,
        n_outputs=n_tasks)

    if train_on_gpu:
        model.cuda()

    model.load_state_dict(torch.load(folder_saving+PATH))

    criterion = nn.MSELoss()

    model.eval()

    predictions = {}
    task_specific_test_loss = {}

    X_test = torch.from_numpy(X_test)
    X_test = X_test.float()
    y_test = torch.from_numpy(y_test)
    y_test = y_test.float()

    data, target = X_test, y_test
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

        ##check shapes before concatenating
        predictions[n] = y_pred.detach().cpu().numpy()
        task_specific_test_loss[n] = loss.item()
        total_loss.append(loss)

    loss = sum(total_loss) / len(total_loss)
    # update validation loss
    test_loss = loss.item()


    print("test loss is: ", str(test_loss))

    return predictions




