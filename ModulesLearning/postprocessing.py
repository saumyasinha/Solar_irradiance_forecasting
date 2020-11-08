import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import matplotlib.pyplot as plt

def postprocessing_target(pred, true, X, index_ghi, index_clearghi, lead):
    '''
    postprocess the target clearness index to solar irradiance (unit should be watt/m2)
    '''
    # Calculating the predicted dw_solar by multiplying the predicted clearness index with GHi clear of current time
    true = np.roll(true, lead)
    true = np.reshape(true, (true.shape[0], 1))
    pred = np.roll(pred, lead)
    pred = np.reshape(pred, (pred.shape[0], 1))
    clearsky = np.reshape(X[:, index_clearghi], (X[:, index_clearghi].shape[0], 1))  # clear sky GHI
    y_true = np.multiply(true, clearsky)
    y_pred = np.multiply(pred, clearsky)
    return y_true, y_pred


def smart_persistence_model(X, y, index_clearghi, lead):

    clearness_index = np.asarray(y)
    clearness_index = np.roll(clearness_index, 2*lead) # should be lead*2 since S(t) = clearness_index(t-lead)*clear_ghi(t)
    clearness_index = np.reshape(clearness_index, (clearness_index.shape[0], 1))
    clearghi = np.asarray(X[:, index_clearghi])
    clearghi = np.reshape(clearghi, (clearghi.shape[0], 1))

    y_persistance = np.multiply(clearness_index, clearghi)
    y_persistance = np.reshape(y_persistance, (y_persistance.shape[0], 1))

    return y_persistance


def normal_persistence_model(X, index_ghi, lead):
    ## index_ghi is dw_solar's index
    ghi = X[:,index_ghi]
    pred_ghi = np.roll(ghi, lead)
    pred_ghi = np.reshape(pred_ghi, (pred_ghi.shape[0], 1))

    return pred_ghi


def final_true_pred_sp_np(true, pred, np, sp, lead, X, index_zen, index_clearghi, zenith_threshold=85):

    # X = np.roll(X, lead, axis = 0)
    # X =  X_test[lead:]
    true = true[2*lead:]
    pred = pred[2*lead:]
    np = np[2 * lead:]
    sp = sp[2 * lead:]
    X = X[2 * lead:]
    #
    print((pred<0).sum())
    # print(true.shape, pred.shape, np.shape, sp.shape, X.shape)
    # #
    # # ## remove 0 clear ghi
    # true = true[X[:, index_clearghi] >0]
    # pred= pred[X[:, index_clearghi] >0]
    # np = np[X[:, index_clearghi] >0]
    # sp = sp[X[:, index_clearghi] >0]
    # X = X[X[:, index_clearghi] > 0]
    # #
    # # # ## remove further daytimes
    # true = true[(X[:, index_zen] < zenith_threshold)]
    # pred = pred[X[:, index_zen] < zenith_threshold]
    # np = np[X[:, index_zen] < zenith_threshold]
    # sp = sp[X[:, index_zen] < zenith_threshold]

    #
    # ## remove negative predictions
    true_day_final = true[(pred>=0)]
    pred_day_final = pred[(pred>=0)]
    np_day_final = np[(pred>=0) ]
    sp_day_final = sp[(pred>=0)]


    print(true_day_final.shape, pred_day_final.shape, np_day_final.shape, sp_day_final.shape)

    return true_day_final, pred_day_final, np_day_final, sp_day_final



def evaluation_metrics(true, pred):
    rmse = sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    diff = (true - pred)
    abs_diff = np.abs(true - pred)
    mb = diff.mean()
    # sd = np.std(abs_diff)
    r2 = r2_score(true, pred)
    return rmse, mae, mb, r2


def skill_score(our, persis):
    '''
    calculate the skill-score, that is the improvement of our model over the baseline smart-persistence model
    '''
    skill = persis - our
    skill = (skill * 100) / persis
    return skill


def plot_results(true_day, pred_day, sp_day, lead, season, folder_plots, model):
    # t = np.reshape(true_day, (1, true_day.shape[0]))
    # p = np.reshape(pred_day, (1, pred_day.shape[0]))
    # s = np.reshape(sp_day, (1, sp_day.shape[0]))

    t = true_day.flatten()
    p = pred_day.flatten()
    s = sp_day.flatten()
    x = np.asarray(range(true_day.shape[0]))
    # x = np.reshape(x, (1, x.shape[0]))

    # print(x.shape, t.shape, p.shape, s.shape)
    plt.figure(figsize=(20, 10))
    # plt.plot(x[:, 150:250], t[:, 150:250], 'g<')
    # plt.plot(x[:, 150:250], p[:, 150:250], 'b*')
    # plt.plot(x[:, 150:250], s[:, 150:250], 'r.')
    plt.plot(x, t, label="true values")
    plt.plot(x, p, label="predicted values")
    plt.plot(x, s, label="smart persistence values")
    plt.legend(loc="upper left")
    plt.savefig(folder_plots+"/final_comparison_plots_for_lead"+str(lead)+"_season"+str(season)+"_with_"+str(model))
    return