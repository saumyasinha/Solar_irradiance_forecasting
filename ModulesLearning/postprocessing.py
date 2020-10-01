import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import matplotlib.pyplot as plt

def postprocessing_target(pred, true, X, index_ghi, index_clearghi, lead):
    '''
    postprocess the target clearness index to solar irradiance (unit should be watt/m2)
    '''
    # Calculating the predicted dw_solar by multiplying the predicted clearness index with GHi clear of current time
    true_temp = np.roll(true, lead)
    true_temp = np.reshape(true_temp, (true_temp.shape[0], 1))
    pred_temp = np.roll(pred, lead)
    pred_temp = np.reshape(pred_temp, (pred_temp.shape[0], 1))

    clearsky = np.reshape(X[:, index_clearghi], (X[:, index_clearghi].shape[0], 1))  # clear sky GHI
    y_true = np.multiply(true_temp, clearsky)
    y_pred = np.multiply(pred_temp, clearsky)
    return y_true, y_pred


def smart_persistence_model(X, y, index_clearghi, lead):
    clearnessind = np.asarray(y)  # this is adjusted according to lead
    clearnessind = np.roll(clearnessind, lead + 1)
    clearghi = np.asarray(X[:, index_clearghi])
    clearghi = np.reshape(clearghi, (clearghi.shape[0], 1))
    y_persis = np.multiply(clearnessind, clearghi)
    y_persis = np.reshape(y_persis, (y_persis.shape[0], 1))
    return y_persis


def normal_persistence_model(X, index_ghi, lead):
    ghi = np.asarray(X[:, index_ghi])
    pred_ghi = np.roll(ghi, lead)
    pred_ghi = np.reshape(pred_ghi, (pred_ghi.shape[0], 1))
    return pred_ghi


def check_and_remove_outliers(true_day, pred_day, np_day, sp_day):
    # filter persistent values when 0 (filtering the outliers)
    ind = []
    for i in range(len(sp_day)):
        if true_day[i] <= 0.1 or pred_day[i] <= 0.0:  # or sp_day1[i]<=0.0:
            ind.append(i)
    all_dt = list(range(sp_day.shape[0]))

    # index with no outliers
    final_index = [x for x in all_dt if x not in ind]
    #     print("all, outliers and final", len(all_elt), len(ind), len(final_index))

    # selecting the final tuples for the daytime and without outliers
    true_day_final = []
    pred_day_final = []
    np_day_final = []
    sp_day_final = []
    for dt in final_index:
        true_day_final.append(true_day[dt])
        pred_day_final.append(pred_day[dt])
        np_day_final.append(np_day[dt])
        sp_day_final.append(sp_day[dt])
    true_day_final = np.asarray(true_day_final).astype(np.float)
    pred_day_final = np.asarray(pred_day_final).astype(np.float)
    np_day_final = np.asarray(np_day_final).astype(np.float)
    sp_day_final = np.asarray(sp_day_final).astype(np.float)
    return true_day_final, pred_day_final, np_day_final, sp_day_final


def select_pred_daytimes_and_remove_outliers(y_true, y_pred, y_np, y_sp, index_daytimes):
    '''
    select the day times values only, and remove outliers
    '''
    true_day = []
    pred_day = []
    np_day = []
    sp_day = []
    for dt in index_daytimes:
        true_day.append(y_true[dt])
        pred_day.append(y_pred[dt])
        np_day.append(y_np[dt])
        sp_day.append(y_sp[dt])
    true_day = np.asarray(true_day).astype(np.float)
    pred_day = np.asarray(pred_day).astype(np.float)
    np_day = np.asarray(np_day).astype(np.float)
    sp_day = np.asarray(sp_day).astype(np.float)

    # filter persistent values when 0 (filtering the outliers)
    ind = []
    for i in range(len(sp_day)):
        if sp_day[i] <= 0.0 or true_day[i] <= 0.0 or pred_day[i] <= 0.0 or np_day[i] <= 0.0 or pred_day[i] >= 1050.0:
            ind.append(i)
    #     print("shape of all the pred: ",y_true.shape, y_pred.shape,y_np.shape, y_sp.shape)
    all_dt = list(range(sp_day.shape[0]))

    # indices with no outliers
    final_index = [x for x in all_dt if x not in ind]

    # selecting the final tuples for the daytime and without outliers
    true_day_final = []
    pred_day_final = []
    np_day_final = []
    sp_day_final = []
    for dt in final_index:
        true_day_final.append(true_day[dt])
        pred_day_final.append(pred_day[dt])
        np_day_final.append(np_day[dt])
        sp_day_final.append(sp_day[dt])

    true_day_final = np.asarray(true_day_final).astype(np.float)
    pred_day_final = np.asarray(pred_day_final).astype(np.float)
    np_day_final = np.asarray(np_day_final).astype(np.float)
    sp_day_final = np.asarray(sp_day_final).astype(np.float)
    return true_day_final, pred_day_final, np_day_final, sp_day_final


def evaluation_metrics(true, pred):
    rmse = sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    diff = (true - pred)
    abs_diff = np.abs(true - pred)
    mb = diff.mean()
    sd = np.std(abs_diff)
    return rmse, mae, mb, sd


def skill_score(our, persis):
    '''
    calculate the skill-score, that is the improvement of our model over the baseline smart-persistence model
    '''
    skill = persis - our
    skill = (skill * 100) / persis
    return skill


def plot_results(true_day, pred_day, sp_day):
    t = np.reshape(true_day, (1, true_day.shape[0]))
    p = np.reshape(pred_day, (1, pred_day.shape[0]))
    s = np.reshape(sp_day, (1, sp_day.shape[0]))

    x = np.asarray(range(true_day.shape[0]))
    x = np.reshape(x, (1, x.shape[0]))

    print(x.shape, t.shape, p.shape, s.shape)
    plt.figure(figsize=(20, 10))
    plt.plot(x[:, 150:250], t[:, 150:250], 'g<')
    plt.plot(x[:, 150:250], p[:, 150:250], 'b*')
    plt.plot(x[:, 150:250], s[:, 150:250], 'r.')
    plt.show()
    return