import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import matplotlib.pyplot as plt

def postprocessing_target(pred, true): #, X, index_ghi, index_clearghi, lead):
    '''
    postprocess the target clearness index to solar irradiance (unit should be watt/m2)
    '''
    # Calculating the predicted dw_solar by multiplying the predicted clearness index with GHi clear of current time
    true_clearness= true[:,0] #np.roll(true, lead)
    true_clearness = np.reshape(true_clearness, (true_clearness.shape[0], 1))
    pred = pred #np.roll(pred, lead)
    pred = np.reshape(pred, (pred.shape[0], 1))
    clearsky = true[:,1]#np.reshape(X[:, index_clearghi], (X[:, index_clearghi].shape[0], 1))  # clear sky GHI
    clearsky = np.reshape(clearsky, (clearsky.shape[0], 1))
    y_true = np.multiply(true_clearness, clearsky)
    y_pred = np.multiply(pred, clearsky)
    return y_true, y_pred


def smart_persistence_model(X, true, index_clearness, lead):

    clearness_index = X[:,index_clearness,-1]
    print(clearness_index.shape,clearness_index.shape[0] )
    # clearness_index = np.roll(clearness_index, lead) #since S(t) = clearness_index(t-lead)*clear_ghi(t)
    clearness_index = np.reshape(clearness_index, (clearness_index.shape[0], 1))
    clearsky = true[:, 1]  # np.reshape(X[:, index_clearghi], (X[:, index_clearghi].shape[0], 1))  # clear sky GHI
    clearsky = np.reshape(clearsky, (clearsky.shape[0], 1))

    y_persistance = np.multiply(clearness_index, clearsky)
    # y_persistance = np.reshape(y_persistance, (y_persistance.shape[0], 1))

    return y_persistance


# def normal_persistence_model(X, index_ghi, lead):
#     ## index_ghi is dw_solar's index
#     ghi = X[:,index_ghi]
#     pred_ghi = np.roll(ghi, lead)
#     pred_ghi = np.reshape(pred_ghi, (pred_ghi.shape[0], 1))
#
#     return pred_ghi

# def climatology_baseline(X_test, df_2017, col_to_indices_mapping, n_features):
#
#     index_month = -n_features+col_to_indices_mapping['month']
#     index_hour = -n_features+col_to_indices_mapping['hour']
#
#     current_month_list = X_test[:,index_month]
#     current_hour_list = X_test[:,index_hour]
#
#     climatology_baseline=[]
#     for i in range(len(X_test)):
#         current_month = current_month_list[i]
#         current_hour = current_hour_list[i]
#         # print(current_hour,current_month)
#         X_2017_for_month_hour = df_2017[(df_2017.month == current_month) & (df_2017.hour == current_hour)]
#
#         climatology_baseline.append(np.average(X_2017_for_month_hour.dw_solar.values))
#
#     climatology_baseline = np.reshape(climatology_baseline,(len(climatology_baseline),1))
#     print(np.sum(np.isnan(climatology_baseline)))
#     return climatology_baseline





## This function is not needed anymore
def final_true_pred_sp_np(true, pred, np, sp, climatology,CH_PeEN, lead, index_zen, index_clearghi, zenith_threshold=85):


    true = true[2*lead:]
    pred = pred[2*lead:]
    np = np[2 * lead:]
    sp = sp[2 * lead:]
    climatology = climatology[2 * lead:]
    CH_PeEN = CH_PeEN[2 * lead:]

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
    true_day_final = true #[(pred>=0)]
    pred_day_final = pred #[(pred>=0)]
    np_day_final = np #[(pred>=0) ]
    sp_day_final = sp #[(pred>=0)]
    climatology_final = climatology  # [(pred>=0)]
    CH_PeEN_final = CH_PeEN  # [(pred>=0)]


    print(true_day_final.shape, pred_day_final.shape, np_day_final.shape, sp_day_final.shape, climatology_final.shape)

    return true_day_final, pred_day_final, np_day_final, sp_day_final, climatology_final, CH_PeEN_final



def evaluation_metrics(true_arr, pred):

    # true_arr = true_arr[true_arr[:,1] > 0]
    # true_arr = true_arr[true_arr[:,2] < 85]
    true = true_arr[:,0]

    rmse = sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    squared_diff = (true - pred)**2
    # abs_diff = np.abs(true - pred)
    # mb = diff.mean()
    # # sd = np.std(abs_diff)
    mb = np.mean(squared_diff)
    sd = np.std(squared_diff)
    r2 = r2_score(true, pred)
    # plt.clf()
    # plt.hist(squared_diff, density=True, bins=30)
    # plt.savefig(path)
    return rmse, mae, mb, sd, r2


def skill_score(our, persis):
    '''
    calculate the skill-score, that is the improvement of our model over the baseline smart-persistence model
    '''
    skill = persis - our
    skill = (skill * 100) / persis
    return skill


# def plot_results(true_day, pred_day, sp_day, lead, season, folder_plots, model):
#     # t = np.reshape(true_day, (1, true_day.shape[0]))
#     # p = np.reshape(pred_day, (1, pred_day.shape[0]))
#     # s = np.reshape(sp_day, (1, sp_day.shape[0]))
#
#     t = true_day.flatten()
#     p = pred_day.flatten()
#     s = sp_day.flatten()
#     x = np.asarray(range(true_day.shape[0]))
#     # x = np.reshape(x, (1, x.shape[0]))
#
#     # print(x.shape, t.shape, p.shape, s.shape)
#     plt.figure(figsize=(20, 10))
#     # plt.plot(x[:, 150:250], t[:, 150:250], 'g<')
#     # plt.plot(x[:, 150:250], p[:, 150:250], 'b*')
#     # plt.plot(x[:, 150:250], s[:, 150:250], 'r.')
#     plt.plot(x, t, label="true values")
#     plt.plot(x, p, label="predicted values")
#     plt.plot(x, s, label="smart persistence values")
#     plt.legend(loc="upper left")
#     plt.savefig(folder_plots+"/final_comparison_plots_for_lead"+str(lead)+"_season"+str(season)+"_with_"+str(model))
#     return