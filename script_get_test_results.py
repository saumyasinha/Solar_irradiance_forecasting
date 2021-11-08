import numpy as np
import pandas as pd
import os
import pickle
# from sklearn.externals import joblib
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from datetime import timedelta
import scipy as sp
from SolarForecasting.ModulesLearning import ml_models as models
from SolarForecasting.ModulesProcessing import collect_data, clean_data
from SolarForecasting.ModulesLearning import preprocessing as preprocess
from SolarForecasting.ModulesLearning import postprocessing as postprocess
from SolarForecasting.ModulesLearning.ModuleLSTM import train as tranformers
from SolarForecasting.ModulesLearning.ModulesCNN import train as cnn
from SolarForecasting.ModulesLearning.ModulesCNN.Model import crps_score
from matplotlib import cm

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# city
city = 'Sioux_Falls_SD'#'Boulder_CO'#'Goodwin_Creek_MS'#'Desert_Rock_NV'

# lead time
lead_times = [24*7] #7days in advance

# season
seasons =['year']
res = '1hour' #15min

# file locations
path_local = "/Users/saumya/Desktop/SolarProject/"
path_cluster = "/pl/active/machinelearning/Solar_forecasting_project/"
path_project = path_local
path = path_project+"Data/"
folder_saving = path_project + city+"/Models/"
folder_plots = path_project + city+"/Plots/"
clearsky_file_path = path+'clear-sky/'+city+'_1min_original.csv'


# scan all the features
features = ['year','month','day','hour','min','zen','dw_solar','uw_solar','direct_n','diffuse','dw_ir','dw_casetemp','dw_dometemp','uw_ir','uw_casetemp','uw_dometemp','uvb','par','netsolar','netir','totalnet','temp','rh','windspd','winddir','pressure']

## ## selected features for the study
final_features = ['year','month','day','hour','zen','dw_solar','uw_solar','direct_n','dw_ir','uw_ir','temp','rh','windspd','winddir','pressure', 'clear_ghi']
ensmeble_col_list = ["ensemble_" + str(ensemble_num) for ensemble_num in range(51)]


# target or Y
target_feature = 'clearness_index'

# start and end month+year
startyear = 2016
endyear = 2018
startmonth = 1
endmonth = 12

testyear = 2018

# hyperparameters
n_timesteps = 12 #1day (can be 12hrs or 48hrs for a few models)
n_features = 16 + 51
quantile =False #False

#hyperparameters for LSTM/Transformer/CNNs
n_layers = 1 #2 #3
num_heads = 2 #4
d_model = 64 #128

hidden_size= 50
batch_size = 16 #32 #16 #16

epochs = 300 #250
lr = 1e-4 #1e-5 #1e-4

alphas = np.arange(0.05, 1.0, 0.05) #range of alpha (for quantile forecast)
q50 = 9  #median index in 'alphas'


def get_data():

    ## collect raw data
    years = [2015, 2016, 2017, 2018] #[2005, 2006, 2007, 2008, 2009]
    object = collect_data.SurfradDataCollector(years, [city], path)

    object.download_data()


    ## cleanse data to get processed version
    for year in years:
        object = clean_data.SurfradDataCleaner(city, year, path)
        object.process(path_to_column_names='ModulesProcessing/column_names.pkl')


def include_previous_features(X, col_to_indices_mapping):
    '''
    Features at time t includes features from previous n_timesteps
    '''
    # y_list = []
    # previous_time_periods = list(range(1, n_timesteps+1))
    # # indices_except_ensmebles = sorted([feature_ind for feature, feature_ind in col_to_indices_mapping.items() if not feature.startswith('ensemble')])
    # # print(indices_except_ensmebles)
    # # Xsub = X[:,indices_except_ensmebles]
    #
    # for l in previous_time_periods:
    #     # print("rolling by: ", l)
    #     X_train_shifted = np.roll(X, l, axis=0)
    #
    #     y_list.append(X_train_shifted)
    #     # y_list.append(dw_solar_rolled)
    # y_list = y_list[::-1]
    #
    # previous_time_periods_columns = np.column_stack(y_list)
    # X = np.column_stack([previous_time_periods_columns, X])
    #
    # print("X shape after adding prev features: ", X.shape)
    # return X

    X = np.expand_dims(X, axis=2)
    # indices_except_ensmebles = sorted(
    #     [feature_ind for feature, feature_ind in col_to_indices_mapping.items() if not feature.startswith('ensemble')])
    # # print(X.shape)
    X_with_prev_timesteps = []
    data_size = X.shape[0]
    for i in range(n_timesteps,data_size):
        prev_list =[]
        for j in range(n_timesteps,0,-1):
            prev = X[i-j,:,:]
            # prev_sub = prev[indices_except_ensmebles,:]
            # print(prev.shape)
            prev_list.append(prev)

        curr = X[i,:,:]
        prev_list.append(curr)
        temp = np.concatenate(prev_list, axis=1)

        # print(temp.shape)
        X_with_prev_timesteps.append(temp)

    # print(len(X_with_prev_timesteps))

    X = np.array(X_with_prev_timesteps)
    # print(X.shape)
    return X


def include_nwp_features(df):

    missing_hour_dates = {2016:[263,265], 2017:[283], 2018:[144]}

    fnwp = path + "NWP/" + city + "/"

    #Alreadt sorted
    # df = dfraw.sort_values(by=['year','month','day','hour'])
    # print("inside nwp")
    # print(dfraw.equals(df))



    for ensemble in ensmeble_col_list:
        df[ensemble] = ""

    nwp_new_columns = []
    for year in [2016,2017,2018]:
        print(len(df[df['year'] == year]))
        year_list = []
        fnwp_yr = fnwp+"nwp_"+str(year)+".npy"
        nwp_yr = np.load(fnwp_yr)
        nwp_yr = nwp_yr/(168*3600)
        days = nwp_yr.shape[0]
        for day in range(days):
            if day not in missing_hour_dates[year]:
                nwp_day = nwp_yr[day]
                print(nwp_day.shape)
                x_1_12 = np.tile(nwp_day[0], (12,1))
                x_13_24 = np.tile(nwp_day[1], (12,1))
                x_day = np.concatenate((x_1_12, x_13_24), axis=0)
                print(x_day.shape)
                year_list.append(x_day)

        year_nwp = np.concatenate(year_list)
        print(year_nwp)
        print(year_nwp.shape)
        nwp_new_columns.append(year_nwp)

    nwp_new_columns = np.concatenate(nwp_new_columns)
    print(np.max(nwp_new_columns), np.min(nwp_new_columns))

    df.loc[:,ensmeble_col_list] = nwp_new_columns


    return df



def get_crps_for_ngboost(model, X, y):
    '''
    CRPS is the evaluation metric to be used for probabilsitic forecasting
    '''

    params = model.pred_dist(X)._params
    loc = params[0]
    scale = np.exp(params[1])
    Z = (y - loc) / scale


    score = scale * (
            Z * (2 * sp.stats.norm.cdf(Z) - 1)
            + 2 * sp.stats.norm.pdf(Z)
            - 1 / np.sqrt(np.pi)
    )


    return np.average(score)


def get_crps_for_ngboost_scaled(model, X, y,X_before_normalized,index_clearghi,lead):
    '''
    CRPS is the evaluation metric to be used for probabilsitic forecasting
    '''

    params = model.pred_dist(X)._params
    loc = params[0]
    scale = np.exp(params[1])

    clearsky = X_before_normalized[:, index_clearghi]
    y = np.roll(y, lead)
    loc = np.roll(loc, lead)
    scale = np.roll(scale, lead)

    print(clearsky.shape, y.shape, loc.shape, scale.shape)
    y = np.multiply(y, clearsky)
    loc = np.multiply(loc, clearsky)
    scale = np.multiply(scale, clearsky)

    Z = (y - loc) / scale



    score = scale * (
            Z * (2 * sp.stats.norm.cdf(Z) - 1)
            + 2 * sp.stats.norm.pdf(Z)
            - 1 / np.sqrt(np.pi)
    )
    print(score.shape)
    score = score[2 * lead:]

    return np.average(score)



def get_CH_PeEN_baseline_crps(X_test, X_train, col_to_indices_mapping, y_test,lead):
    index_hour = -n_features + col_to_indices_mapping['hour']
    index_clearghi = -n_features + col_to_indices_mapping['clear_ghi']
    index_ghi = -n_features + col_to_indices_mapping['dw_solar']

    current_hour_list = X_test[:, index_hour]

    CH_PeEN_baseline = []
    CH_PeEN_point_baseline = []

    for i in range(len(X_test)):
        current_hour = current_hour_list[i]
        # print(current_hour,current_month)
        X_train_for_hour = X_train[X_train[:,index_hour]==current_hour]

        hour_dist =X_train_for_hour[:, index_ghi]/X_train_for_hour[:, index_clearghi]

        CH_PeEN_point_baseline.append(np.average(hour_dist))

        quantiles = np.quantile(hour_dist, alphas)
        CH_PeEN_baseline.append(quantiles)

    CH_PeEN_baseline_quantiles = np.asarray(CH_PeEN_baseline)
    CH_PeEN_point_baseline = np.reshape(CH_PeEN_point_baseline, (len(CH_PeEN_point_baseline), 1))

    clearghi = np.asarray(X_test[:, index_clearghi])
    clearghi = np.reshape(clearghi, (clearghi.shape[0], 1))

    CH_PeEN_baseline_quantiles = np.multiply(CH_PeEN_baseline_quantiles, clearghi)
    CH_PeEN_point_baseline = np.multiply(CH_PeEN_point_baseline, clearghi)

    crps_CH_PeEN = crps_score(CH_PeEN_baseline_quantiles, y_test, alphas, post_process=True, lead=lead)


    return CH_PeEN_point_baseline, CH_PeEN_baseline_quantiles, crps_CH_PeEN

def get_climatology_baseline_crps(X_test, X_train, col_to_indices_mapping, y_test,lead):


    # GHIs_train = X_train[:,index_ghi]
    #
    # climatology_point_baseline = np.average(GHIs_train)
    # climatology_point_baseline = np.full(y_test.shape, climatology_point_baseline)
    #
    #
    # quantiles = np.quantile(GHIs_train, alphas)
    # climatology_probabilistic_baseline = np.full((y_test.shape[0], len(alphas)), quantiles)
    #
    #
    # crps_climatology = crps_score(climatology_probabilistic_baseline, y_test, alphas, post_process=True, lead=lead)
    #
    # return climatology_point_baseline, climatology_probabilistic_baseline, crps_climatology

    index_hour = -n_features + col_to_indices_mapping['hour']
    index_ghi = -n_features + col_to_indices_mapping['dw_solar']

    current_hour_list = X_test[:, index_hour]

    climatology_baseline = []
    climatology_point_baseline = []

    for i in range(len(X_test)):
        current_hour = current_hour_list[i]
        # print(current_hour,current_month)
        X_train_for_hour = X_train[X_train[:, index_hour] == current_hour]

        hour_dist = X_train_for_hour[:, index_ghi]

        climatology_point_baseline.append(np.average(hour_dist))

        quantiles = np.quantile(hour_dist, alphas)
        climatology_baseline.append(quantiles)

    climatology_baseline_quantiles = np.asarray(climatology_baseline)
    climatology_point_baseline = np.reshape(climatology_point_baseline, (len(climatology_point_baseline), 1))

    crps_climatology = crps_score(climatology_baseline_quantiles, y_test, alphas, post_process=True, lead=lead)

    return climatology_point_baseline, climatology_baseline_quantiles, crps_climatology


def reliability_diagrams(alphas, outputs, outputs_ch_pn, outputs_climatology, target, folder_saving):


    outputs_tcn_attn = np.load(
        "/Users/saumya/Desktop/SolarProject/Boulder_CO/Models/year/ML_models_2018/cnn/1hour/final_tcn_week_ahead_1days_lag_small_kernel_1hr_res_attn_quantile/tcn_attention.npy")
    outputs_transformers = np.load(
        "/Users/saumya/Desktop/SolarProject/Boulder_CO/Models/year/ML_models_2018/cnn/1hour/final_transformers_2day_2heads_less_dmodel_lag_week_ahead_1hr_res_quantile/transformer.npy")
    print(outputs_tcn_attn.shape, outputs_transformers.shape)

    ## had to add the following for boulder, since timesteps were different for tcn/transformers
    # target = target[24:, :]
    total_obs = len(target)
    # outputs = outputs[24:,:]
    # outputs_tcn_attn = outputs_tcn_attn[24:,:]
    # outputs_ch_pn = outputs_ch_pn[24:,:]
    # outputs_climatology = outputs_climatology[24:,:]
    # print(outputs_tcn_attn.shape, outputs_transformers.shape)


    obs_proportion_list = []
    obs_proportion_list_chpn = []
    obs_proportion_list_climatology = []
    obs_proportion_list_tcn_attn =[]
    obs_proportion_list_transformers = []
    for i, alpha in zip(range(len(alphas)), alphas):
        output = outputs[:, i].reshape((-1, 1))
        output_ch_pn = outputs_ch_pn[:, i].reshape((-1, 1))
        output_climatology = outputs_climatology[:, i].reshape((-1, 1))
        output_transformers = outputs_transformers[:, i].reshape((-1, 1))
        output_tcn_attn = outputs_tcn_attn[:, i].reshape((-1, 1))

        obs_proportion_list.append(np.sum(target<=output)/total_obs)
        obs_proportion_list_chpn.append(np.sum(target <= output_ch_pn) / total_obs)
        obs_proportion_list_climatology.append(np.sum(target <= output_climatology) / total_obs)
        obs_proportion_list_tcn_attn.append(np.sum(target <= output_tcn_attn) / total_obs)
        obs_proportion_list_transformers.append(np.sum(target <= output_transformers) / total_obs)

    print(obs_proportion_list == obs_proportion_list_climatology)

    fig, ax = plt.subplots()
    # ax.plot(alphas, alphas)
    ax.plot(alphas, obs_proportion_list, marker='o', color ='g',label="TCN")
    ax.plot(alphas, obs_proportion_list_tcn_attn, marker='d', color='y', label="TCN_with_attention")
    ax.plot(alphas, obs_proportion_list_transformers, marker='h', color='m', label="Transformers")
    ax.plot(alphas, obs_proportion_list_chpn, marker="s", color ='b',label="CH_PeEN")
    ax.plot(alphas, obs_proportion_list_climatology, marker="*",color ='k', label="Climatology")
    ax.set_xlabel('Nominal proportion')
    ax.set_ylabel('Observed proportion')
    ax.legend(loc='upper left')


    plt.savefig(folder_saving + "reliability_with_all_plots")
    plt.close()


def sharpness_diagrams(alphas, outputs, outputs_ch_pn, outputs_climatology,folder_saving):
    outputs_tcn_attn = np.load(
        "/Users/saumya/Desktop/SolarProject/Boulder_CO/Models/year/ML_models_2018/cnn/1hour/final_tcn_week_ahead_1days_lag_small_kernel_1hr_res_attn_quantile/tcn_attention.npy")
    outputs_transformers = np.load(
        "/Users/saumya/Desktop/SolarProject/Boulder_CO/Models/year/ML_models_2018/cnn/1hour/final_transformers_2day_2heads_less_dmodel_lag_week_ahead_1hr_res_quantile/transformer.npy")


    print(outputs_tcn_attn.shape, outputs_transformers.shape)


    intervals = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    sharpness = []
    sharpness_ch_pn = []
    sharpness_climatology = []
    sharpness_tcn_attn = []
    sharpness_transformers = []
    for i in intervals:

        h = 0.5+(i/2)
        l = 0.5-(i/2)

        h = np.where(np.around(alphas,2) == np.around(h,2))[0][0]
        l = np.where(np.around(alphas,2) == np.around(l,2))[0][0]

        sharpness.append(np.mean(outputs[:,h] - outputs[:,l]))
        sharpness_ch_pn.append(np.mean(outputs_ch_pn[:, h] - outputs_ch_pn[:, l]))
        sharpness_climatology.append(np.mean(outputs_climatology[:, h] - outputs_climatology[:, l]))
        sharpness_tcn_attn.append(np.mean(outputs_tcn_attn[:, h] - outputs_tcn_attn[:, l]))
        sharpness_transformers.append(np.mean(outputs_transformers[:, h] - outputs_transformers[:, l]))


    print(sharpness==sharpness_climatology)

    fig, ax = plt.subplots()
    # plt.plot(alphas, alphas)
    ax.plot(intervals, sharpness, marker='o', color ='g',label = "TCN")
    ax.plot(intervals, sharpness_tcn_attn, marker='d', color='y', label="TCN_with_attention")
    ax.plot(intervals, sharpness_transformers, marker='h', color='m', label="Transformers")
    ax.plot(intervals, sharpness_ch_pn, marker="s", color ='b',label="CH_PeEN")
    ax.plot(intervals, sharpness_climatology, marker="*", color ='k',label=" Hourly Climatology")
    ax.set_xlabel('intervals')
    ax.set_ylabel('Average width')
    ax.legend(loc = 'upper left')

    plt.savefig(folder_saving + "sharpness_with_all_plots")
    plt.close()


def probabilistic_prediction_plots(X_test,outputs,y_test, col_to_indices_mapping,folder_saving):
    # outputs_tcn_attn = np.load(
    #     "/Users/saumya/Desktop/SolarProject/Boulder_CO/Models/year/ML_models_2018/cnn/1hour/final_tcn_week_ahead_1days_lag_small_kernel_1hr_res_attn_quantile/tcn_attention.npy")
    # outputs_tcn = np.load(
    #     "/Users/saumya/Desktop/SolarProject/Boulder_CO/Models/year/ML_models_2018/cnn/1hour/final_tcn_week_ahead_1days_lag_small_kernel_1hr_res_quantile/tcn.npy")
    # outputs_lstm = np.load(
    #     "/Users/saumya/Desktop/SolarProject/Boulder_CO/Models/year/ML_models_2018/cnn/1hour/final_lstm_50_week_ahead_2day_lag_1hr_res_quantile/lstm.npy")
    #
    # sub_outputs_tcn_attn = outputs_tcn_attn[7*24:8*24,:]
    # sub_outputs_tcn = outputs_tcn[7*24:8*24,:]
    # sub_outputs_lstm = outputs_lstm[6 * 24:7 * 24, :]
    index_hour = -n_features + col_to_indices_mapping['hour']
    index_month = -n_features + col_to_indices_mapping['month']
    print(X_test[X_test[:,index_month]==3][:,index_hour])
    print(X_test[:,index_month]==3)
    index_ghi = -n_features + col_to_indices_mapping['dw_solar']
    true = y_test[X_test[:,index_month]==3][:10*3,:] #X_test[X_test[:,index_month]==3][:10*5,index_ghi]
    sub_outputs = outputs[X_test[:,index_month]==3][:10*3,:]
    true  = np.reshape(true,-1)
    print(sub_outputs.shape, true.shape)

    true_full_day=[]
    for i in range(3):
        true_before = [0 for _ in range(13)]
        print(true_before+true[10*i:10*(i+1)].tolist()+[0])
        true_full_day.extend(true_before+true[10*i:10*(i+1)].tolist()+[0])




    x_full_day = list(range(len(true_full_day)))
    # x_only_day = x_full_day[13:23] + x_full_day[24+13:24+23] + x_full_day[48+13:48+23]
    # x_only_day = list(range(13,23)) + list(range(24+13,24+23)) + list(range(48+13,48+23))
    # print(x_full_day,x_only_day)


    # x = range(len(true))
    fig, ax = plt.subplots(1)
    ax.plot(x_full_day, true_full_day, lw=1, label='true', color='g')
    ax.plot(x_full_day[13:23], sub_outputs[:10,9], lw=1, label='predicted median', color='blue')
    ax.plot(x_full_day[24+13:24+23], sub_outputs[10:20, 9], lw=1, color='blue')
    ax.plot(x_full_day[48+13:48+23], sub_outputs[20:30, 9], lw=1, color='blue')
    # ax.fill_between(x, sub_outputs[:,0],sub_outputs[:,-1], facecolor='gray', alpha=0.5,
    #                 label='prediction')

    total_quantiles = sub_outputs.shape[1]
    for i in range(int(total_quantiles/2)):
        # label = ""
        print(i,total_quantiles-i-1)
        print(alphas[i], alphas[total_quantiles - i - 1])
        c = plt.cm.Blues(0.2 + .6 * (float(i) / total_quantiles * 2))
        ax.fill_between(x_full_day[13:23], sub_outputs[:10,i], sub_outputs[:10,total_quantiles-i-1], color=c)
        ax.fill_between(x_full_day[24+13:24+23], sub_outputs[10:20, i], sub_outputs[10:20, total_quantiles - i - 1], color=c)
        ax.fill_between(x_full_day[48+13:48+23], sub_outputs[20:30, i], sub_outputs[20:30, total_quantiles - i - 1], color=c)

    # plt.legend(framealpha=1)
    # plt.show()

    # ax.fill_between(x_test[6 * 24:7 * 24], sub_outputs_lstm[:, 0], sub_outputs_lstm[:, -1], facecolor='green', alpha=0.5,
    #                 label='prediction lstm')
    # ax.fill_between(x_test[6 * 24:7 * 24], sub_outputs_tcn[:, 0], sub_outputs_tcn[:, -1], facecolor='yellow', alpha=0.5,
    #                 label='prediction tcn ')
    # ax.fill_between(x_test[6 * 24:7 * 24], sub_outputs_tcn_attn[:, 0], sub_outputs_tcn_attn[:, -1], facecolor='red', alpha=0.5,
    #                 label='prediction tcn+attention')
    ax.set_xticks([])
    ax.legend(loc='upper left')
    ax.set_xlabel('Time: 3 days in March 2018')
    ax.set_ylabel('Irradiance')

    plt.savefig(folder_saving + "probabilistic_prediction_best_plots_include_night")
    plt.close()

def main():


    # # pre-processing steps
    #
    # # extract the input data files (SURFAD data)
    # processed_file_path = path + 'processed/' + city
    # if not os.path.isdir(processed_file_path):
    #     get_data()
    # combined_csv = preprocess.extract_frame(processed_file_path)
    # print("The columns of the initial data file: ", combined_csv.columns)
    #
    # # extract the features from the input
    # dataset = combined_csv[features]
    # print('dataset size: ',len(dataset))
    #
    # #1hour resolution
    # dataset['MinFlag'] = dataset['min'].apply(preprocess.generateFlag)
    # # dataset = dataset.groupby(['year', 'month', 'day', 'hour', 'MinFlag']).mean()
    # dataset = dataset.groupby(['year', 'month', 'day', 'hour']).mean()
    # dataset.reset_index(inplace=True)
    # print('dataset size: ',len(dataset))
    #
    # print(dataset.isnull().values.any())
    #
    # ##checking month-wise hour counts per year (turns out we don't have every hour of every day!!
    # # df = dataset[dataset['year']==2016]
    # # print(df.groupby(['month', 'day'])['hour'].count())
    # # (based on:
    # # 2016: Sep 20: only 15 hrs (244+20)
    # # Sep 22: only 9 hrs(244 + 22)
    # # 2017:Oct 11: only 14 hours(273 + 11)
    # # 2018: April 25: only 15 hours) we drop the follwoing indices
    # dataset = dataset.drop(dataset[(dataset['year']==2016) & (dataset['month']==9) & (dataset['day']==20)].index)
    # dataset = dataset.drop(
    #     dataset[(dataset['year'] == 2016) & (dataset['month'] == 9) & (dataset['day'] == 22)].index)
    # dataset = dataset.drop(
    #     dataset[(dataset['year'] == 2017) & (dataset['month'] == 10) & (dataset['day'] == 11)].index)
    # dataset = dataset.drop(
    #     dataset[(dataset['year'] == 2018) & (dataset['month'] == 4) & (dataset['day'] == 25)].index)
    #
    # # df = dataset[dataset['year'] == 2016]
    # # print(df.groupby(['month', 'day'])['hour'].count())
    # dataset.reset_index(inplace=True)
    # print('dataset size: ', len(dataset))
    #
    # # read the clear-sky values
    # clearsky = pd.read_csv(clearsky_file_path, skiprows=37, delimiter=';')
    # print("The columns of the clear sky file: ", clearsky.columns)
    # #
    # # divide the observation period in form of year, month, day, hour, min (adding them as variables)
    # clearsky[['year', 'month', 'day', 'hour', 'min']] = clearsky['# Observation period'].apply(preprocess.extract_time)
    # clearsky = clearsky.groupby(['year', 'month', 'day', 'hour']).mean()
    # # clearsky['MinFlag'] = clearsky['min'].apply(preprocess.generateFlag)
    # # clearsky = clearsky.groupby(['year', 'month', 'day', 'hour', 'MinFlag']).mean()
    # print("clearsky rows before merging: ", len(clearsky))
    #
    # # merging the clear sky values with SURFAD input dataset
    # # df = dataset.merge(clearsky, on=['year', 'month', 'day', 'hour', 'MinFlag'], how='inner')
    # df = dataset.merge(clearsky, on=['year', 'month', 'day', 'hour'], how='inner')
    #
    # # renaming the clear sky GHI
    # df = df.rename(columns={'Clear sky GHI': 'clear_ghi'})
    # print("stats of all raw features/columns")
    # print(df.describe())
    #
    #
    # # selecting only the required columns
    # df = df[final_features]
    #
    # # get dataset for the study period
    # df = preprocess.extract_study_period(df,startmonth, startyear, endmonth, endyear)
    # print("\n\n after extracting study period")
    # df.reset_index(drop=True, inplace=True)
    # print(df.tail)
    #
    #
    # ## Add NWP features
    # df = include_nwp_features(df)
    # # print("ssrd outliers: ", min(df), max(df))
    # print(df.describe())
    #
    # ## convert negatives to 0 for all features
    # df[df<0] = 0
    # print("stats of selected features/columns after converting all negatives to 0")
    # print(df.describe())
    #
    # ## adding the clearness index (taking care of "dive-by-zero")
    # print("0 clear GHIs or GHI", len(df[df.clear_ghi==0]), len(df[df.dw_solar==0]))
    # # df['clearness_index'] = df['dw_solar'] / df['clear_ghi']
    # df['clearness_index'] = df['dw_solar'].div(df['clear_ghi'])
    # df[~np.isfinite(df)] = 0
    # df[np.isnan(df)] = 0
    # print("checking for infinity/nans")
    # print(np.isinf(df).values.sum())
    # print(np.isnan(df).values.sum())
    # # # #
    processed_file_path = path + 'processed/' + city + "/"
    # df.to_pickle(processed_file_path + "final_data_At_" + res + "_resolution_2016-2018_updated.pkl")
    df = pd.read_pickle(processed_file_path + "final_data_At_" + res + "_resolution_2016-2018_updated.pkl") ##this files inlcudes the day times - I'll be dropping them later!!

    final_features.extend(ensmeble_col_list)
    ## name of the regression model (this helps to distinguish the models within a "CNN"/"LSTM"/"Transformer" super-folder)
    reg = "tcn_week_ahead_12_seq_lag_small_kernel_1hr_res"
    #reg = "lstm_50_week_ahead_2day_lag_1hr_res_quantile"
    #reg = "transformers_2day_2heads_less_dmodel_lag_week_ahead_1hr_res"

    for season_flag in seasons:
        ## ML_models_2018 is the folder to save results on testyear 2018
        os.makedirs(
            folder_saving + season_flag + "/final_ML_models_"+str(testyear)+"/cnn/"+str(res)+"/"+reg+"/",
            exist_ok=True)
        f = open(folder_saving + season_flag + "/final_ML_models_"+str(testyear)+"/cnn/"+str(res)+"/"+reg+"/results.txt", 'a')

        for lead in lead_times:
            # create dataset with lead
            df_lead = preprocess.create_lead_dataset(df, lead, final_features, target_feature)
            df_lead = df_lead[:len(df_lead)-lead]
            print(df_lead.describe())
            df_lead_2018 = df_lead[df_lead.year == 2018]
            print("len 2018 df lead", len(df_lead_2018))
            ## dropping rows with 0 clear_ghi and taking only daytimes
            df_lead = df_lead[(df_lead['clear_ghi'] > 0) & (df_lead['clear_ghi_target'] > 0) ] #0]
            # print(len(df_lead))
            df_lead = df_lead[(df_lead['zen'] < 85) & (df_lead['zen_target'] < 85)]
            df_lead.reset_index(drop=True, inplace=True)
            print("after removing data points with 0 clear_ghi and selecting daytimes", len(df_lead))


            ## get the yearly/seasonal data
            df, test_startdate, test_enddate = preprocess.get_yearly_or_season_data(df_lead, season_flag, testyear)
            print("\n\n after getting seasonal data (test_startdate; test_enddate)", test_startdate, test_enddate)
            print(df.tail)

            # dividing into training and test set
            df_train, df_heldout = preprocess.train_test_spilt(df, season_flag, testyear)
            print("\n\n after dividing_training_test")
            print("train_set\n", len(df_train))
            print("test_set\n", len(df_heldout))

            if len(df_train) > 0 and len(df_heldout) > 0:
                # extract the X_train, y_train, X_test, y_test
                X_train, y_train, X_heldout, y_heldout, col_to_indices_mapping = preprocess.get_train_test_data(
                    df_train, df_heldout, final_features, target_feature)
                print("\n\n train and test df shapes ")
                print(X_train.shape, y_train.shape, X_heldout.shape, y_heldout.shape)

                ## explicitly setting these two indices for using later
                # index_ghi = col_to_indices_mapping['dw_solar']
                index_clearness = col_to_indices_mapping['clearness_index_input']

                # including features from prev imestamps
                X_train = include_previous_features(X_train, col_to_indices_mapping)
                X_heldout = include_previous_features(X_heldout, col_to_indices_mapping)

                # X_train = X_train[n_timesteps:,:]
                # X_heldout = X_heldout[n_timesteps:, :]
                y_train = y_train[n_timesteps:, :]
                y_heldout = y_heldout[n_timesteps:, :]

                print("Final train size: ", X_train.shape, y_train.shape)
                print("Final heldout size: ", X_heldout.shape, y_heldout.shape)

                ## dividing the X_train data into train(70%)/valid(30%) the heldout data is kept hidden
                X_train, X_valid, y_train, y_valid = train_test_split(
                    X_train, y_train, test_size=0.3, random_state=42)#42


                X_test_before_normalized = X_heldout.copy()
                X_train_before_normalized = X_train.copy()

                # print("sanity check: ",n_features,len(col_to_indices_mapping))
                ## normalizing the heldout with the X_train used for training
                X_train, X_valid, X_test = preprocess.standardize_from_train(X_train, X_valid, X_heldout)
                # ,folder_saving + season_flag + "/final_ML_models_"+str(testyear)+"/cnn/"+str(res)+"/"+reg+"/")

                y_train_model = y_train[:, 0]
                y_valid_model = y_valid[:, 0]
                y_test_model = y_heldout[:, 0]

                print("valid size", X_valid.shape, y_valid.shape)
                print("heldout size:", X_test.shape, y_heldout.shape)

                y_test=y_heldout
                y_true = y_test


                # index_ghi = -n_features+index_ghi
                # index_clearness = -n_features+index_clearness


                f.write("\n" + city + " at Lead " + str(lead) + " and " + season_flag + " Season"+"\n")
                #
                # with open(folder_saving + season_flag + "/ML_models_"+str(testyear)+"/cnn/"+str(res)+"/"+reg+"//model_at_lead_" + str(lead) + ".pkl", 'rb') as file:
                #     model = pickle.load(file)

                ## If doing probabilsitic prediction, just use the test function to scale the ys and crps score too
                ## In case of deterministic use the postprocessing modules
                y_pred, y_valid_pred, valid_crps, test_crps_scaled, y_test = cnn.test_DCNN_with_attention(quantile, X_valid,y_valid,
                                                                                           X_test, y_test,
                                                                                           n_timesteps + 1, n_features,
                                                                                           folder_saving + season_flag + "/final_ML_models_" + str(
                                                                                               testyear) + "/cnn/" + str(
                                                                                               res) + "/" + reg + "/",
                                                                                           "dcnn_lag_for_lead_" + str(
                                                                                               lead),  X_test_before_normalized, lead)  # "multi_horizon_dcnn", n_outputs=n_output_steps)

                # y_pred, y_valid_pred, valid_crps, test_crps_scaled  = tranformers.test_LSTM(quantile, X_valid, y_valid,X_test, y_test, n_timesteps+1, n_features,hidden_size,alphas, q50,
                #                                folder_saving + season_flag + "/final_ML_models_"+str(testyear)+"/cnn/"+str(res)+"/"+reg+"/","model_lag_for_lead_" + str(lead),X_test_before_normalized, index_clearghi, lead)#"multi_horizon_dcnn", n_outputs=n_output_steps)

                # y_pred, y_valid_pred, valid_crps, test_crps_scaled, y_test = tranformers.test_transformer(quantile, X_valid, y_valid,
                #                                                                           X_test, y_test, n_timesteps + 1,
                #                                                                            n_features, n_layers, factor,
                #                                                                            num_heads, d_model, alphas,
                #                                                                            q50,
                #                                                                            folder_saving + season_flag + "/final_ML_models_" + str(
                #                                                                                testyear) + "/cnn/" + str(
                #                                                                                res) + "/" + reg + "/",
                #                                                                           "dcnn_lag_for_lead_" + str(
                #                                                                                lead),X_test_before_normalized, index_clearghi, lead)  # "multi_horizon_dcnn", n_outputs=n_output_steps)
                #
                #

            #
                # probabilistic_prediction_plots(X_test_before_normalized,y_pred, y_test, col_to_indices_mapping, folder_saving + season_flag + "/ML_models_" + str(
                 #                                                                              testyear) + "/cnn/" + str(
                  #                                                                             res) + "/" )
            # # # #
            #     y_CH_PeEN, CH_PeEN_baseline_quantiles, crps_CH_PeEN = get_CH_PeEN_baseline_crps(X_test_before_normalized, X_train_before_normalized, col_to_indices_mapping, y_test, lead)
            #     y_climatology, climatology_probabilistic_baseline, crps_climatology = get_climatology_baseline_crps(X_test_before_normalized, X_train_before_normalized, col_to_indices_mapping, y_test, lead)

            #     # reliability_diagrams(np.arange(0.05, 1.0, 0.05), y_pred, CH_PeEN_baseline_quantiles, climatology_probabilistic_baseline, y_test,
            # #                          folder_saving + season_flag + "/ML_models_" + str(
            # #                              testyear) + "/cnn/" + str(
            # #                              res) + "/")
            # #
            # #     sharpness_diagrams(np.arange(0.05, 1.0, 0.05),  y_pred, CH_PeEN_baseline_quantiles, climatology_probabilistic_baseline,
            # #                        folder_saving + season_flag + "/ML_models_" + str(
            # #                            testyear) + "/cnn/" + str(
            # #                            res) + "/")
            #
            #
                #
                # y_test = np.reshape(y_heldout, -1)
                # y_valid = np.reshape(y_valid, -1)
                #
                # y_true = y_test
                #
                # y_pred = model.predict(X_test) #.reshape((X_test.shape[0], n_timesteps+1, n_features)))
                #
                # y_pred = np.reshape(y_pred, -1)

            #
            # #
            # #
            #
                print("\n" + city + " at Lead " + str(lead) + " and " + season_flag + " Season")

                print("#####TEST#################")
            #     # postprocessing on test
                y_true, y_pred = postprocess.postprocessing_target(y_pred, y_true) #, X_test_before_normalized, index_ghi, index_clearghi, lead)
                # # normal and smart persistence model
            #     y_np = postprocess.normal_persistence_model(X_test_before_normalized, index_ghi, lead)
                print(len(X_test_before_normalized), index_clearness)
                y_sp = postprocess.smart_persistence_model(X_test_before_normalized, y_test, index_clearness, lead)
            #     # y_climatology = postprocess.climatology_baseline(X_test_before_normalized, df_2017, col_to_indices_mapping, n_features)
            #     true_day_test, pred_day_test, np_day_test, sp_day_test, climatology_test, CH_PeEN_test = postprocess.final_true_pred_sp_np(y_true, y_pred, y_np, y_sp, y_climatology, y_CH_PeEN, lead, X_test_before_normalized, index_zen, index_clearghi)
            #     #
                rmse_our, mae_our, mb_our, sd_our, r2_our = postprocess.evaluation_metrics(y_true, y_pred)
                #
                print("Performance of our model (rmse, mae, mb, sd, r2): \n\n", round(rmse_our, 2), round(mae_our, 2),
                      round(mb_our, 2), round(sd_our, 2), round(r2_our, 2))


                rmse_sp, mae_sp, mb_sp, sd_sp, r2_sp = postprocess.evaluation_metrics(y_true, y_sp)
                print("Performance of smart persistence model (rmse, mae, mb, sd, r2): \n\n", round(rmse_sp, 2),
                      round(mae_sp, 2),
                      round(mb_sp, 2), round(sd_sp, 2), round(r2_sp, 2))
            #
            #     rmse_np, mae_np, mb_np, sd_np, r2_np = postprocess.evaluation_metrics(true_day_test, np_day_test)
            #     print("Performance of normal persistence model (rmse, mae, mb, sd, r2): \n\n", round(rmse_np, 2),
            #           round(mae_np, 2),
            #           round(mb_np, 2), round(sd_np, 2), round(r2_np, 2))
            #
            #     rmse_clm, mae_clm, mb_clm, sd_clm, r2_clm = postprocess.evaluation_metrics(true_day_test, climatology_test)
            #     print("Performance of hourly climatology baseline model (rmse, mae, mb, sd, r2): \n\n", round(rmse_clm, 2),
            #           round(mae_clm, 2),
            #           round(mb_clm, 2), round(sd_clm, 2), round(r2_clm, 2))
            #
            #     rmse_chp, mae_chp, mb_chp, sd_chp, r2_chp = postprocess.evaluation_metrics(true_day_test,
            #                                                                                CH_PeEN_test)
            #     print("Performance of CH_PeEN baseline model (rmse, mae, mb, sd, r2): \n\n", round(rmse_chp, 2),
            #           round(mae_chp, 2),
            #           round(mb_chp, 2), round(sd_chp, 2), round(r2_chp, 2))
            #
            #     # calculate the skill score of our model over persistence model
            #     skill_sp = postprocess.skill_score(rmse_our, rmse_sp)
            #     print("\nSkill rmse of our model over smart persistence: ", round(skill_sp, 2))
            #
            #     skill_np = postprocess.skill_score(rmse_our, rmse_np)
            #     print("\nSkill rmse of our model over normal persistence: ", round(skill_np, 2))
            #
            #     skill_clm = postprocess.skill_score(rmse_our, rmse_clm)
            #     print("\nSkill rmse of our model over hourly climatology baseline: ", round(skill_clm, 2))
            #
            #     skill_chp = postprocess.skill_score(rmse_our, rmse_chp)
            #     print("\nSkill rmse of our model over CH_PeEN baseline: ", round(skill_chp, 2))
            #
            #     f.write('skill rmse score on heldout data for year 2018 for lead' + str(lead) + '=' + str(round(skill_sp, 2)) + '\n')
            #     f.write('skill wrt hourly climatology rmse score on heldout data for year 2018 for lead' + str(lead) + '=' + str(
            #         round(skill_clm, 2)) + '\n')
            #     f.write(
            #         'skill wrt CH_PeEN rmse score on heldout data for year 2018 for lead' + str(lead) + '=' + str(
            #             round(skill_chp, 2)) + '\n')
            #
            #     skill_sp = postprocess.skill_score(mae_our, mae_sp)
            #     print("\nSkill mae of our model over smart persistence: ", round(skill_sp, 2))
            #
            #     skill_np = postprocess.skill_score(mae_our, mae_np)
            #     print("\nSkill mae of our model over normal persistence: ", round(skill_np, 2))
            #
            #     skill_clm = postprocess.skill_score(mae_our, mae_clm)
            #     print("\nSkill mae of our model over climatology baseline: ", round(skill_clm, 2))
            #
            #     skill_chp = postprocess.skill_score(mae_our, mae_chp)
            #     print("\nSkill mae of our model over climatology baseline: ", round(skill_chp, 2))
            #
            #     f.write('skill mae score on heldout data for year 2018 for lead' + str(lead) + '=' + str(
            #         round(skill_sp, 2)) + '\n')
            #     f.write(
            #         'skill wrt hourly climatology mae score on heldout data for year 2018 for lead' + str(lead) + '=' + str(
            #             round(skill_clm, 2)) + '\n')
            #     f.write(
            #         'skill wrt CH_PeEN mae score on heldout data for year 2018 for lead' + str(lead) + '=' + str(
            #             round(skill_chp, 2)) + '\n')
            # # #     # #
            #     valid_crps = get_crps_for_ngboost(model, X_valid, y_valid)
            #     # test_crps = get_crps_for_ngboost(model, X_test, y_test)
            # #     # print("before crps",y_test.shape)
            #     test_crps_scaled = get_crps_for_ngboost_scaled(model, X_test, y_test, X_test_before_normalized, index_clearghi, lead)
            # #
            #     print('CRPS score on valid data for lead' + str(lead) + '=' + str(
            #         round(valid_crps, 2)) + '\n')
            # # #     # # # # print('CRPS score on heldout data for year 2018 for lead' + str(lead) + '=' + str(round(test_crps, 2)) + '\n')
            # # #     # # # # f.write('CRPS score on heldout data for year 2018 for lead' + str(lead) + '=' + str(round(test_crps, 2)) + '\n')
            # #     # # #
            #     print('CRPS scaled score on heldout data for year 2018 for lead' + str(lead) + '=' + str(
            #         round(test_crps_scaled, 2)) + '\n')
            #     f.write('CRPS scaled score on heldout data for year 2018 for lead' + str(lead) + '=' + str(
            #         round(test_crps_scaled, 2)) + '\n')
            #     #
                # print('CRPS CH_PeEN scaled score on heldout data for year 2018 for lead' + str(lead) + '=' + str(
                #     round(crps_CH_PeEN, 2)) + '\n')
                # f.write('CRPS CH_PeEN scaled score on heldout data for year 2018 for lead' + str(lead) + '=' + str(
                #     round(crps_CH_PeEN, 2)) + '\n')
                #
                # print('CRPS hourly Climatology scaled score on heldout data for year 2018 for lead' + str(lead) + '=' + str(
                #     round(crps_climatology, 2)) + '\n')
                # f.write('CRPS hourly Climatology scaled score on heldout data for year 2018 for lead' + str(lead) + '=' + str(
                #     round(crps_climatology, 2)) + '\n')

                ## saving predictions with and without night times
                test_with_predictions_array = X_test_before_normalized[:,[col_to_indices_mapping['month'], col_to_indices_mapping['day'], col_to_indices_mapping['hour']],-1]
                # test_with_predictions_array['true_GHI'] = y_test
                test_with_predictions_array = pd.DataFrame(test_with_predictions_array, columns=['month', 'day', 'hour'])

                test_with_predictions_array['predictions_GHI'] = y_pred


                test_with_predictions_array_with_night_times = df_lead_2018[
                    ['year', 'month', 'day', 'hour', 'clearness_index']]


                test_with_predictions_array_with_night_times = test_with_predictions_array_with_night_times.merge(
                    test_with_predictions_array, on=['month', 'day', 'hour'], how='left')

                print(test_with_predictions_array_with_night_times.describe())

                print(np.nanmax(test_with_predictions_array_with_night_times.predictions_GHI.values))
                np.save(folder_saving + season_flag + "/final_ML_models_" + str(testyear) + "/cnn/" + str(
                    res) + "/" + reg + "/" + 'model_predictions_with_all_info_including_night_times.npy',
                        test_with_predictions_array_with_night_times)

            # else:
            #     print("not enough data for the season: ", season_flag, "and lead: ", lead)

        f.close()


if __name__=='__main__':
    main()
