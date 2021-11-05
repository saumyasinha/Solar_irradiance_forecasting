import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dateutil import parser
from datetime import datetime
import datetime
import pickle
import os
from datetime import date, timedelta

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def extract_frame(dir_path):
    '''
     read the raw data input(csvs) and combine them into a dataframe
    '''
    subdir_list = glob.glob(dir_path + '/*/')
    file_list = []
    for subdir in subdir_list:
        file_list.extend(glob.glob(str(subdir) + '*.csv'))
    combined_csv = pd.concat([pd.read_csv(str(f)) for f in file_list])
    return (combined_csv)


def extract_time(x):
    datestring = x.split('/')[0]
    dt = parser.parse(datestring)
    return pd.Series((dt.year, dt.month, dt.day, dt.hour, dt.minute))





def extract_study_period(dataframe,startmonth, startyear, endmonth, endyear):
    year = np.arange(startyear, endyear + 1, 1)
    yearlist = []
    for ind in range(len(year)):
        tempvar = dataframe[dataframe['year'] == year[ind]]
        if ind == 0:
            tempvar = tempvar[tempvar['month'] >= startmonth]
        elif ind == (len(year) - 1):
            tempvar = tempvar[tempvar['month'] <= endmonth]
        yearlist.append(tempvar)
        del tempvar
    df_yearly = pd.concat(yearlist)
    ## added this sorting just to be sure
    # df_yearly = df_yearly.sort_values(by=['year', 'month', 'day', 'hour', 'MinFlag'])
    return df_yearly


# # not used anymore
# def adjust_outlier_clearness_index(dataframe):
#     # print(len(dataframe.loc[(dataframe.clearness_index < 0.0)]))
#     # print(len(dataframe.loc[(dataframe.clearness_index >= 5)]))
#     dataframe.loc[(dataframe.clearness_index < 0.0), 'clearness_index'] = 0.0
#     dataframe.loc[(dataframe.clearness_index >= 4), 'clearness_index'] = 4.0 # is 4.0 arbitrary here?
#
#     ## look at the outlier again by plotting
#     return dataframe
#
# ## not used anymore
# def adjust_boundary_values(dataframe):
#     '''
#     adjust outlier values, precisely negative solar and negative clear_ghi
#     '''
#
#     dataframe.loc[(dataframe.dw_solar < 0), 'dw_solar'] = 0.0
#     dataframe.loc[(dataframe.clear_ghi <= 0), 'clear_ghi'] = 1.0  #or 4.0
#
#
#     ## plot the clear ghi and remove the outliers, doesn't need to be normalized between 0 and 1 (4 might be better)
#     return dataframe

#
# def get_df_for_all_seasons(dataframe):
#     '''
#     create dataframes for the four seasons (fall, winter, spring, summer)
#     '''
#     # extracting the data on month scale
#     df_sep = dataframe[dataframe.month == 9]
#     df_oct = dataframe[dataframe.month == 10]
#     df_nov = dataframe[dataframe.month == 11]
#     df_dec = dataframe[dataframe.month == 12]
#     df_jan = dataframe[dataframe.month == 1]
#     df_feb = dataframe[dataframe.month == 2]
#     df_mar = dataframe[dataframe.month == 3]
#     df_apr = dataframe[dataframe.month == 4]
#     df_may = dataframe[dataframe.month == 5]
#     df_jun = dataframe[dataframe.month == 6]
#     df_jul = dataframe[dataframe.month == 7]
#     df_aug = dataframe[dataframe.month == 8]
#     # season formation
#     df_fall = pd.concat([df_sep, df_oct, df_nov])
#     df_winter = pd.concat([df_dec, df_jan, df_feb])
#     df_spring = pd.concat([df_mar, df_apr, df_may])
#     df_summer = pd.concat([df_jun, df_jul, df_aug])
#     return df_fall, df_winter, df_spring, df_summer

def create_lead_dataset(dataframe, lead, final_set_of_features, target):
    '''
    create dataset for the given lead by adjusting the lead between the input features and target irradiance
    '''
    dataframe_lead = dataframe[final_set_of_features]
    target = np.asarray(dataframe[target])
    print("for target, it is rolled by lead (Y var) \n")
    target = np.roll(target, -lead)
    # target[-lead:] = np.nan
    dataframe_lead['clearness_index'] = target
    dataframe_lead['clear_ghi_target'] = np.roll(np.asarray(dataframe['clear_ghi']), -lead)
    dataframe_lead['zen_target'] = np.roll(np.asarray(dataframe['zen']), -lead)

    # remove rows which have any value as NaN
    # dataframe_lead = dataframe_lead.dropna()
    print("*****************")
    print("dataframe with lead size: ",len(dataframe_lead))
    return dataframe_lead




def get_yearly_or_season_data(df_lead, season_flag, testyear):

    if season_flag == 'year' or season_flag == 'yearly':
        df = df_lead
        # start_date = date(testyear, 9, 1)
        # end_date = date(testyear+1, 8, 31)
        start_date = date(testyear, 1, 1)
        end_date = date(testyear, 12, 31)
    elif season_flag == 'fall':
        df = df_lead[df_lead.month.isin([9,10,11])]
        start_date = date(testyear, 9, 1)
        end_date = date(testyear, 11, 30)
    elif season_flag == 'winter':
        df = df_lead[df_lead.month.isin([12,1,2])]
        start_date = date(testyear, 12, 1)
        end_date = date(testyear+1, 2, 28)
    elif season_flag == 'spring':
        df = df_lead[df_lead.month.isin([3,4,5])]
        start_date = date(testyear+1, 3, 1)
        end_date = date(testyear+1, 5, 31)
    elif season_flag == 'summer':
        df = df_lead[df_lead.month.isin([6,7,8])]
        start_date = date(testyear+1, 6, 1)
        end_date = date(testyear+1, 8, 31)
    else:
        print("Please provide a valid season...")

    return df, start_date, end_date


def train_test_spilt(dataframe, season_flag, testyear):
    '''
    divide the total dataset into training and test set for each season (or year)
    '''

    if season_flag == 'fall':
        dataframe_test = dataframe[dataframe.year == testyear]
        dataframe_train = dataframe[dataframe.year != testyear]
    elif season_flag == 'winter':
        dataframe_test = dataframe[((dataframe.year == testyear) & (dataframe.month == 12)) | (
                    (dataframe.year == testyear + 1) & (dataframe.month == 1)) | (
                                               (dataframe.year == testyear + 1) & (dataframe.month == 2))]
        dataframe_train = pd.concat([dataframe, dataframe_test, dataframe_test]).drop_duplicates(keep=False)
    elif season_flag == 'spring' or season_flag == 'summer':
        dataframe_test = dataframe[dataframe.year == testyear + 1]
        dataframe_train = dataframe[dataframe.year != testyear + 1]
    elif season_flag == 'year' or season_flag == 'yearly':
        # dataframe_test = dataframe[((dataframe.year == testyear) & (dataframe.month >= 9) & (dataframe.month <= 12)) | (
        #             (dataframe.year == testyear + 1) & (dataframe.month >= 1) & (dataframe.month <= 8))]
        # dataframe_train = pd.concat([dataframe, dataframe_test, dataframe_test]).drop_duplicates(keep=False)
        dataframe_test = dataframe[dataframe.year == testyear]
        dataframe_train = dataframe[dataframe.year != testyear]

    # removing the rows with the GHI(dw_solar) variable as Null (why would it be NULL??)
    # print(len(dataframe_train[dataframe_train['dw_solar'].isnull()]))
    # dataframe_train_dropna = dataframe_train[dataframe_train['dw_solar'].isnull() == False]
    # dataframe_test_dropna = dataframe_test[dataframe_test['dw_solar'].isnull() == False]
    #     dataframe_test_s.reset_index(inplace=True)
    # print(len(pd.merge(dataframe_train, dataframe_test, how='inner')))
    return dataframe_train, dataframe_test


def get_train_test_data(dataframe_train, dataframe_test, final_set_of_features, target):
    '''
    Get X_train, y_train, X_test, y_test
    '''
    final_features = []
    for feature in final_set_of_features:
        # if feature not in ['year', 'month', 'day', 'hour', 'MinFlag']:
        if feature not in ['year', 'day', 'MinFlag']:
            final_features.append(feature)

    final_features.extend(['clearness_index_input'])#,'smart_persistence'])

    # index_zen=-1
    # # storing the position/indices of clear_ghi, ghi, and zen
    # for ind in range(len(final_features)):
    #     if final_features[ind] == 'clear_ghi':
    #         index_clearghi = ind
    #     if final_features[ind] == 'dw_solar':
    #         index_ghi =  ind
    #     if final_features[ind] == 'zen':
    #         index_zen = ind

    print("total features: ", len(final_features))
    dataframe_train['clearness_index_input'] = dataframe_train['dw_solar']/dataframe_train['clear_ghi']
    dataframe_test['clearness_index_input'] = dataframe_test['dw_solar'] / dataframe_test['clear_ghi']

    col_to_indices_mapping = {k: v for v, k in enumerate(final_features)}
    print(col_to_indices_mapping)
    # Selecting the final features and target variables
    X_train = np.asarray(dataframe_train[final_features]).astype(np.float)
    X_test = np.asarray(dataframe_test[final_features]).astype(np.float)

    print(list(dataframe_train.columns))
    # y_train = np.asarray(np.vstack(dataframe_train[target].values.tolist())).astype(np.float)
    # y_test = np.asarray(np.vstack(dataframe_test[target].values.tolist())).astype(np.float)
    target_columns = [target, 'clear_ghi_target', 'zen_target']
    y_train = np.asarray(dataframe_train[target_columns]).astype(np.float)
    print(y_train)
    y_test = np.asarray(dataframe_test[target_columns]).astype(np.float)



    return X_train, y_train, X_test, y_test, col_to_indices_mapping


# ## Don't need this anymore
# def select_daytimes(X, index_zen, zenith_threhsold = 85):
#     zen = np.asarray(X[:, index_zen])
#     daytimes = []
#     for i in range(zen.shape[0]):
#         if zen[i] < zenith_threhsold:
#             daytimes.append(i)
#     #     print("Only daytimes: ", len(daytimes))
#     return daytimes


def filter_dayvalues_and_zero_clearghi(X_all, y_all, index_zen, index_clearghi, zenith_threshold = 85):
    '''
    filter only the day time data and remove 0 clear_ghi from data
    '''
    # out_ind = []
    # for i in range(y_train_all.shape[0]):
    #     if y_train_all[i, 0] <= 0.0:
    #         out_ind.append(i)
    # # subtracting ignore list from day times
    # a = set(index_daytimes_train)
    # b = set(out_ind)
    # final_daytime_train = list(a - b)
    #
    # X_train = []
    # y_train = []
    # X_test = []
    # y_test = []
    # for i in final_daytime_train:
    #     X_train.append(X_train_all[i, :])
    #     y_train.append(y_train_all[i, :])
    # for i in index_daytimes_test:
    #     X_test.append(X_test_all[i, :])
    #     y_test.append(y_test_all[i, :])
    # X_train = np.asarray(X_train).astype(np.float)
    # y_train = np.asarray(y_train).astype(np.float)
    # X_test = np.asarray(X_test).astype(np.float)
    # y_test = np.asarray(y_test).astype(np.float)

    # Faster implementation
    # print("After removal of night values and training outliers: train and test: ",X_train.shape, X_test.shape)
    X = X_all[np.where(X_all[:, index_clearghi] >0)]
    y = y_all[np.where(X_all[:,index_clearghi] >0)]

    # X_train = X_train_all[np.where(y_train_all[:, 0] > 0)]
    # y_train = y_train_all[np.where(y_train_all[:, 0] > 0)]

    print("After removing 0 clear ghi: ",len(X))

    X = X[np.where(X[:,index_zen] < zenith_threshold)]
    y = y[np.where(X[:,index_zen] < zenith_threshold)]

    print("After further removing any daytimes left: ", len(X))


    return X, y



def standardize_from_train(X_train, X_valid, X_test):

    '''
    Standardize (or 'normalize') the feature matrices.
    '''


    if X_train is not None:
        cols = X_train.shape[1]
        # print(cols)
        standarize_dict = {}
        for i in range(cols):
            # if i%total_features==index_ghi:
            #     # print("here")
            #     index_clearghi = i+diff
            #     ## normalizing of dw_solar wrt clearGHI (to take the cloud factor into account)
            #     mean_clear = np.mean(X_train[:, index_clearghi])
            #     std_clear = np.std(X_train[:, index_clearghi])
            #     X_train[:, index_ghi] = (X_train[:, index_ghi] - mean_clear) / std_clear
            #     X_valid[:, index_ghi] = (X_valid[:, index_ghi] - mean_clear) / std_clear
            #
            #     if X_test is not None:
            #         X_test[:, index_ghi] = (X_test[:, index_ghi] - mean_clear) / std_clear
            #     standarize_dict[i] = (mean_clear, std_clear)
            #     # max_clear = np.max(X_train[:, index_clearghi])
            #     # min_clear = np.min(X_train[:, index_clearghi])
            #     # X_train[:, index_ghi] = (X_train[:, index_ghi] - min_clear) / (max_clear - min_clear)
            #     # X_valid[:, index_ghi] = (X_valid[:, index_ghi] - min_clear) / (max_clear - min_clear)
            #     # X_test[:, index_ghi] = (X_test[:, index_ghi] - min_clear) / (max_clear - min_clear)
            #     # standarize_dict[i] = (max_clear, min_clear)

            mean = np.mean(X_train[:,i])
            std = np.std(X_train[:,i])
            max = np.max(X_train[:,i])
            min = np.min(X_train[:,i])
            # print(min,max)
            ##normalize or standarize ?
            X_train[:,i] = (X_train[:,i] - mean)/std
            X_valid[:, i] = (X_valid[:, i] - mean) / std
            if X_test is not None:
                X_test[:,i] = (X_test[:,i] - mean)/std
            standarize_dict[i] = (mean,std)
            # X_train[:,i] = (X_train[:,i] - min)/(max-min)
            # X_valid[:, i] = (X_valid[:, i] - min) / (max-min)
            # X_test[:,i] = (X_test[:,i] - min)/(max-min)
            # standarize_dict[i] = (max,min)



        # with open(folder_saving+"standarize_data_for_lead_"+str(lead)+".pickle", 'wb') as handle:
        #     pickle.dump(standarize_dict, handle)


    # else:
    #     # print("in else")
    #     cols = X_test.shape[1]
    #
    #     with open(folder_saving+"standarize_data_for_lead_"+str(lead)+".pickle", 'rb') as handle:
    #         standarize_dict = pickle.load(handle)
    #
    #     for i in range(cols):
    #         ##normalize or standarize ?
    #         mean = standarize_dict[i][0]
    #         std = standarize_dict[i][1]
    #         # print(mean,std)
    #         # print("test before",np.mean(X_test[:,i]))
    #         X_test[:,i] = (X_test[:,i] - mean)/std
    #         # print("test after",np.mean(X_test[:, i]))
    #         # max = standarize_dict[i][0]
    #         # min = standarize_dict[i][1]
    #         if X_valid is not None:
    #             # print("valid before", np.mean(X_valid[:, i]))
    #             X_valid[:,i] = (X_valid[:,i] - mean)/std
    #             # print("valid after", np.mean(X_valid[:, i]))

    return X_train, X_valid, X_test


def shuffle(X,y, city, res):
    filename = 'indices/'+str(city)+'/indices_'+str(res)+".npy"
    if os.path.isfile(filename) == False:
        print("here")
        p = np.random.permutation(len(X))
        np.save(filename, p)
    else:
        p = np.load(filename)

    return X[p], y[p]


def generateFlag(x):
    # if int(x) < 15 and int(x) >= 0:
    #     return 1
    # elif int(x) < 30 and int(x) >= 15:
    #     return 2
    # elif int(x) < 45 and int(x) >= 30:
    #     return 3
    # elif int(x) <= 60 and int(x) >= 45:
    #     return 4

    if int(x) < 5 and int(x) >= 0:
        return 1
    elif int(x) < 10 and int(x) >= 5:
        return 2
    elif int(x) < 15 and int(x) >= 10:
        return 3
    elif int(x) < 20 and int(x) >= 15:
        return 4
    if int(x) < 25 and int(x) >= 20:
        return 5
    elif int(x) < 30 and int(x) >= 25:
        return 6
    elif int(x) < 35 and int(x) >= 30:
        return 7
    elif int(x) < 40 and int(x) >= 35:
        return 8
    if int(x) < 45 and int(x) >= 40:
        return 9
    elif int(x) < 50 and int(x) >= 45:
        return 10
    elif int(x) < 55 and int(x) >= 50:
        return 11
    elif int(x) <= 60 and int(x) >= 55:
        return 12

