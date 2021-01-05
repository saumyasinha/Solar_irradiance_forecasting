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


## not used anymore
def adjust_boundary_values(dataframe):
    '''
    adjust outlier values, precisely negative solar and negative clear_ghi
    '''

    dataframe.loc[(dataframe.dw_solar < 0), 'dw_solar'] = 0.0
    dataframe.loc[(dataframe.clear_ghi <= 0), 'clear_ghi'] = 1.0  #or 4.0


    ## plot the clear ghi and remove the outliers, doesn't need to be normalized between 0 and 1 (4 might be better)
    return dataframe


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


# not used anymore
def adjust_outlier_clearness_index(dataframe):
    # print(len(dataframe.loc[(dataframe.clearness_index < 0.0)]))
    # print(len(dataframe.loc[(dataframe.clearness_index >= 5)]))
    dataframe.loc[(dataframe.clearness_index < 0.0), 'clearness_index'] = 0.0
    dataframe.loc[(dataframe.clearness_index >= 4), 'clearness_index'] = 4.0 # is 4.0 arbitrary here?

    ## look at the outlier again by plotting
    return dataframe


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

    # remove rows which have any value as NaN
    # dataframe_lead = dataframe_lead.dropna()
    print("*****************")
    print("dataframe with lead size: ",len(dataframe_lead))
    return dataframe_lead

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


def get_yearly_or_season_data(df_lead, season_flag, testyear):

    if season_flag == 'year' or season_flag == 'yearly':
        df = df_lead
        start_date = date(testyear, 9, 1)
        end_date = date(testyear+1, 8, 31)
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
        dataframe_test = dataframe[((dataframe.year == testyear) & (dataframe.month >= 9) & (dataframe.month <= 12)) | (
                    (dataframe.year == testyear + 1) & (dataframe.month >= 1) & (dataframe.month <= 8))]
        dataframe_train = pd.concat([dataframe, dataframe_test, dataframe_test]).drop_duplicates(keep=False)

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
        if feature not in ['year', 'month', 'day', 'MinFlag']:
            final_features.append(feature)

    # storing the position/indices of clear_ghi, ghi, and zen
    for ind in range(len(final_features)):
        if final_features[ind] == 'clear_ghi':
            index_clearghi = ind
        if final_features[ind] == 'dw_solar':
            index_ghi = ind
        if final_features[ind] == 'zen':
            index_zen = ind

    col_to_indices_mapping = {k: v for v, k in enumerate(final_features)}
    # Selecting the final features and target variables
    X_train = np.asarray(dataframe_train[final_features]).astype(np.float)
    X_test = np.asarray(dataframe_test[final_features]).astype(np.float)


    y_train = np.asarray(np.vstack(dataframe_train[target].values.tolist())).astype(np.float)
    y_test = np.asarray(np.vstack(dataframe_test[target].values.tolist())).astype(np.float)
    print(y_train)

    # add clearness_index beyond lead as a feature input (important) -- are we adding the current time clearness index here?
    # y_train_roll = np.roll(y_train, lead)
    # y_test_roll = np.roll(y_test, lead)
    # X_train = np.concatenate((X_train, y_train_roll), axis=1)
    # X_test = np.concatenate((X_test, y_test_roll), axis=1)

    return X_train, y_train, X_test, y_test, index_clearghi, index_ghi, index_zen, col_to_indices_mapping


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




def standardize_from_train(X_train, X_valid, X_test, folder_saving, model, lead=""):
    '''
    Standardize (or 'normalize') the feature matrices.
    '''

    if X_train is not None:
        cols = X_train.shape[1]
        standarize_dict = {}
        for i in range(cols):

            mean = np.mean(X_train[:,i])
            std = np.std(X_train[:,i])
            max = np.max(X_train[:,i])
            min = np.min(X_train[:,i])
            ##normalize or standarize ?
            X_train[:,i] = (X_train[:,i] - mean)/std
            X_valid[:, i] = (X_valid[:, i] - mean) / std
            X_test[:,i] = (X_test[:,i] - mean)/std
            standarize_dict[i] = (mean,std)

        with open(folder_saving+model+"_standarize_data_for_lead_"+str(lead)+".pickle", 'wb') as handle:
            pickle.dump(standarize_dict, handle)


    else:
        cols = X_test.shape[1]

        with open(folder_saving+model+"_standarize_data_for_lead_"+str(lead)+".pickle", 'rb') as handle:
            standarize_dict = pickle.load(handle)

        for i in range(cols):

            mean = standarize_dict[i][0]
            std = standarize_dict[i][1]
            ##normalize or standarize ?
            X_test[:,i] = (X_test[:,i] - mean)/std


    return X_train, X_valid, X_test


def shuffle(X,y):

    if os.path.isfile('indices.npy') == False:
        print("here")
        p = np.random.permutation(len(X))
        np.save('indices.npy', p)
    else:
        p = np.load('indices.npy')

    return X[p], y[p]


def generateFlag(x):
    if int(x) < 15 and int(x) >= 0:
        return 1
    elif int(x) < 30 and int(x) >= 15:
        return 2
    elif int(x) < 45 and int(x) >= 30:
        return 3
    elif int(x) <= 60 and int(x) >= 45:
        return 4

