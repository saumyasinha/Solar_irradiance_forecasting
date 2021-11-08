import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
#
# from SolarForecasting.ModulesProcessing import collect_data, clean_data
# from SolarForecasting.ModulesLearning import preprocessing as preprocess
# from SolarForecasting.ModulesLearning import postprocessing as postprocess
# from SolarForecasting.ModulesLearning.ModulesCNN import train as cnn
# from SolarForecasting.ModulesLearning.ModuleLSTM import train as tranformers

from ModulesProcessing import collect_data, clean_data
from ModulesLearning import preprocessing as preprocess
from ModulesLearning import postprocessing as postprocess
from ModulesLearning.ModulesCNN import train as cnn
from ModulesLearning.ModuleLSTM import train as tranformers



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
path_project = path_cluster
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
n_timesteps = 48 #1day (can be 12hrs or 48hrs for a few models)
n_features = 16 + 51
quantile =False #False

#hyperparameters for LSTM/Transformer/CNNs
n_layers = 1 #2 #3
num_heads = 2 #4
d_model = 64 #128

hidden_size= 50
batch_size = 16 #32 #16 #16 

epochs = 300 #250
lr = 1e-5 #1e-5 #1e-4

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




# def create_mulitple_lead_dataset(dataframe, final_set_of_features, target):
#     dataframe_lead = dataframe[final_set_of_features]
#     target = np.asarray(dataframe[target])
#
#     y_list = []
#     for lead in lead_times:
#     # for lead in range(1,n_output_steps+1):
#         print("rolling by: ",lead)
#         target = np.roll(target, -lead)
#         y_list.append(target)
#
#     dataframe_lead['clearness_index'] = np.column_stack(y_list).tolist()
#     dataframe_lead['clearness_index'] = dataframe_lead['clearness_index'].apply(tuple)
#     max_lead = np.max(lead_times)
#     dataframe_lead = dataframe_lead[:len(dataframe_lead) - max_lead]
#
#     # dataframe_lead['clearness_index'].values[-max_lead:] = np.nan
#     print(dataframe_lead['clearness_index'])
#
#     # remove rows which have any value as NaN
#     # dataframe_lead = dataframe_lead.dropna()
#     print("*****************")
#     print("dataframe with lead size: ", len(dataframe_lead))
#     return dataframe_lead




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
    reg = "tcn_week_ahead_48_seq_lag_large_kernel_1hr_res"
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
            df_lead_2018 = df_lead[df_lead.year == 2018]
            print(df_lead.describe())
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
                index_ghi = col_to_indices_mapping['dw_solar']
                index_clearghi = col_to_indices_mapping['clear_ghi']

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

                print("train/valid sizes: ", X_train.shape, " ", X_valid.shape)


                # normalizing the Xtrain, Xvalid and Xtest data
                X_train, X_valid, X_test = preprocess.standardize_from_train(X_train, X_valid, None)
                                                                             # ,folder_saving + season_flag + "/final_ML_models_"+str(testyear)+"/cnn/"+str(res)+"/"+reg+"/")

                # f.write("epochs = " + str(epochs) + '\n')
                # f.write("batch_size = " + str(batch_size) + '\n')
                # f.write("learning rate = " + str(lr) + '\n')
                # # f.write("n_layers = " + str(n_layers) + '\n')
                # # f.write("factor = " + str(factor) + '\n')
                # # f.write("num_heads = " + str(num_heads) + '\n')
                # # f.write("d_model = " + str(d_model) + '\n')
                # f.write("seq_len = " + str(n_timesteps) + '\n')
                # f.write("total features = " + str(n_features)+ '\n')
                # f.write("alphas = " + str(len((alphas))) + '\n')
                #
                y_train_model = y_train[:,0]
                y_valid_model = y_valid[:,0]
                y_test_model = y_heldout[:, 0]
                cnn.train_DCNN_with_attention(quantile,X_train, y_train_model, X_valid, y_valid_model, n_timesteps+1, n_features,
                                                                                          folder_saving + season_flag + "/final_ML_models_" + str(
                                                                                              testyear) + "/cnn/" + str(
                                                                                              res) + "/" + reg + "/",
                                                                                          model_saved="dcnn_lag_for_lead_" + str(
                                                                                              lead))
                y_pred, y_valid_pred, valid_crps, test_crps,y_test = cnn.test_DCNN_with_attention(quantile, X_valid, y_valid_model,
                                                                                          None, None,
                                                                                          n_timesteps + 1, n_features,
                                                                                          folder_saving + season_flag + "/final_ML_models_" + str(
                                                                                              testyear) + "/cnn/" + str(
                                                                                              res) + "/" + reg + "/",
                                                                                          model_saved="dcnn_lag_for_lead_" + str(
                                                                                              lead))  # "multi_horizon_dcnn", n_outputs=n_output_steps)

            #     tranformers.train_LSTM(quantile, X_train, y_train_model, X_valid, y_valid_model, n_timesteps+1, n_features,hidden_size , batch_size, epochs, lr,alphas, q50,
            #                              folder_saving + season_flag + "/final_ML_models_"+str(testyear)+"/cnn/"+str(res)+"/"+reg+"/",model_saved ="model_lag_for_lead_" + str(lead)) #"multi_horizon_dcnn", n_outputs=n_output_steps)
            #
            #     y_pred, y_valid_pred, valid_crps, test_crps  = tranformers.test_LSTM(quantile, X_valid, y_valid_model,None,None, n_timesteps+1, n_features,hidden_size,alphas, q50,
            #                                   folder_saving + season_flag + "/final_ML_models_"+str(testyear)+"/cnn/"+str(res)+"/"+reg+"/",model_saved = "model_lag_for_lead_" + str(lead))#"multi_horizon_dcnn", n_outputs=n_output_steps)
            # #     tranformers.train_transformer(quantile, X_train, y_train_model, X_valid, y_valid_model, n_timesteps+1, n_features, n_layers, num_heads, d_model, batch_size, epochs, lr,alphas, q50,
            # #                               folder_saving + season_flag + "/final_ML_models_"+str(testyear)+"/cnn/"+str(res)+"/"+reg+"/",model_saved ="dcnn_lag_for_lead_" + str(lead)) #"multi_horizon_dcnn", n_outputs=n_output_steps)
            #
            #     y_pred, y_valid_pred, valid_crps, test_crps  = tranformers.test_transformer(quantile, X_valid, y_valid_model,None,None, n_timesteps+1, n_features,n_layers,  num_heads, d_model,alphas, q50,
            #                                   folder_saving + season_flag + "/final_ML_models_"+str(testyear)+"/cnn/"+str(res)+"/"+reg+"/",model_saved = "dcnn_lag_for_lead_" + str(lead))#"multi_horizon_dcnn", n_outputs=n_output_steps)
            #
            # #

                f.write("\n" + city + " at Lead " + str(lead) + " and " + season_flag + " Season")
                print("\n" + city + " at Lead " + str(lead) + " and " + season_flag + " Season")


                print("##########VALID##########")
                rmse_our, mae_our, mean_our,std_our,r2_our = postprocess.evaluation_metrics(y_valid, y_valid_pred)
                print("Performance of our model (rmse, mae, mb, sd, r2, crps): \n\n", round(rmse_our, 2), round(mae_our, 2),
                      round(mean_our, 2), round(std_our, 2), round(r2_our, 2), round(valid_crps, 2))
                f.write('\n evaluation metrics (rmse, mae, mb, sd, r2, crps) on valid data for ' + reg + '=' + str(
                    round(rmse_our, 2)) + "," + str(round(mae_our, 2)) + "," +
                        str(round(mean_our, 2)) + "," + str(round(std_our, 2)) + "," + str(round(r2_our, 2)) + "," + str(round(valid_crps, 2)) +'\n')


            else:
                print("not enough data for the season: ", season_flag, "and lead: ", lead)

        f.close()


if __name__ == '__main__':
    main()




