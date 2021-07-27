import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from datetime import timedelta


from SolarForecasting.ModulesProcessing import collect_data, clean_data
from SolarForecasting.ModulesLearning import preprocessing as preprocess
from SolarForecasting.ModulesLearning import postprocessing as postprocess
from SolarForecasting.ModulesLearning.ModulesCNN import train as cnn
from SolarForecasting.ModulesLearning.ModuleLSTM import train as tranformers



pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# All the variables and hyper-parameters

# city
city = 'Fort_Peck_MT'#'Sioux_Falls_SD'

# lead time
lead_times = [2,4,8,12,12*2,12*3,12*4,12*5,12*6]

# season
seasons =['year'] #,'fall', 'winter', 'spring', 'summer']
res = '5min' #15min

# file locations

# path_desktop = "C:\\Users\Shivendra\Desktop\SolarProject\solar_forecasting/"
path_local = "/Users/saumya/Desktop/SolarProject/"
path_cluster = "/pl/active/machinelearning/Solar_forecasting_project/"
path_project = path_local
path = path_project+"Data/"
folder_saving = path_project + city+"/Models/"
folder_plots = path_project + city+"/Plots/"
clearsky_file_path = path+'clear-sky/'+city+'_1min_original.csv'


# scan all the features (except the flags)
features = ['year','month','day','hour','min','zen','dw_solar','uw_solar','direct_n','diffuse','dw_ir','dw_casetemp','dw_dometemp','uw_ir','uw_casetemp','uw_dometemp','uvb','par','netsolar','netir','totalnet','temp','rh','windspd','winddir','pressure']

# selected features for the study
#final_features = ['year','month','day','hour','MinFlag','zen','dw_solar','dw_ir','temp','rh','windspd','winddir','pressure','clear_ghi']
# final_features = ['year','month','day','hour','MinFlag','zen','dw_solar','uw_solar','direct_n','diffuse','dw_ir','dw_casetemp','dw_dometemp','uw_ir','uw_casetemp','uw_dometemp','uvb','par','temp','rh','windspd','winddir','pressure', 'clear_ghi']
## ## selected features for the study (exploring multiple combinations)
final_features = ['year','month','day','hour','MinFlag','zen','dw_solar','uw_solar','direct_n','dw_ir','uw_ir','temp','rh','windspd','winddir','pressure', 'clear_ghi']
# final_features = ['year','month','day','hour','MinFlag','zen','dw_solar','direct_n','dw_ir','temp','windspd','winddir','pressure', 'clear_ghi']

# target or Y
target_feature = ['clearness_index']

# start and end month+year
startyear = 2015 #2005
endyear = 2018 #2009
startmonth = 9
endmonth = 8

# test year
# testyear = 2008  # i.e all of Fall(Sep2008-Nov2008), Winter(Dec2008-Feb2009), Spring(Mar2009-May2009), Summer(June2009-Aug2009), year(Sep2008-Aug2009)
testyear = 2017

# hyperparameters


n_timesteps = 144#72#24*7 for tcn # 144 for 5-min reoslution
# n_output_steps = len(lead_times)
n_features = 15#12 for SAND #15 for tcn(taking 12(even) features for mulit-head and transformers)
quantile = True #True

#hyperparameters for the multi-attention model

n_layers = 1 #2 #3
factor = 12 #12
num_heads = 4 #4
d_model = 64 #128
batch_size = 8 #32 #16 #16 


epochs = 300 #250
lr = 1e-5 #1e-5 #1e-4

alphas = np.arange(0.05, 1.0, 0.05)
# alphas = np.arange(0.05, 1, 0.225)
q50 = 9  # 2


def get_data():

    ## collect raw data
    years = [2015, 2016, 2017, 2018] #[2005, 2006, 2007, 2008, 2009]
    object = collect_data.SurfradDataCollector(years, [city], path)

    object.download_data()


    ## cleanse data to get processed version
    for year in years:
        object = clean_data.SurfradDataCleaner(city, year, path)
        object.process(path_to_column_names='ModulesProcessing/column_names.pkl')


def include_previous_features(X, index_ghi):
    '''
    Features at time t includes features from previous n_timesteps
    '''
    y_list = []
    previous_time_periods = list(range(1, n_timesteps+1))
    # dw_solar = X[:, index_ghi]

    for l in previous_time_periods:
        # print("rolling by: ", l)
        X_train_shifted = np.roll(X, l)
        y_list.append(X_train_shifted)
        # y_list.append(dw_solar_rolled)
    y_list = y_list[::-1]
    # print(y_list)
    previous_time_periods_columns = np.column_stack(y_list)
    X = np.column_stack([previous_time_periods_columns, X])
    # X = np.transpose(np.array(y_list), ((1, 0, 2)))
    # max_lead = np.max(previous_time_periods)
    # X = X[max_lead:]
    print("X shape after adding prev features: ", X.shape)
    return X


def create_mulitple_lead_dataset(dataframe, final_set_of_features, target):
    dataframe_lead = dataframe[final_set_of_features]
    target = np.asarray(dataframe[target])

    y_list = []
    for lead in lead_times:
    # for lead in range(1,n_output_steps+1):
        print("rolling by: ",lead)
        target = np.roll(target, -lead)
        y_list.append(target)

    dataframe_lead['clearness_index'] = np.column_stack(y_list).tolist()
    dataframe_lead['clearness_index'] = dataframe_lead['clearness_index'].apply(tuple)
    max_lead = np.max(lead_times)
    dataframe_lead = dataframe_lead[:len(dataframe_lead) - max_lead]

    # dataframe_lead['clearness_index'].values[-max_lead:] = np.nan
    print(dataframe_lead['clearness_index'])

    # remove rows which have any value as NaN
    # dataframe_lead = dataframe_lead.dropna()
    print("*****************")
    print("dataframe with lead size: ", len(dataframe_lead))
    return dataframe_lead

# def create_labels_for_wavenet(dataframe, lead, final_set_of_features, target):
#
#     dataframe_lead = dataframe[final_set_of_features]
#
#     target = np.asarray(dataframe[target]).tolist()
#     target = [item for sublist in target for item in sublist]
#
#     dataframe_lead = dataframe_lead[:len(dataframe_lead)-lead]
#     dataframe_lead['clearness_index'] = [[] for _ in range(len(dataframe_lead))]
#     for i in range(len(dataframe_lead)):
#         # dataframe_lead['clearness_index'][i]=[]
#         for t in range(i-n_timesteps+1,i+1):
#             dataframe_lead['clearness_index'][i].append(target[t+lead])
#
#         dataframe_lead['clearness_index'][i] = tuple(dataframe_lead['clearness_index'][i])
#         # remove rows which have any value as NaN
#     # dataframe_lead['clearness_index'] = dataframe_lead['clearness_index'].apply(tuple)
#     print(dataframe_lead.iloc[0]['clearness_index'])
#     print("*****************")
#     print("dataframe with lead size: ",len(dataframe_lead))
#     return dataframe_lead
#



def main():

    ## pre-processing steps

    # extract the input data files (SURFAD data)
    processed_file_path = path + 'processed/' + city
    if not os.path.isdir(processed_file_path):
        get_data()
    combined_csv = preprocess.extract_frame(processed_file_path)
    print("The columns of the initial data file: ", combined_csv.columns)

    # extract the features from the input
    dataset = combined_csv[features]
    print('dataset size: ',len(dataset))

    #1hour resolution #15 mins resolution
    dataset['MinFlag'] = dataset['min'].apply(preprocess.generateFlag)
    dataset = dataset.groupby(['year', 'month', 'day', 'hour', 'MinFlag']).mean()
    # dataset = dataset.groupby(['year', 'month', 'day', 'hour']).mean()
    dataset.reset_index(inplace=True)
    print('dataset size: ',len(dataset))

    print(dataset.isnull().values.any())

    # read the clear-sky values
    clearsky = pd.read_csv(clearsky_file_path, skiprows=37, delimiter=';')
    print("The columns of the clear sky file: ", clearsky.columns)

    # divide the observation period in form of year, month, day, hour, min (adding them as variables)
    clearsky[['year', 'month', 'day', 'hour', 'min']] = clearsky['# Observation period'].apply(preprocess.extract_time)
    # clearsky = clearsky.groupby(['year', 'month', 'day', 'hour']).mean()
    clearsky['MinFlag'] = clearsky['min'].apply(preprocess.generateFlag)
    clearsky = clearsky.groupby(['year', 'month', 'day', 'hour', 'MinFlag']).mean()
    print("clearsky rows before merging: ", len(clearsky))

    # merging the clear sky values with SURFAD input dataset
    df = dataset.merge(clearsky, on=['year', 'month', 'day', 'hour', 'MinFlag'], how='inner')
    # df = dataset.merge(clearsky, on=['year', 'month', 'day', 'hour'], how='inner')

    # renaming the clear sky GHI
    df = df.rename(columns={'Clear sky GHI': 'clear_ghi'})
    print("stats of all raw features/columns")
    print(df.describe())


    # selecting only the required columns
    df = df[final_features]

    # get dataset for the study period
    df = preprocess.extract_study_period(df,startmonth, startyear, endmonth, endyear)
    print("\n\n after extracting study period")
    df.reset_index(drop=True, inplace=True)
    print(df.tail)


    # ## removing outliers from this dataset and then removing nan rows
    # df = preprocess.remove_negative_values(df, final_features[5:])
    # df = df.dropna()

    ## convert negatives to 0 for all features
    df[df<0] = 0
    print("stats of selected features/columns after converting all negatives to 0")
    print(df.describe())

    # adjust the boundary values (no need to do this anymore -- will drop the rows with 0 clear_ghi later)
    # df = preprocess.adjust_boundary_values(df)


    # adding the clearness index and dropping rows with 0 clear_ghi and taking only daytimes
    df = df[df['clear_ghi'] > 0]
    df_final = df[df['zen']<85]
    df_final['clearness_index'] = df_final['dw_solar'] / df_final['clear_ghi']
    # df_final['clearness_index'] = df_final['dw_solar']
    df_final.reset_index(drop=True, inplace=True)
    print("after removing data points with 0 clear_ghi and selecting daytimes",len(df_final))

    # df_lead = create_mulitple_lead_dataset(df_final, final_features, target_feature)


    reg = "dcnn_with_tcn_without_attention_5mins"






    for season_flag in seasons:
        ## ML_models_2008 is the folder to save results on testyear 2008
        ## creating different folder for different methods: nn for fully connected networks, rf for random forest etc.
        os.makedirs(
            folder_saving + season_flag + "/ML_models_"+str(testyear)+"/cnn/"+str(res)+"/"+reg+"/",
            exist_ok=True)
        f = open(folder_saving + season_flag + "/ML_models_"+str(testyear)+"/cnn/"+str(res)+"/"+reg+"/results.txt", 'a')

        for lead in lead_times:
            # create dataset with lead
            df_lead = preprocess.create_lead_dataset(df_final, lead, final_features, target_feature)
            # df_lead = create_labels_for_wavenet(df_final, lead, final_features, target_feature)
            df_lead = df_lead[:len(df_lead)-lead]

                # get the seasonal data you want

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
                X_train, y_train, X_heldout, y_heldout, index_clearghi, index_ghi, index_zen, col_to_indices_mapping = preprocess.get_train_test_data(
                    df_train, df_heldout, final_features, target_feature)#, lead)
                print("\n\n train and test df shapes ")
                print(X_train.shape, y_train.shape, X_heldout.shape, y_heldout.shape)


                # including features from prev imestamps
                X_train = include_previous_features(X_train, index_ghi)
                X_heldout = include_previous_features(X_heldout, index_ghi)

                X_train = X_train[n_timesteps:,:]
                X_heldout = X_heldout[n_timesteps:, :]
                y_train = y_train[n_timesteps:, :]
                y_heldout = y_heldout[n_timesteps:, :]

                print("Final train size: ", X_train.shape, y_train.shape)
                print("Final heldout size: ", X_heldout.shape, y_heldout.shape)

                ## dividing the X_train data into train(70%)/valid(20%)/test(10%), the heldout data is kept hidden

                X_train, X_test, y_train, y_test = train_test_split(
                    X_train, y_train, test_size=0.3, random_state=42)#42
                X_valid, X_test, y_valid, y_test = train_test_split(
                    X_test, y_test, test_size=0.3, random_state=42)#42


                print("train/valid/test sizes: ", len(X_train), " ", len(X_valid), " ", len(X_test))


                # normalizing the Xtrain, Xvalid and Xtest data and saving the mean,std of train to normalize the heldout data later
                X_train, X_valid, X_test = preprocess.standardize_from_train(X_train, X_valid, X_test,index_ghi,index_clearghi, len(col_to_indices_mapping),
                                                                             folder_saving + season_flag + "/ML_models_"+str(testyear)+"/cnn/"+str(res)+"/"+reg+"/", lead = lead)

                # f.write("epochs = " + str(epochs) + '\n')
                # f.write("batch_size = " + str(batch_size) + '\n')
                # f.write("learning rate = " + str(lr) + '\n')
                # f.write("n_layers = " + str(n_layers) + '\n')
                # f.write("factor = " + str(factor) + '\n')
                # f.write("num_heads = " + str(num_heads) + '\n')
                # f.write("d_model = " + str(d_model) + '\n')
                f.write("seq_len = " + str(n_timesteps) + '\n')
                f.write("total features = " + str(n_features)+ '\n')
                f.write("alphas = " + str(len((alphas))) + '\n')


                cnn.train_DCNN_with_attention(quantile,X_train, y_train, X_valid, y_valid, n_timesteps+1, n_features,
                                                                                         folder_saving + season_flag + "/ML_models_" + str(
                                                                                             testyear) + "/cnn/" + str(
                                                                                             res) + "/" + reg + "/",
                                                                                         model_saved="dcnn_lag_for_lead_" + str(
                                                                                             lead))
                y_pred, y_valid_pred, valid_crps, test_crps = cnn.test_DCNN_with_attention(quantile, X_valid, y_valid,
                                                                                         X_test, y_test,
                                                                                         n_timesteps + 1, n_features,
                                                                                         folder_saving + season_flag + "/ML_models_" + str(
                                                                                             testyear) + "/cnn/" + str(
                                                                                             res) + "/" + reg + "/",
                                                                                         model_saved="dcnn_lag_for_lead_" + str(
                                                                                             lead))  # "multi_horizon_dcnn", n_outputs=n_output_steps)

                # tranformers.train_transformer(quantile, X_train, y_train, X_valid, y_valid, n_timesteps+1, n_features, n_layers, factor, num_heads, d_model, batch_size, epochs, lr,alphas, q50,
                #                            folder_saving + season_flag + "/ML_models_"+str(testyear)+"/cnn/"+str(res)+"/"+reg+"/",model_saved ="dcnn_lag_for_lead_" + str(lead)) #"multi_horizon_dcnn", n_outputs=n_output_steps)
                #
                # y_pred, y_valid_pred, valid_crps, test_crps  = tranformers.test_transformer(quantile, X_valid, y_valid, X_test, y_test, n_timesteps+1, n_features,n_layers, factor, num_heads, d_model,alphas, q50,
                #                                folder_saving + season_flag + "/ML_models_"+str(testyear)+"/cnn/"+str(res)+"/"+reg+"/",model_saved = "dcnn_lag_for_lead_" + str(lead))#"multi_horizon_dcnn", n_outputs=n_output_steps)
                #


            # for i in range(n_output_steps):
            #
            #     lead = lead_times[i]
            #     y_test_for_this_lead = y_test[:,i]
            #     y_valid_for_this_lead = y_valid[:,i]
            #     y_pred_for_this_lead = y_pred[i][:,q50].cpu().detach().numpy()
            #     y_valid_pred_for_this_lead = y_valid_pred[i][:,q50].cpu().detach().numpy()
            #     valid_crps_for_this_lead = valid_crps[i]
            #     test_crps_for_this_lead = test_crps[i

            #

                f.write("\n" + city + " at Lead " + str(lead) + " and " + season_flag + " Season")
                print("\n" + city + " at Lead " + str(lead) + " and " + season_flag + " Season")


                print("##########VALID##########")
                rmse_our, mae_our, mean_our,std_our,r2_our = postprocess.evaluation_metrics(y_valid, y_valid_pred)
                print("Performance of our model (rmse, mae, mb, sd, r2, crps): \n\n", round(rmse_our, 2), round(mae_our, 2),
                      round(mean_our, 2), round(std_our, 2), round(r2_our, 2), round(valid_crps, 2))
                f.write('\n evaluation metrics (rmse, mae, mb, sd, r2, crps) on valid data for ' + reg + '=' + str(
                    round(rmse_our, 2)) + "," + str(round(mae_our, 2)) + "," +
                        str(round(mean_our, 2)) + "," + str(round(std_our, 2)) + "," + str(round(r2_our, 2)) + "," + str(round(valid_crps, 2)) +'\n')

                print("##########Test##########")
                rmse_our, mae_our, mean_our, std_our, r2_our = postprocess.evaluation_metrics(y_test, y_pred)
                print("Performance of our model (rmse, mae, mb, sd, r2, crps): \n\n", round(rmse_our, 2), round(mae_our, 2),
                      round(mean_our, 2), round(std_our, 2), round(r2_our, 2), round(test_crps, 2))
                f.write('\n evaluation metrics (rmse, mae, mb, sd, r2, crps) on test data for ' + reg + '=' + str(
                    round(rmse_our, 2)) + "," + str(round(mae_our, 2)) + "," +
                        str(round(mean_our, 2)) + "," + str(round(std_our, 2)) + "," + str(
                    round(r2_our, 2)) + "," + str(round(test_crps, 2)) + '\n')


            else:
                print("not enough data for the season: ", season_flag, "and lead: ", lead)

        f.close()


if __name__ == '__main__':
    main()




