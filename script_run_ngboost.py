import numpy as np
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from ngboost import NGBRegressor
import matplotlib.pyplot as plt
from SolarForecasting.ModulesProcessing import collect_data,clean_data
from SolarForecasting.ModulesLearning import preprocessing as preprocess
from SolarForecasting.ModulesLearning import postprocessing as postprocess
from ngboost.scores import CRPS, MLE
import scipy as sp
from scipy.stats import norm as dist


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# All the variables and hyper-parameters

# city
city = 'Sioux_Falls_SD'

# lead time i.e how much in advance you want to make a prediction (lead of 4 corresponds to 1 hour..since the data is at 15min resolution)
lead_times =  [2,4,8,12,12*2,12*3,12*4,12*5,12*6]

# season
seasons =['year'] #from ['fall', 'winter', 'spring', 'summer', 'year']
res = '5min'

# file locations
# path_desktop = "C:\\Users\Shivendra\Desktop\SolarProject\solar_forecasting/"
path_local = "/Users/saumya/Desktop/SolarProject/"
path_cluster = "/pl/active/machinelearning/Solar_forecasting_project/"
path_project = path_local
path = path_project+"Data/"
folder_saving = path_project + city+"/Models/"
folder_plots = path_project + city+"/Plots/"
clearsky_file_path = path+'clear-sky/'+city+'_1min_original.csv'

# scan all the features
features = ['year','month','day','hour','min','zen','dw_solar','uw_solar','direct_n','diffuse','dw_ir','dw_casetemp','dw_dometemp','uw_ir','uw_casetemp','uw_dometemp','uvb','par','netsolar','netir','totalnet','temp','rh','windspd','winddir','pressure']

# selected features for the study
# final_features = ['year','month','day','hour','MinFlag','zen','dw_solar','dw_ir','temp','rh','windspd','winddir','pressure','clear_ghi']

## selected features for the study
final_features = ['year','month','day','hour','MinFlag','zen','dw_solar','uw_solar','direct_n','dw_ir','uw_ir','temp','rh','windspd','winddir','pressure', 'clear_ghi']


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
n_timesteps = 0#24
n_features = 15 #(after including month and hour)


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
    print("orig data stats", dataset.describe())
     #15 mins resolution
    dataset['MinFlag'] = dataset['min'].apply(preprocess.generateFlag)
    dataset = dataset.groupby(['year', 'month', 'day', 'hour', 'MinFlag']).mean()
    print()
    # dataset = dataset.groupby(['year', 'month', 'day', 'hour']).mean()
    dataset.reset_index(inplace=True)
    print('dataset size : ',len(dataset))

    print(dataset.isnull().values.any())

    # read the clear-sky values
    clearsky = pd.read_csv(clearsky_file_path, skiprows=37, delimiter=';')
    print("The columns of the clear sky file: ", clearsky.columns)

    # divide the observation period in form of year, month, day, hour, min (adding them as variables)
    clearsky[['year', 'month', 'day', 'hour', 'min']] = clearsky['# Observation period'].apply(preprocess.extract_time)
    # clearsky = clearsky.groupby(['year', 'month', 'day', 'hour']).mean()
    clearsky['MinFlag'] = clearsky['min'].apply(preprocess.generateFlag)
    print("clearsky before converting to 5min res")
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


    ## convert negatives to 0 for all features
    df[df<0] = 0
    print("stats of selected features/columns after converting all negatives to 0")
    print(df.describe())


    # adding the clearness index and dropping rows with 0 clear_ghi and taking only daytimes
    df = df[df['clear_ghi'] > 0]
    df_final = df[df['zen']<85]
    df_final['clearness_index'] = df_final['dw_solar'] / df_final['clear_ghi']
    # df_final['clearness_index'] = df_final['dw_solar']
    df_final.reset_index(drop=True, inplace=True)
    print("after removing data points with 0 clear_ghi and selecting daytimes",len(df_final))
    # print(df_final.describe())

    # Plotting time series
    # x = np.asarray(range(df_final.shape[0]))
    # plt.figure(figsize=(20, 10))
    # plt.plot(x, df_final.dw_solar.values, label="GHI values")
    # plt.plot(x, df_final.clear_ghi.values, label="clearGHI index")
    # plt.legend(loc="upper left")
    # plt.savefig("time series of clearGHI and GHI")
    # plt.clf()
    # x = np.asarray(range(60))
    # plt.figure(figsize=(20, 10))
    # plt.plot(x, df_final.dw_solar.values[:60], label="GHI values")
    # plt.plot(x, df_final.clear_ghi.values[:60], label="clearGHI index")
    # plt.legend(loc="upper left")
    # plt.savefig("time series of clearGHI and GHI zoomed")
    # plt.clf()
    # x = np.asarray(range(60))
    # plt.figure(figsize=(20, 10))
    # plt.plot(x, df_final.clearness_index.values[:60], label="Clearness Index values")
    # plt.legend(loc="upper left")
    # plt.savefig("time series of clearness index zoomed")
    # plt.clf()
    #
    reg = "ngboost_without_lag_5mins"  ## giving a name to the regression models -- useful when saving results

    for season_flag in seasons:
        os.makedirs(folder_saving + season_flag + "/ML_models_"+str(testyear)+"/probabilistic/"+str(res)+"/"+reg+"/", exist_ok=True)
        f = open(folder_saving + season_flag + "/ML_models_"+str(testyear)+"/probabilistic/"+str(res)+"/"+reg+"//results.txt", 'a')

        for lead in lead_times:
            # create dataset with lead
            df_lead = preprocess.create_lead_dataset(df_final, lead, final_features, target_feature)
            df_lead = df_lead[:len(df_lead) - lead]

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
                    df_train, df_heldout, final_features, target_feature, lead)
                print("\n\n train and test df shapes ")
                print(X_train.shape, y_train.shape, X_heldout.shape, y_heldout.shape)


                # including features from prev imestamps - didn't need to do that for NgBoost
                # X_train = include_previous_features(X_train, index_ghi)
                # X_heldout = include_previous_features(X_heldout, index_ghi)
                #
                # X_train = X_train[n_timesteps:, :]
                # X_heldout = X_heldout[n_timesteps:, :]
                # y_train = y_train[n_timesteps:, :]
                # y_heldout = y_heldout[n_timesteps:, :]

                print("Final train size: ", X_train.shape, y_train.shape)
                print("Final heldout size: ", X_heldout.shape, y_heldout.shape)

                ## dividing the X_train data into train(70%)/valid(20%)/test(10%), the heldout data is kept hidden
                X_train, X_test, y_train, y_test = train_test_split(
                    X_train, y_train, test_size=0.3, random_state=42)
                X_valid, X_test, y_valid, y_test = train_test_split(
                    X_test, y_test, test_size=0.3, random_state=42)

                print("train/valid/test sizes: ",len(X_train)," ",len(X_valid)," ", len(X_test))

                # normalizing the Xtrain, Xvalid and Xtest data and saving the mean,std of train to normalize the heldout data later
                X_train, X_valid, X_test = preprocess.standardize_from_train(X_train, X_valid, X_test, index_ghi, index_clearghi, len(col_to_indices_mapping), folder_saving+season_flag + "/ML_models_"+str(testyear)+"/probabilistic/"+str(res)+"/"+reg+"/",lead)


                y_train = np.reshape(y_train, -1)
                y_test = np.reshape(y_test, -1)
                y_valid = np.reshape(y_valid, -1)

                ## model built and saved (commented if it's already built and just being loaded as below)
                model = NGBRegressor(n_estimators=2000).fit(X_train, y_train)

                pickle.dump(model, open(
                    folder_saving + season_flag + "/ML_models_"+str(testyear)+"/probabilistic/"+str(res)+"/"+reg+"/model_at_lead_" + str(lead) + ".pkl",
                    "wb"))

                # with open(folder_saving + season_flag + "/ML_models_"+str(testyear)+"/probabilistic/"+str(res)+"/"+reg+"//model_at_lead_" + str(lead) + ".pkl", 'rb') as file:
                #     model = pickle.load(file)

                print(model)
                f.write("\n" + city + " at Lead " + str(lead) + " and " + season_flag + " Season")

                ## making predictions
                y_pred = model.predict(X_test)
                y_valid_pred = model.predict(X_valid)

                y_pred = np.reshape(y_pred, -1)
                y_valid_pred = np.reshape(y_valid_pred, -1)

                print("\n" + city + " at Lead " + str(lead) + " and " + season_flag + " Season")


                ## Obtain the evaluation metrics (other than crps, in case it can be used later)
                print("##########VALID##########")
                rmse_our, mae_our, mean_our, std_our, r2_our = postprocess.evaluation_metrics(y_valid,
                                                                                              y_valid_pred)
                print("Performance of our model (rmse, mae, mb,sd, r2): \n\n", round(rmse_our, 2), round(mae_our, 2),
                      round(mean_our, 2), round(std_our, 2), round(r2_our, 2))
                f.write('\n evaluation metrics (rmse, mae, mb,sd, r2) on valid data for ' + reg + '=' + str(
                    round(rmse_our, 2)) + "," + str(round(mae_our, 2)) + "," +
                        str(round(mean_our, 2)) + "," + str(round(std_our, 2)) + "," + str(round(r2_our, 2)) + '\n')

                print("##########Test##########")
                rmse_our, mae_our, mean_our, std_our, r2_our = postprocess.evaluation_metrics(y_test,
                                                                                              y_pred)
                print("Performance of our model (rmse, mae, mb,sd, r2): \n\n", round(rmse_our, 2), round(mae_our, 2),
                      round(mean_our, 2), round(std_our, 2), round(r2_our, 2))
                f.write('\n evaluation metrics (rmse, mae, mb,sd, r2) on test data for ' + reg + '=' + str(
                    round(rmse_our, 2)) + "," + str(round(mae_our, 2)) + "," +
                        str(round(mean_our, 2)) + "," + str(round(std_our, 2)) + "," + str(round(r2_our, 2)) + '\n')


                ## get CRPS score
                crps_valid = get_crps_for_ngboost(model, X_valid, y_valid)
                crps_test = get_crps_for_ngboost(model, X_test, y_test)

                f.write('\n CRPS score on valid data for lead ' + str(lead) + '=' + str(
                    round(crps_valid, 2)) + '\n')
                f.write('\n CRPS score on test data for lead ' + str(lead) + '=' + str(
                    round(crps_test, 2)) + '\n')




            else:
                    print("not enough data for the season: ", season_flag, "and lead: ", lead)

        f.close()

if __name__=='__main__':
    main()


