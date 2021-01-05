import numpy as np
import pandas as pd
import os
import pickle
# from sklearn.externals import joblib
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from SolarForecasting.ModulesProcessing import collect_data,clean_data
from SolarForecasting.ModulesLearning import preprocessing as preprocess
from SolarForecasting.ModulesLearning import postprocessing as postprocess
from SolarForecasting.ModulesLearning import model as models
from SolarForecasting.ModulesLearning import clustering as clustering


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# All the variables and hyper-parameters

# city
city = 'Penn_State_PA'

# lead time
lead_times = [1] #from [1,2,3,4,5,6]

# season
seasons =['summer'] #from ['fall', 'winter', 'spring', 'summer', 'year']

# file locations
path_project = "C:\\Users\Shivendra\Desktop\SolarProject\solar_forecasting/"
path = path_project+"Data/"
folder_saving = path_project + "Models/"
folder_plots = path_project + "Plots/"
clearsky_file_path = path+'clear-sky/'+city+'_15mins_original.csv'

# scan all the features (except the flags)
features = ['year','month','day','hour','min','zen','dw_solar','uw_solar','direct_n','diffuse','dw_ir','dw_casetemp','dw_dometemp','uw_ir','uw_casetemp','uw_dometemp','uvb','par','netsolar','netir','totalnet','temp','rh','windspd','winddir','pressure']

# selected features for the study
# final_features = ['year','month','day','hour','MinFlag','zen','dw_solar','dw_ir','temp','rh','windspd','winddir','pressure','clear_ghi']
## exploring more features
final_features = ['year','month','day','hour','MinFlag','zen','dw_solar','uw_solar','direct_n','dw_ir','uw_ir','temp','rh','windspd','winddir','pressure', 'clear_ghi']
# features_to_cluster_on = ['dw_solar','dw_ir', 'temp','pressure', 'windspd']
features_to_cluster_on = ['dw_solar','temp']

# target or Y
target_feature = ['clearness_index']

# start and end month+year
startyear = 2005
endyear = 2009
startmonth = 9
endmonth = 8

# test year
testyear = 2008  # i.e all of Fall(Sep2008-Nov2008), Winter(Dec2008-Feb2009), Spring(Mar2009-May2009), Summer(June2009-Aug2009), year(Sep2008-Aug2009)

# hyperparameters
epochs = 500

# If clustering before prediction
n_clusters = 3 #6


def get_data():

    ## collect raw data
    years = [2005, 2006, 2007, 2008, 2009]
    object = collect_data.SurfradDataCollector(years, [city], path)

    object.download_data()


    ## cleanse data to get processed version
    for year in years:
        object = clean_data.SurfradDataCleaner(city, year, path)
        object.process(path_to_column_names='ModulesProcessing/column_names.pkl')


def include_previous_features(X, index_ghi):

    y_list = []
    previous_time_periods = [1,2,3,4,5,6,7,8] #[1,2]
    for l in previous_time_periods:
        print("rolling by: ", l)
        X_train_shifted = np.roll(X, l)
        # y_list.append(X_train_shifted)
        y_list.append(X_train_shifted[:, index_ghi])

    print(y_list)
    previous_time_periods_columns = np.column_stack(y_list)
    # print(previous_time_periods_columns[8:15])
    X = np.column_stack([X,previous_time_periods_columns])
    # max_lead = np.max(previous_time_periods)
    # X = X[max_lead:]
    print("X shape after adding prev features: ", X.shape)
    return X


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

    # 15 mins resolution
    dataset['MinFlag'] = dataset['min'].apply(preprocess.generateFlag)
    dataset = dataset.groupby(['year', 'month', 'day', 'hour', 'MinFlag']).mean()
    dataset.reset_index(inplace=True)
    print('dataset size on a 15 min resolution: ',len(dataset))

    # read the clear-sky values
    clearsky = pd.read_csv(clearsky_file_path, skiprows=37, delimiter=';')
    print("The columns of the clear sky file: ", clearsky.columns)

    # divide the observation period in form of year, month, day, hour, min (adding them as variables)
    clearsky[['year', 'month', 'day', 'hour', 'min']] = clearsky['# Observation period'].apply(preprocess.extract_time)
    clearsky['MinFlag'] = clearsky['min'].apply(preprocess.generateFlag)

    # merging the clear sky values with SURFAD input dataset
    df = dataset.merge(clearsky, on=['year', 'month', 'day', 'hour', 'MinFlag'], how='inner')

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
    df_final.reset_index(drop=True, inplace=True)
    print("after removing data points with 0 clear_ghi and selecting daytimes",len(df_final))

    # adjust the outliers for clearness index (no need to do this anymore -- since 0 clear_ghi is already taken care of)
    # df = preprocess.adjust_outlier_clearness_index(df)
    # print("\n\n after adjusting outliers of clearness index")
    # print(df.tail)


    for season_flag in seasons:
        ## ML_models_2008 is the folder to save results on testyear 2008
        ## creating different folder for different methods: nn for fully connected networks, rf for random forest etc.
        os.makedirs(folder_saving + season_flag + "/ML_models_2008/rf/modified_features/", exist_ok=True)
        f = open(folder_saving + season_flag + '/ML_models_2008/rf/modified_features/results.txt', 'a')

        for lead in lead_times:
            # create dataset with lead
            df_lead = preprocess.create_lead_dataset(df_final, lead, final_features, target_feature)

            # get the seasonal data you want
            df, test_startdate, test_enddate = preprocess.get_yearly_or_season_data(df_lead, season_flag, testyear)
            print("\n\n after getting seasonal data (test_startdate; test_enddate)", test_startdate, test_enddate)
            print(df.tail)



            # dividing into training and test set
            df_train, df_heldout = preprocess.train_test_spilt(df, season_flag, testyear)
            print("\n\n after dividing_training_test")
            print("train_set\n",len(df_train))
            print("test_set\n",len(df_heldout))


            if len(df_train)>0 and len(df_heldout)>0:
                # extract the X_train, y_train, X_test, y_test
                X_train, y_train, X_heldout, y_heldout, index_clearghi, index_ghi, index_zen,col_to_indices_mapping = preprocess.get_train_test_data(
                    df_train, df_heldout, final_features, target_feature)
                print("\n\n train and test df shapes ")
                print(X_train.shape, y_train.shape, X_heldout.shape, y_heldout.shape)
                print(col_to_indices_mapping)
                # filter the training to have only day values
                # X_train, y_train = preprocess.filter_dayvalues_and_zero_clearghi(X_train_all, y_train_all,
                #                                                                       index_zen, index_clearghi)

                ## including features from t-1 and t-2 timestamps
                X_train = include_previous_features(X_train, index_ghi)
                X_heldout = include_previous_features(X_heldout, index_ghi)
                print(X_train[:10])
                print("Final train size: ", X_train.shape, y_train.shape)
                print("Final heldout size: ", X_heldout.shape, y_heldout.shape)

                ## dividing the X_train data into train(70%)/valid(20%)/test(10%), the heldout data is kept hidden
                X_train, y_train = preprocess.shuffle(X_train, y_train)
                training_samples = int(0.7 * len(X_train))
                X_valid = X_train[training_samples:]
                X_train = X_train[:training_samples]
                y_valid = y_train[training_samples:]
                y_train = y_train[:training_samples]

                valid_samples = int(0.7*len(X_valid))
                X_test = X_valid[valid_samples:]
                X_valid = X_valid[:valid_samples]
                y_test = y_valid[valid_samples:]
                y_valid = y_valid[:valid_samples]


                print("train/valid/test sizes: ",len(X_train)," ",len(X_valid)," ", len(X_test))


                reg = "rf_modified_features" ## giving a name to the regression models -- useful when saving results

                # normalizing the Xtrain, Xvalid and Xtest data and saving the mean,std of train to normalize the heldout data later
                X_train, X_valid, X_test = preprocess.standardize_from_train(X_train, X_valid, X_test, folder_saving+season_flag + "/ML_models_2008/rf/modified_features/",reg, lead)


                y_train = np.reshape(y_train, -1)
                y_test = np.reshape(y_test, -1)
                y_valid = np.reshape(y_valid, -1)

                # call the gridSearch and saving model
                model = models.rfSearch_model(X_train, y_train)
                pickle.dump(model, open(
                    folder_saving + season_flag + "/ML_models_2008/rf/modified_features/model_at_lead_" + str(lead) + ".pkl",
                    "wb"))
                # for name, importance in zip(final_features[5:], model.best_estimator_.feature_importances_):
                #     print(name, "=", importance)
                # model = models.fnn_train(X_train, y_train, folder_saving+ season_flag + '/ML_models_2008/nn/', epochs=epochs, model_saved="FNN_single_task_at_lead_"+str(lead))

                ## When including clustering for these models
                # features_indices_to_cluster_on = [col_to_indices_mapping[f] for f in features_to_cluster_on]
                # print(features_indices_to_cluster_on)
                #
                # kmeans = clustering.clustering(X_train, features_indices_to_cluster_on,
                #                                n_clusters=n_clusters)
                # cluster_labels = kmeans.labels_
                #
                # print(Counter(cluster_labels))
                #
                # cluster_labels_valid = clustering.get_closest_clusters(X_valid, kmeans, features_indices_to_cluster_on)
                # cluster_labels_test = clustering.get_closest_clusters(X_test, kmeans, features_indices_to_cluster_on)
                #
                # X_train, X_valid, X_test = clustering.normalizing_per_cluster(X_train, X_valid, X_test, cluster_labels,
                #                                                               cluster_labels_valid, cluster_labels_test,folder_saving+season_flag + "/ML_models_2008/rf/clustering/",reg, lead)


                # model_dict = clustering.train(X_train,y_train, cluster_labels, n_clusters)
                # pickle.dump(kmeans, open(
                #     folder_saving + season_flag + "/ML_models_2008/rf/clustering/kmeans_at_lead_" + str(lead) + ".pkl",
                #     "wb"))
                # pickle.dump(model_dict, open(
                #     folder_saving + season_flag + "/ML_models_2008/rf/clustering/dict_of_models_" + str(lead) + ".pkl",
                #     "wb"))

                f.write("\n" + city + " at Lead " + str(lead) + " and " + season_flag + " Season")
                f.write("\n best parameter found: ")
                f.write(str(model.best_params_))
                # y_pred = models.fnn_test(X_test, model.best_estimator_)
                # y_pred = clustering.cluster_and_predict(X_test, model_dict, cluster_labels_test)
                y_pred = model.predict(X_test)
                #
                # y_valid_pred = models.fnn_test(X_valid, model.best_estimator_)
                # y_valid_pred = clustering.cluster_and_predict(X_valid, model_dict, cluster_labels_valid)
                y_valid_pred = model.predict(X_valid)

                y_pred = np.reshape(y_pred, -1)
                y_valid_pred = np.reshape(y_valid_pred, -1)

                print("\n" + city + " at Lead " + str(lead) + " and " + season_flag + " Season")

                print("##########VALID##########")
                rmse_our, mae_our, mb_our, r2_our = postprocess.evaluation_metrics(y_valid, y_valid_pred)
                print("Performance of our model (rmse, mae, mb, r2): \n\n", round(rmse_our, 1), round(mae_our, 1),
                      round(mb_our, 1), round(r2_our, 1))
                f.write('\n evaluation metrics (rmse, mae, mb, r2) on valid data for ' + reg + '=' + str(round(rmse_our, 1))+ "," + str(round(mae_our, 1)) + ","+
                      str(round(mb_our, 1)) + "," + str(round(r2_our, 1)) + '\n')

                print("##########Test##########")
                rmse_our, mae_our, mb_our, r2_our = postprocess.evaluation_metrics(y_test, y_pred)
                # print("Performance of our model (rmse, mae, mb, r2): \n\n", round(rmse_our, 1), round(mae_our, 1),
                #       round(mb_our, 1), round(r2_our, 1))
                f.write('\n evaluation metrics (rmse, mae, mb, r2) on test data for ' + reg + '=' + str(
                    round(rmse_our, 1)) + "," + str(round(mae_our, 1)) + "," +
                        str(round(mb_our, 1)) + "," + str(round(r2_our, 1)) + '\n')


            else:
                print("not enough data for the season: ", season_flag, "and lead: ", lead)

        f.close()


if __name__=='__main__':
    main()


