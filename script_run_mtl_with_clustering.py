import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pickle
import datetime
from collections import Counter
from sklearn.model_selection import train_test_split
from SolarForecasting.ModulesProcessing import collect_data,clean_data
from SolarForecasting.ModulesLearning import preprocessing as preprocess
from SolarForecasting.ModulesLearning import postprocessing as postprocess
from SolarForecasting.ModulesMultiTaskLearning import train_model,test_and_save_predictions
from SolarForecasting.ModulesLearning import clustering as clustering


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# All the variables and hyper-parameters

# city
city = 'Penn_State_PA'

# lead time
lead_times = [1,2,3,4,5,6] #from [1,2,3,4,5,6]

# season
seasons =['summer'] #from ['fall', 'winter', 'spring', 'summer', 'year']

# file locations
path_project = "C:\\Users\Shivendra\Desktop\SolarProject\solar_forecasting/"
path = path_project+"Data/"
primary_folder_saving = path_project + "Models/"
folder_plots = path_project + "Plots/"
clearsky_file_path = path+'clear-sky/'+city+'_15mins_original.csv'

# scan all the features (except the flags)
features = ['year','month','day','hour','min','zen','dw_solar','uw_solar','direct_n','diffuse','dw_ir','dw_casetemp','dw_dometemp','uw_ir','uw_casetemp','uw_dometemp','uvb','par','netsolar','netir','totalnet','temp','rh','windspd','winddir','pressure']

# selected features for the study
# final_features = ['year','month','day','hour','MinFlag','zen','dw_solar','dw_ir','temp','rh','windspd','winddir','pressure','clear_ghi']
## explore more features, but a lot of them are categorical so might want to remove those
final_features = ['year','month','day','hour','MinFlag','zen','dw_solar','uw_solar','direct_n','dw_ir','uw_ir','temp','rh','windspd','winddir','pressure', 'clear_ghi']
# features_to_cluster_on = ['dw_solar','dw_ir', 'temp','pressure', 'windspd']
features_to_cluster_on = ['dw_solar','temp']
n_clusters = 3 #6
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
bs = 32
epochs = 750
lr_list = [0.0001] #0.001 originally
hidden_sizes_list = [[128,64]]   ## whatever I get from the STL model for that forecast period
task_specific_hidden_sizes_list = [None,[8],[16],[32]]
weight_decay_list = [0.005] #1e-5 originally

reg = "mtl_fine_tuned"


def get_data():

    ## collect raw data
    years = [2005, 2006, 2007, 2008, 2009]
    object = collect_data.SurfradDataCollector(years, [city], path)

    object.download_data()


    ## cleanse data to get processed version
    for year in years:
        object = clean_data.SurfradDataCleaner(city, year, path)
        object.process(path_to_column_names='ModulesProcessing/column_names.pkl')


def include_previous_features(X):

    y_list = []
    previous_time_periods = [1,2]
    for l in previous_time_periods:
        print("rolling by: ", l)
        X_train_shifted = np.roll(X, l)
        y_list.append(X_train_shifted)

    previous_time_periods_columns = np.column_stack(y_list)
    X = np.column_stack([X,previous_time_periods_columns])
    # max_lead = np.max(previous_time_periods)
    # X = X[max_lead:]
    print("X shape after adding t-1,t-2 features: ", X.shape)
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

    # adjust the outliers for clearness index (no need to do this anymore -- since I am already taking care of 0 clear_ghi)
    # df = preprocess.adjust_outlier_clearness_index(df)
    # print("\n\n after adjusting outliers of clearness index")
    # print(df.tail)


    for season_flag in seasons:
        # current date
        now = datetime.datetime.now()
        date_with_hr_sec = now.strftime("%Y_%m_%d_%H_%M_%S")
        folder_saving = primary_folder_saving + season_flag + "/" + reg + "/" + date_with_hr_sec + "/"
        os.makedirs(folder_saving, exist_ok=True)
        # f = open(folder_saving + 'results.txt', 'a')

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
                X_train, y_train, X_heldout, y_heldout, index_clearghi, index_ghi, index_zen, col_to_indices_mapping = preprocess.get_train_test_data(
                    df_train, df_heldout, final_features, target_feature)
                print("\n\n train and test df shapes ")
                print(X_train.shape, y_train.shape, X_heldout.shape, y_heldout.shape)


                ## including features from t-1 and t-2 timestamps
                X_train = include_previous_features(X_train)
                X_heldout = include_previous_features(X_heldout)

                print("Final train size: ", X_train.shape, y_train.shape)
                print("Final heldout size: ", X_heldout.shape, y_heldout.shape)


                ## dividing the X_train data into train(70%)/valid(15/5)/test(15%), the heldout data is kept hidden
                X_train, y_train = preprocess.shuffle(X_train, y_train)
                training_samples = int(0.7 * len(X_train))
                X_valid = X_train[training_samples:]
                X_train = X_train[:training_samples]
                y_valid = y_train[training_samples:]
                y_train = y_train[:training_samples]

                valid_samples = int(0.7 * len(X_valid))
                X_test = X_valid[valid_samples:]
                X_valid = X_valid[:valid_samples]
                y_test = y_valid[valid_samples:]
                y_valid = y_valid[:valid_samples]

                print("train/valid/test sizes: ", len(X_train), " ", len(X_valid), " ", len(X_test))

                y_train  = np.reshape(y_train, -1)
                y_test = np.reshape(y_test, -1)
                y_valid = np.reshape(y_valid, -1)

                input_size = X_train.shape[1]

                features_indices_to_cluster_on = [col_to_indices_mapping[f] for f in features_to_cluster_on]
                print(features_indices_to_cluster_on)
                # kmeans = clustering.clustering(X_train, features_indices_to_cluster_on,
                #                                                n_clusters=n_clusters)
                # pickle.dump(kmeans, open(folder_saving + "kmeans_for_lead_"+str(lead)+".pkl", "wb"))
                with open(primary_folder_saving + season_flag + "/" + reg + "/2020_12_17_16_14_01/" + "kmeans_for_lead_"+str(lead)+".pkl","rb") as clusterfile:
                    kmeans = pickle.load(clusterfile)
                cluster_labels = kmeans.labels_

                c = Counter(cluster_labels)

                cluster_labels_valid = clustering.get_closest_clusters(X_valid, kmeans, features_indices_to_cluster_on)
                cluster_labels_test = clustering.get_closest_clusters(X_test, kmeans, features_indices_to_cluster_on)

                X_train, X_valid, X_test = clustering.normalizing_per_cluster(X_train, X_valid, X_test, cluster_labels,
                                                                              cluster_labels_valid, cluster_labels_test, folder_saving, reg, lead)
                pretrained_path = primary_folder_saving + season_flag + "/" + "ML_models_2008/nn/FNN_single_task_at_lead_" + str(
                    lead)+".pkl"

                counter = 1
                for lr in lr_list:
                    for hidden_sizes in hidden_sizes_list:
                        for task_specific_hidden_sizes in task_specific_hidden_sizes_list:
                            for weight_decay in weight_decay_list:
                                folder_sub_saving = folder_saving+"hyperparameter_tuning_"+str(counter)+"/"
                                counter = counter+1

                                train_model.train_with_clusters(X_train, y_train, X_valid, y_valid, cluster_labels, cluster_labels_valid, n_clusters, input_size, hidden_sizes, task_specific_hidden_sizes,folder_saving = folder_sub_saving, model_saved = reg + "_for_lead_" + str(
                        lead), n_epochs = epochs, lr = lr, batch_size = bs, weight_decay = weight_decay, lead = lead, pretrained_path = pretrained_path)

                                y_valid_pred = test_and_save_predictions.get_predictions_with_clustering_on_test(
                                    reg+"_for_lead_"+str(lead), X_valid,
                                    y_valid, input_size, hidden_sizes,task_specific_hidden_sizes, n_clusters,cluster_labels_valid ,
                                    folder_sub_saving, pretrained_path)

                                y_pred = test_and_save_predictions.get_predictions_with_clustering_on_test(
                                    reg+"_for_lead_"+str(lead), X_test,
                                    y_test, input_size, hidden_sizes,task_specific_hidden_sizes, n_clusters, cluster_labels_test,
                                    folder_sub_saving, pretrained_path)

                                f = open(folder_sub_saving + 'results.txt','a')

                                f.write("\n" + city + " at Lead " + str(lead) + " and " + season_flag + " Season")
                                f.write("\n"+reg)
                                f.write("\nClustering output and based on: "+str(features_to_cluster_on)+" "+str(c))
                                f.write("\nArchitecture:\n")
                                f.write("hidden size: " + str(hidden_sizes) + ",")
                                f.write("task specific size: " + str(task_specific_hidden_sizes)+",")
                                f.write("epochs: " + str(epochs)+" , batch size: "+ str(bs)+" ,lr: "+str(lr)+" ,weight_decay: "+str(weight_decay))


                                print("\n" + city + " at Lead " + str(lead) + " and " + season_flag + " Season")

                                print("##########VALID##########")
                                rmse_our, mae_our, mb_our, r2_our = postprocess.evaluation_metrics(y_valid, y_valid_pred)
                                print("Performance of our model (rmse, mae, mb, r2): \n\n", round(rmse_our, 1), round(mae_our, 1),
                                      round(mb_our, 1), round(r2_our, 1))
                                f.write('\n evaluation metrics (rmse, mae, mb, r2) on valid data for ' + reg + '=' + str(
                                    round(rmse_our, 1)) + "," + str(round(mae_our, 1)) + "," +
                                        str(round(mb_our, 1)) + "," + str(round(r2_our, 1)) + '\n')

                                print("##########Test##########")
                                rmse_our, mae_our, mb_our, r2_our = postprocess.evaluation_metrics(y_test, y_pred)
                                # print("Performance of our model (rmse, mae, mb, r2): \n\n", round(rmse_our, 1), round(mae_our, 1),
                                #       round(mb_our, 1), round(r2_our, 1))
                                f.write('\n evaluation metrics (rmse, mae, mb, r2) on test data for ' + reg + '=' + str(
                                    round(rmse_our, 1)) + "," + str(round(mae_our, 1)) + "," +
                                        str(round(mb_our, 1)) + "," + str(round(r2_our, 1)) + '\n')
                                f.close()

            else:
                print("not enough data for the season: ", season_flag, "and lead: ", lead)


if __name__=='__main__':
    main()

