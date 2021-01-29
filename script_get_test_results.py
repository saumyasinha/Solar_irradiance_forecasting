import numpy as np
import pandas as pd
import os
import pickle
# from sklearn.externals import joblib
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from ngboost import NGBRegressor
from SolarForecasting.ModulesProcessing import collect_data,clean_data
from SolarForecasting.ModulesLearning import preprocessing as preprocess
from SolarForecasting.ModulesLearning import postprocessing as postprocess
from SolarForecasting.ModulesLearning import model as models
from SolarForecasting.ModulesLearning import clustering as clustering
from ngboost.scores import CRPS, MLE


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# All the variables and hyper-parameters

# city
city = 'Penn_State_PA'

# lead time
lead_times = [16,20,24,28,32] #from [1,2,3,4,5,6,7,8,9,10,11,12]

# season
seasons =['year'] #from ['fall', 'winter', 'spring', 'summer', 'year']
res = '15min' #15min

# file locations
# path_project = "C:\\Users\Shivendra\Desktop\SolarProject\solar_forecasting/"
path_project = "/Users/saumya/Desktop/SolarProject/"

path = path_project+"Data/"
folder_saving = path_project + city+"/Models/"
folder_plots = path_project + city+"/Plots/"
clearsky_file_path = path+'clear-sky/'+city+'_15min_original.csv'

# scan all the features (except the flags)
features = ['year','month','day','hour','min','zen','dw_solar','uw_solar','direct_n','diffuse','dw_ir','dw_casetemp','dw_dometemp','uw_ir','uw_casetemp','uw_dometemp','uvb','par','netsolar','netir','totalnet','temp','rh','windspd','winddir','pressure']

# selected features for the study
# final_features = ['year','month','day','hour','MinFlag','zen','dw_solar','dw_ir','temp','rh','windspd','winddir','pressure','clear_ghi']
## exploring more features
final_features = ['year','month','day','hour','MinFlag','zen','dw_solar','uw_solar','direct_n','dw_ir','uw_ir','temp','rh','windspd','winddir','pressure', 'clear_ghi']
# final_features = ['year','month','day','hour','zen','dw_solar','uw_solar','direct_n','dw_ir','uw_ir','temp','rh','windspd','winddir','pressure', 'clear_ghi']

# features_to_cluster_on = ['dw_solar','dw_ir', 'temp','pressure', 'windspd']
# features_to_cluster_on = ['dw_solar','temp']

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
n_timesteps = 3
n_features = 12

# If clustering before prediction
# n_clusters = 3 #6


def get_data():

    ## collect raw data
    years = [2005, 2006, 2007, 2008, 2009]
    object = collect_data.SurfradDataCollector(years, [city], path)

    object.download_data()


    ## cleanse data to get processed version
    for year in years:
        object = clean_data.SurfradDataCleaner(city, year, path)
        object.process(path_to_column_names='ModulesProcessing/column_names.pkl')

# def include_previous_features(X):
#
#     y_list = []
#     previous_time_periods = [1,2]
#     for l in previous_time_periods:
#         print("rolling by: ", l)
#         X_train_shifted = np.roll(X, l)
#         y_list.append(X_train_shifted)
#
#     previous_time_periods_columns = np.column_stack(y_list)
#     X = np.column_stack([X,previous_time_periods_columns])
#     # max_lead = np.max(previous_time_periods)
#     # X = X[max_lead:]
#     print("X shape after adding t-1,t-2 features: ", X.shape)
#     return X
def include_previous_features(X, index_ghi):

    y_list = []
    previous_time_periods = [1,2,3,4,5,6,7,8,9,10,11,12] #[1,2]
    dw_solar = X[:, index_ghi]

    for l in previous_time_periods:
        print("rolling by: ", l)
        dw_solar_rolled = np.roll(dw_solar,l)
        # X_train_shifted = np.roll(X,l)
        # y_list.append(X_train_shifted)
        y_list.append(dw_solar_rolled)


        # print(y_list)
        previous_time_periods_columns = np.column_stack(y_list)
        # print(previous_time_periods_columns[:10])
        X = np.column_stack([X,previous_time_periods_columns])
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
        print("rolling by: ",lead)
        target = np.roll(target, -lead)
        y_list.append(target)

    dataframe_lead['clearness_index'] = np.column_stack(y_list).tolist()
    dataframe_lead['clearness_index'] = dataframe_lead['clearness_index'].apply(tuple)
    # max_lead = np.max(lead_times)
    # dataframe_lead['clearness_index'].values[-max_lead:] = np.nan
    print(dataframe_lead['clearness_index'])

    # remove rows which have any value as NaN
    # dataframe_lead = dataframe_lead.dropna()
    print("*****************")
    print("dataframe with lead size: ", len(dataframe_lead))
    return dataframe_lead



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
    print('dataset size on a 1hour resolution: ',len(dataset))

    # read the clear-sky values
    clearsky = pd.read_csv(clearsky_file_path, skiprows=37, delimiter=';')
    print("The columns of the clear sky file: ", clearsky.columns)

    # divide the observation period in form of year, month, day, hour, min (adding them as variables)
    clearsky[['year', 'month', 'day', 'hour', 'min']] = clearsky['# Observation period'].apply(preprocess.extract_time)
    # clearsky = clearsky.groupby(['year', 'month', 'day', 'hour']).mean()
    clearsky['MinFlag'] = clearsky['min'].apply(preprocess.generateFlag)
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
    # print(df_final.describe())



    for season_flag in seasons:
        ## ML_models_2008 is the folder to save results on testyear 2008
        ## creating different folder for different methods: nn for fully connected networks, rf for random forest etc.

        f = open(folder_saving + season_flag + "/ML_models_2008/probabilistic/"+str(res)+"/ngboost_Without_prevtimesteps//results.txt", 'a')

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

                ## including features from t-1 and t-2 timestamps
                # X_train = include_previous_features(X_train)
                # X_heldout = include_previous_features(X_heldout)


                print("Final train size: ", X_train.shape, y_train.shape)
                print("Final heldout size: ", X_heldout.shape, y_heldout.shape)


                X_test_before_normalized = X_heldout.copy()

                reg = "ngboost_Without_prevtimesteps" ## giving a name to the regression models -- useful when saving results

                ## normalizing the heldout with the X_train used for training
                X_train, X_valid, X_test = preprocess.standardize_from_train(X_train=None, X_valid=None, X_test=X_heldout, folder_saving = folder_saving + season_flag + "/ML_models_2008/probabilistic/"+str(res)+"/ngboost_Without_prevtimesteps/",model = reg, lead = lead)

                # X_train, y_train = preprocess.shuffle(X_train, y_train)

                y_test = np.reshape(y_heldout, -1)

                f.write("\n" + city + " at Lead " + str(lead) + " and " + season_flag + " Season")

                with open(folder_saving + season_flag + "/ML_models_2008/probabilistic/"+str(res)+"/ngboost_Without_prevtimesteps/model_at_lead_" + str(lead) + ".pkl", 'rb') as file:
                    model = pickle.load(file)
                y_true = y_test

                # model_file = open(
                #     folder_saving + season_flag + "/ML_models_2008/rf/modified_features/model_at_lead_" + str(
                #         lead) + ".pkl", 'rb')
                # model = pickle.load(model_file)
                # y_pred = models.fnn_test(X_test, net = None, model_path=folder_saving+ season_flag + "/ML_models_2008/nn/FNN_single_task_at_lead_"+str(lead))
                # y_pred = clustering.cluster_and_predict(X_test, model_dict, cluster_labels_test)
                y_pred = model.predict(X_test)

                y_pred = np.reshape(y_pred, -1)

                print("\n" + city + " at Lead " + str(lead) + " and " + season_flag + " Season")

                # rmse_our, mae_our, mb_our, r2_our = postprocess.evaluation_metrics(y_test, y_pred)
                # print("Performance for clearness index of our model (rmse, mae, mb, r2): \n\n", round(rmse_our, 2), round(mae_our, 2),
                #       round(mb_our, 2), round(r2_our, 2))

                print("#####TEST#################")
                # ## postprocessing on test
                y_true, y_pred = postprocess.postprocessing_target(y_pred, y_true, X_test_before_normalized, index_ghi, index_clearghi, lead)
                # normal and smart persistence model
                y_np = postprocess.normal_persistence_model(X_test_before_normalized, index_ghi, lead)
                y_sp = postprocess.smart_persistence_model(X_test_before_normalized, y_test, index_clearghi, lead)
                true_day_test, pred_day_test, np_day_test, sp_day_test = postprocess.final_true_pred_sp_np(y_true, y_pred, y_np, y_sp, lead, X_test_before_normalized, index_zen, index_clearghi)

                rmse_our, mae_our, mb_our, r2_our = postprocess.evaluation_metrics(true_day_test, pred_day_test)

                print("Performance of our model (rmse, mae, mb, r2): \n\n", round(rmse_our, 2), round(mae_our, 2),
                      round(mb_our, 2), round(r2_our, 2))


                rmse_sp, mae_sp, mb_sp, r2_sp = postprocess.evaluation_metrics(true_day_test, sp_day_test)
                print("Performance of smart persistence model (rmse, mae, mb, r2): \n\n", round(rmse_sp, 2),
                      round(mae_sp, 2),
                      round(mb_sp, 2), round(r2_sp, 2))

                rmse_np, mae_np, mb_np, r2_np = postprocess.evaluation_metrics(true_day_test, np_day_test)
                print("Performance of normal persistence model (rmse, mae, mb, r2): \n\n", round(rmse_np, 1),
                      round(mae_np, 1),
                      round(mb_np, 1), round(r2_np, 1))

                # calculate the skill score of our model over persistence model
                skill_sp = postprocess.skill_score(rmse_our, rmse_sp)

                print("\nSkill of our model over smart persistence: ", round(skill_sp, 2))

                skill_np = postprocess.skill_score(rmse_our, rmse_np)
                print("\nSkill of our model over normal persistence: ", round(skill_np, 2))


                f.write('score on heldout data for year 2008' + reg + '=' + str(round(skill_sp, 2)) + '\n')
                # # postprocess.plot_results(true_day_test, pred_day_test, sp_day_test, lead, season_flag, folder_plots,
                # #                          model="random_forest_model")


            else:
                print("not enough data for the season: ", season_flag, "and lead: ", lead)

        f.close()


if __name__=='__main__':
    main()


