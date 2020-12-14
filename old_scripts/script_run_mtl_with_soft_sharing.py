import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from SolarForecasting.ModulesProcessing import collect_data,clean_data
from SolarForecasting.ModulesLearning import preprocessing as preprocess
from SolarForecasting.ModulesLearning import postprocessing as postprocess
from SolarForecasting.ModulesMultiTaskLearning import train_model,test_and_save_predictions
from SolarForecasting.ModulesLearning import model as models
from SolarForecasting.ModulesLearning import clustering as clustering



pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# All the variables and hyper-parameters

# city
city = 'Penn_State_PA'

# lead time
lead_times = [3,4,5] #from [1,2,3,4]

# season
seasons =['summer'] #from ['fall', 'winter', 'spring', 'summer', 'year']

# file locations
path_project = "/Users/saumya/Desktop/SolarProject/"
path = path_project+"Data/"
folder_saving = path_project + "Models/"
folder_plots = path_project + "Plots/"
clearsky_file_path = path+'clear-sky/'+city+'_15mins_original.csv'

# scan all the features (except the flags)
features = ['year','month','day','hour','min','zen','dw_solar','uw_solar','direct_n','diffuse','dw_ir','dw_casetemp','dw_dometemp','uw_ir','uw_casetemp','uw_dometemp','uvb','par','netsolar','netir','totalnet','temp','rh','windspd','winddir','pressure']

# selected features for the study
# final_features = ['year','month','day','hour','MinFlag','zen','dw_solar','dw_ir','temp','rh','windspd','winddir','pressure','clear_ghi']
## explore more features, but a lot of them are categorical so might want to remove those
final_features = ['year','month','day','hour','MinFlag','zen','dw_solar','uw_solar','direct_n','dw_ir','uw_ir','temp','rh','windspd','winddir','pressure', 'clear_ghi']

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
lr = 0.0001 #0.0001

reg = "soft_sharing"


soft_loss_weights = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] #not needed for hard sharing



def get_data():

    ## collect raw data
    years = [2005, 2006, 2007, 2008, 2009]
    object = collect_data.SurfradDataCollector(years, [city], path)

    object.download_data()


    ## cleanse data to get processed version
    for year in years:
        object = clean_data.SurfradDataCleaner(city, year, path)
        object.process(path_to_column_names='ModulesProcessing/column_names.pkl')

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
    max_lead = np.max(lead_times)
    dataframe_lead['clearness_index'].values[-max_lead:] = np.nan
    print(dataframe_lead['clearness_index'])

    # remove rows which have any value as NaN
    dataframe_lead = dataframe_lead.dropna()
    print("*****************")
    print("dataframe with lead size: ", len(dataframe_lead))
    return dataframe_lead

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


    ## convert negatives to 0 for all features
    df[df<0] = 0
    print("stats of selected features/columns after converting all negatives to 0")
    print(df.describe())




    # adding the clearness index and dropping rows with 0 clear_ghi and taking only daytimes
    df = df[df['clear_ghi'] > 0]
    df = df[df['zen']<85]
    df['clearness_index'] = df['dw_solar'] / df['clear_ghi']
    df.reset_index(drop=True, inplace=True)
    print("after removing data points with 0 clear_ghi and selecting daytimes",len(df))

    # create dataset with lead
    df_lead = create_mulitple_lead_dataset(df, final_features, target_feature)


    for season_flag in seasons:

        f = open(folder_saving + season_flag + '/'+reg+'/results.txt', 'a')

        # get the seasonal data you want
        df, test_startdate, test_enddate = preprocess.get_yearly_or_season_data(df_lead, season_flag)
        print("\n\n after getting seasonal data (test_startdate; test_enddate)", test_startdate, test_enddate)
        print(df.tail)



        # dividing into training and test set
        df_train, df_test = preprocess.train_test_spilt(df, season_flag, testyear)
        print("\n\n after dividing_training_test")
        print("train_set\n",len(df_train))
        print("test_set\n",len(df_test))

        if len(df_train)>0 and len(df_test)>0:
            # extract the X_train, y_train, X_test, y_test for training
            X_train, y_train, X_test, y_test, index_clearghi, index_ghi, index_zen, col_to_indices_mapping = preprocess.get_train_test_data(
                df_train, df_test, final_features, target_feature)
            print("\n\n train and test df shapes ")
            print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

            # X_train = include_previous_features(X_train)
            # X_test = include_previous_features(X_test)

            print(pd.DataFrame(X_train).describe())
            print(pd.DataFrame(X_test).describe())

            ## Divide train into train+valid
            training_samples = int(0.8 * len(X_train))
            print(training_samples)
            X_valid = X_train[training_samples:]
            X_train = X_train[:training_samples]
            y_valid = y_train[training_samples:]
            y_train = y_train[:training_samples]

            ## normalize and shuffle the training dataset
            X_test_before_normalized = X_test.copy()
            X_valid_before_normalized = X_valid.copy()
            X_train, X_valid, X_test = preprocess.standardize_from_train(X_train, X_valid, X_test)
            X_train, y_train = preprocess.shuffle(X_train, y_train)

            print(X_train.shape, X_valid.shape, X_test.shape)
            input_size = X_train.shape[1]
            hidden_size = 24#16 for hard, 24 for soft
            n_hidden = 2

            for soft_loss_weight in soft_loss_weights:
                train_model.train(X_train, y_train, X_valid, y_valid, input_size, hidden_size, n_hidden, n_tasks = len(lead_times),folder_saving = folder_saving+season_flag + "/"+ reg + "/"+str(soft_loss_weight)+"/", model_saved = reg+"_sharing", n_epochs = epochs, lr = lr, batch_size = bs, soft_loss_weight= soft_loss_weight)
                #
                predictions_valid = test_and_save_predictions.get_predictions_on_test(reg+"/"+str(soft_loss_weight)+"/"+reg+"_sharing", X_valid,
                                                                                 y_valid, input_size, hidden_size,
                                                                                 n_hidden, len(lead_times),
                                                                                 folder_saving + season_flag + "/",soft_loss_weight)




                predictions_test = test_and_save_predictions.get_predictions_on_test(reg+"/"+str(soft_loss_weight)+"/"+reg+"_sharing",X_test,y_test, input_size, hidden_size, n_hidden, len(lead_times), folder_saving+season_flag + "/", soft_loss_weight)

                # model = models.rfGridSearch_model(X_train, y_train)
                # model = models.multi_output_model(X_train,y_train)
                # predictions = model.predict(X_test)
                f.write("\n" + " With soft loss weight as " + str(soft_loss_weight) +"\n")
                print("With soft loss weight as ",soft_loss_weight)

                for n in range(len(lead_times)):
                    lead = lead_times[n]

                    y_pred = predictions_test[n]
                    y_test_for_this_lead = y_test[:,n]
                    y_true = y_test_for_this_lead

                    y_valid_pred = predictions_valid[n]
                    y_valid_for_this_lead = y_valid[:, n]
                    y_valid_true = y_valid_for_this_lead

                    print("\n" + city + " at Lead " + str(lead) + " and " + season_flag + " Season")
                    f.write("\n" + city + " at Lead " + str(lead) + " and " + season_flag + " Season"+"\n")
                    print("##########VALID##########")
                    ## postprocessing for valid
                    y_valid_true, y_valid_pred = postprocess.postprocessing_target(y_valid_pred, y_valid_true,
                                                                                   X_valid_before_normalized, index_ghi,
                                                                                   index_clearghi, lead)

                    y_np = postprocess.normal_persistence_model(X_valid_before_normalized, index_ghi, lead)
                    y_sp = postprocess.smart_persistence_model(X_valid_before_normalized, y_valid_for_this_lead, index_clearghi, lead)
                    true_day_valid, pred_day_valid, np_day_valid, sp_day_valid = postprocess.final_true_pred_sp_np(
                        y_valid_true,
                        y_valid_pred, y_np,
                        y_sp, lead,
                        X_valid_before_normalized,
                        index_zen,
                        index_clearghi)

                    # calculate the error measures................................................
                    rmse_our, mae_our, mb_our, r2_our = postprocess.evaluation_metrics(true_day_valid, pred_day_valid)
                    print("Performance of our model (rmse, mae, mb, r2): \n\n", round(rmse_our, 1), round(mae_our, 1),
                          round(mb_our, 1), round(r2_our, 1))

                    rmse_sp, mae_sp, mb_sp, r2_sp = postprocess.evaluation_metrics(true_day_valid, sp_day_valid)
                    print("Performance of smart persistence model (rmse, mae, mb, r2): \n\n", round(rmse_sp, 1),
                          round(mae_sp, 1),
                          round(mb_sp, 1), round(r2_sp, 1))

                    rmse_np, mae_np, mb_np, r2_np = postprocess.evaluation_metrics(true_day_valid, np_day_valid)
                    print("Performance of normal persistence model (rmse, mae, mb, r2): \n\n", round(rmse_np, 1),
                          round(mae_np, 1),
                          round(mb_np, 1), round(r2_np, 1))

                    # calculate the skill score of our model over persistence model
                    skill_sp = postprocess.skill_score(rmse_our, rmse_sp)
                    print("\nSkill of our model over smart persistence: ", round(skill_sp, 1))

                    skill_np = postprocess.skill_score(rmse_our, rmse_np)
                    print("\nSkill of our model over normal persistence: ", round(skill_np, 1))

                    f.write('score on valid data for ' + reg + '=' + str(round(skill_sp, 1)) + '\n')

                    print("#####TEST#################")
                    ## postprocessing on test
                    y_true, y_pred = postprocess.postprocessing_target(y_pred, y_true, X_test_before_normalized, index_ghi,
                                                                       index_clearghi, lead)
                    # normal and smart persistence model
                    y_np = postprocess.normal_persistence_model(X_test_before_normalized, index_ghi, lead)
                    y_sp = postprocess.smart_persistence_model(X_test_before_normalized, y_test_for_this_lead, index_clearghi, lead)
                    true_day_test, pred_day_test, np_day_test, sp_day_test = postprocess.final_true_pred_sp_np(y_true,
                                                                                                               y_pred, y_np,
                                                                                                               y_sp, lead,
                                                                                                               X_test_before_normalized,
                                                                                                               index_zen,
                                                                                                               index_clearghi)


                    rmse_our, mae_our, mb_our, r2_our = postprocess.evaluation_metrics(true_day_test, pred_day_test)
                    # print("Performance of our model (rmse, mae, mb, r2): \n\n", round(rmse_our, 1), round(mae_our, 1),
                    #       round(mb_our, 1), round(r2_our, 1))

                    rmse_sp, mae_sp, mb_sp, r2_sp = postprocess.evaluation_metrics(true_day_test, sp_day_test)
                    # print("Performance of smart persistence model (rmse, mae, mb, r2): \n\n", round(rmse_sp, 1),
                    #       round(mae_sp, 1),
                    #       round(mb_sp, 1), round(r2_sp, 1))

                    rmse_np, mae_np, mb_np, r2_np = postprocess.evaluation_metrics(true_day_test, np_day_test)
                    # print("Performance of normal persistence model (rmse, mae, mb, r2): \n\n", round(rmse_np, 1),
                    #       round(mae_np, 1),
                    #       round(mb_np, 1), round(r2_np, 1))

                    # calculate the skill score of our model over persistence model
                    skill_sp = postprocess.skill_score(rmse_our, rmse_sp)
                    # print("\nSkill of our model over smart persistence: ", round(skill_sp, 1))

                    skill_np = postprocess.skill_score(rmse_our, rmse_np)
                    # print("\nSkill of our model over normal persistence: ", round(skill_np, 1))

                    f.write('score on test data for ' + reg + '=' + str(round(skill_sp, 1)) + '\n')

                    # postprocess.plot_results(true_day_test, pred_day_test, sp_day_test, lead, season_flag, folder_plots, model = "hard_parameter_sharing_model")


if __name__=='__main__':
    main()

# Hard-sharing: 2 hidden layers of size 16 and individual 1 of size 8 (including regularization)
# Penn_State_PA at Lead 1 and summer Season
# ##########VALID##########
# 0
# (2938,) (2938,) (2938,) (2938,)
# Performance of our model (rmse, mae, mb, r2):
#
#  105.2 72.4 13.2 0.9
# Performance of smart persistence model (rmse, mae, mb, r2):
#
#  114.4 65.8 0.3 0.9
# Performance of normal persistence model (rmse, mae, mb, r2):
#
#  119.4 78.6 -0.0 0.8
#
# Skill of our model over smart persistence:  8.0
#
# Skill of our model over normal persistence:  11.9
# #####TEST#################
# 0
# (4886,) (4886,) (4886,) (4886,)
# Performance of our model (rmse, mae, mb, r2):
#
#  126.2 88.7 25.6 0.8
# Performance of smart persistence model (rmse, mae, mb, r2):
#
#  119.5 69.7 0.6 0.8
# Performance of normal persistence model (rmse, mae, mb, r2):
#
#  123.0 79.2 0.0 0.8
#
# Skill of our model over smart persistence:  -5.6
#
# Skill of our model over normal persistence:  -2.6
#
# Penn_State_PA at Lead 2 and summer Season
# ##########VALID##########
# 0
# (2936,) (2936,) (2936,) (2936,)
# Performance of our model (rmse, mae, mb, r2):
#
#  127.6 87.2 12.9 0.8
# Performance of smart persistence model (rmse, mae, mb, r2):
#
#  139.8 82.7 0.5 0.8
# Performance of normal persistence model (rmse, mae, mb, r2):
#
#  164.8 117.6 0.1 0.7
#
# Skill of our model over smart persistence:  8.7
#
# Skill of our model over normal persistence:  22.6
# #####TEST#################
# 0
# (4884,) (4884,) (4884,) (4884,)
# Performance of our model (rmse, mae, mb, r2):
#
#  148.8 105.4 22.4 0.8
# Performance of smart persistence model (rmse, mae, mb, r2):
#
#  147.6 89.3 1.0 0.8
# Performance of normal persistence model (rmse, mae, mb, r2):
#
#  171.3 119.8 0.6 0.7
#
# Skill of our model over smart persistence:  -0.8
#
# Skill of our model over normal persistence:  13.1
#
# Penn_State_PA at Lead 3 and summer Season
# ##########VALID##########
# 0
# (2934,) (2934,) (2934,) (2934,)
# Performance of our model (rmse, mae, mb, r2):
#
#  143.8 101.2 16.9 0.8
# Performance of smart persistence model (rmse, mae, mb, r2):
#
#  153.4 92.4 -0.2 0.7
# Performance of normal persistence model (rmse, mae, mb, r2):
#
#  203.2 152.4 -0.4 0.5
#
# Skill of our model over smart persistence:  6.3
#
# Skill of our model over normal persistence:  29.2
# #####TEST#################
# 0
# (4882,) (4882,) (4882,) (4882,)
# Performance of our model (rmse, mae, mb, r2):
#
#  161.9 118.9 21.6 0.7
# Performance of smart persistence model (rmse, mae, mb, r2):
#
#  163.3 101.3 1.0 0.7
# Performance of normal persistence model (rmse, mae, mb, r2):
#
#  204.3 152.4 1.1 0.5
#
# Skill of our model over smart persistence:  0.9
#
# Skill of our model over normal persistence:  20.8
#
# Penn_State_PA at Lead 4 and summer Season
# ##########VALID##########
# 0
# (2932,) (2932,) (2932,) (2932,)
# Performance of our model (rmse, mae, mb, r2):
#
#  156.2 112.4 19.1 0.7
# Performance of smart persistence model (rmse, mae, mb, r2):
#
#  168.9 104.0 -2.2 0.7
# Performance of normal persistence model (rmse, mae, mb, r2):
#
#  238.9 185.4 -2.4 0.4
#
# Skill of our model over smart persistence:  7.5
#
# Skill of our model over normal persistence:  34.6
# #####TEST#################
# 0
# (4880,) (4880,) (4880,) (4880,)
# Performance of our model (rmse, mae, mb, r2):
#
#  171.4 127.5 20.7 0.7
# Performance of smart persistence model (rmse, mae, mb, r2):
#
#  178.3 113.1 -0.4 0.6
# Performance of normal persistence model (rmse, mae, mb, r2):
#
#  233.2 180.2 0.5 0.4
#
# Skill of our model over smart persistence:  3.9
#
# Skill of our model over normal persistence:  26.5
#
# Penn_State_PA at Lead 5 and summer Season
# ##########VALID##########
# 0
# (2930,) (2930,) (2930,) (2930,)
# Performance of our model (rmse, mae, mb, r2):
#
#  166.2 123.1 22.5 0.7
# Performance of smart persistence model (rmse, mae, mb, r2):
#
#  181.1 112.4 -5.2 0.6
# Performance of normal persistence model (rmse, mae, mb, r2):
#
#  265.2 212.2 -6.8 0.2
#
# Skill of our model over smart persistence:  8.2
#
# Skill of our model over normal persistence:  37.3
# #####TEST#################
# 0
# (4878,) (4878,) (4878,) (4878,)
# Performance of our model (rmse, mae, mb, r2):
#
#  178.6 135.7 14.3 0.6
# Performance of smart persistence model (rmse, mae, mb, r2):
#
#  188.5 121.0 -3.2 0.6
# Performance of normal persistence model (rmse, mae, mb, r2):
#
#  255.5 202.9 -2.4 0.2
#
# Skill of our model over smart persistence:  5.2
#
# Skill of our model over normal persistence:  30.1
#
# Penn_State_PA at Lead 6 and summer Season
# ##########VALID##########
# 0
# (2928,) (2928,) (2928,) (2928,)
# Performance of our model (rmse, mae, mb, r2):
#
#  174.3 129.9 21.2 0.6
# Performance of smart persistence model (rmse, mae, mb, r2):
#
#  196.8 122.5 -7.3 0.5
# Performance of normal persistence model (rmse, mae, mb, r2):
#
#  291.6 239.9 -12.9 0.0
#
# Skill of our model over smart persistence:  11.4
#
# Skill of our model over normal persistence:  40.2
# #####TEST#################
# 0
# (4876,) (4876,) (4876,) (4876,)
# Performance of our model (rmse, mae, mb, r2):
#
#  188.3 144.6 7.4 0.6
# Performance of smart persistence model (rmse, mae, mb, r2):
#
#  199.4 129.9 -6.3 0.5
# Performance of normal persistence model (rmse, mae, mb, r2):
#
#  286.1 228.3 -8.0 0.0
#
# Skill of our model over smart persistence:  5.5
#
# Skill of our model over normal persistence:  34.2


## I modified dropout rate from 0.1 to 0.5, introduced early stopping, did nto inlcude pevious features, change network structure (not any longer)
# , validation size(not any longer) (didn't seem appropriate -  used different learning rates for different models)
#changed from average to sum of losses