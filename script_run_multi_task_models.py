import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from SolarForecasting.ModulesProcessing import collect_data,clean_data
from SolarForecasting.ModulesLearning import preprocessing as preprocess
from SolarForecasting.ModulesLearning import postprocessing as postprocess
from SolarForecasting.ModulesMultiTaskLearning import train_model,test_and_save_predictions
from SolarForecasting.ModulesLearning import model as models



pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# All the variables and hyper-parameters

# city
city = 'Penn_State_PA'

# lead time
lead_times = [4,5,6] #from [1,2,3,4]

# season
seasons =['winter','summer'] #from ['fall', 'winter', 'spring', 'summer', 'year']

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
epochs = 500
lr = 0.0001 #0.0001


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
            X_train, y_train, X_test, y_test, index_clearghi, index_ghi, index_zen = preprocess.get_train_test_data(
                df_train, df_test, final_features, target_feature)
            print("\n\n train and test df shapes ")
            print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

            print(pd.DataFrame(X_train).describe())
            print(pd.DataFrame(X_test).describe())


            ## normalize and shuffle the training dataset
            X_test_before_normalized = X_test.copy()
            X_train, X_test = preprocess.standardize_from_train(X_train, X_test)
            X_train, y_train = preprocess.shuffle(X_train, y_train)

            total_test_samples = len(X_test)
            valid_samples = int(0.5 * total_test_samples)
            print("validation data size: ", valid_samples)
            X_valid = X_test[:valid_samples]
            # X_test = X_test[valid_samples:]
            y_valid = y_test[:valid_samples]
            # y_test = y_test[valid_samples:]

            input_size = X_train.shape[1]
            hidden_size = input_size
            n_hidden = 1
            train_model.train(X_train, y_train, X_valid, y_valid, input_size, hidden_size, n_hidden, n_tasks = len(lead_times),folder_saving = folder_saving+season_flag + "/", model_saved = "hard_parameter_sharing", n_epochs = epochs, lr = lr, batch_size = bs)
            #
            predictions = test_and_save_predictions.get_predictions_on_test("hard_parameter_sharing",X_test,y_test, input_size, hidden_size, n_hidden, len(lead_times), folder_saving+season_flag + "/")

            # model = models.rfGridSearch_model(X_train, y_train)
            # model = models.multi_output_model(X_train,y_train)
            # predictions = model.predict(X_test)


            for n in range(len(lead_times)):

                y_pred = predictions[n]
                # y_pred = predictions[:,n]
                lead = lead_times[n]
                y_test_for_this_lead = y_test[:,n]
                y_true = y_test_for_this_lead
                y_true, y_pred = postprocess.postprocessing_target(y_pred, y_true, X_test_before_normalized, index_ghi,
                                                                   index_clearghi, lead)
                print(y_true[:5],y_test_for_this_lead[:5])
                # normal and smart persistence model
                y_np = postprocess.normal_persistence_model(X_test_before_normalized, index_ghi, lead)
                y_sp = postprocess.smart_persistence_model(X_test_before_normalized, y_test_for_this_lead, index_clearghi, lead)

                #     # selecting the data tuples which are during day for the test-period
                #     index_daytimes = select_daytimes(X_test, index_zen)

                # extracting the daytimes values and removal of outliers
                #     [true_day, pred_day, np_day, sp_day] = select_pred_daytimes_remove_outliers(y_true, y_pred, y_np, y_sp, index_daytimes)
                true_day, pred_day, np_day, sp_day = postprocess.final_true_pred_sp_np(y_true, y_pred, y_np, y_sp, lead,
                                                                                       X_test_before_normalized,
                                                                                       index_zen, index_clearghi)
                # total_test_samples = len(true_day)
                # valid_samples = int(0.5 * total_test_samples)
                # print("validation data size: ", valid_samples)
                true_day_valid = true_day[:valid_samples]
                true_day_test = true_day[valid_samples:]
                pred_day_valid = pred_day[:valid_samples]
                pred_day_test = pred_day[valid_samples:]
                sp_day_valid = sp_day[:valid_samples]
                sp_day_test = sp_day[valid_samples:]
                np_day_valid = np_day[:valid_samples]
                np_day_test = np_day[valid_samples:]

                print("\n" + city + " at Lead " + str(lead) + " and " + season_flag + " Season")

                print("##########VALID##########")
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


                print("#####TEST#################")
                rmse_our, mae_our, mb_our, r2_our = postprocess.evaluation_metrics(true_day_test, pred_day_test)
                print("Performance of our model (rmse, mae, mb, r2): \n\n", round(rmse_our, 1), round(mae_our, 1),
                      round(mb_our, 1), round(r2_our, 1))

                rmse_sp, mae_sp, mb_sp, r2_sp = postprocess.evaluation_metrics(true_day_test, sp_day_test)
                print("Performance of smart persistence model (rmse, mae, mb, r2): \n\n", round(rmse_sp, 1),
                      round(mae_sp, 1),
                      round(mb_sp, 1), round(r2_sp, 1))

                rmse_np, mae_np, mb_np, r2_np = postprocess.evaluation_metrics(true_day_test, np_day_test)
                print("Performance of normal persistence model (rmse, mae, mb, r2): \n\n", round(rmse_np, 1),
                      round(mae_np, 1),
                      round(mb_np, 1), round(r2_np, 1))

                # calculate the skill score of our model over persistence model
                skill_sp = postprocess.skill_score(rmse_our, rmse_sp)
                print("\nSkill of our model over smart persistence: ", round(skill_sp, 1))

                skill_np = postprocess.skill_score(rmse_our, rmse_np)
                print("\nSkill of our model over normal persistence: ", round(skill_np, 1))

                postprocess.plot_results(true_day_test, pred_day_test, sp_day_test, lead, season_flag, folder_plots, model = "multi_task_model")


if __name__=='__main__':
    main()

## with just random regressors to get multiple outputs - I only get the first y as I want and not the rest two
