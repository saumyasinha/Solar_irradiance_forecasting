import numpy as np
import pandas as pd
import os
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from SolarForecasting.ModulesProcessing import collect_data,clean_data
from SolarForecasting.ModulesLearning import preprocessing as preprocess
from SolarForecasting.ModulesLearning import postprocessing as postprocess
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
seasons =['winter', 'summer'] #from ['fall', 'winter', 'spring', 'summer', 'year']

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
## exploring more features
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
epochs = 100
lr = 0.0001


def get_data():

    ## collect raw data
    years = [2005, 2006, 2007, 2008, 2009]
    object = collect_data.SurfradDataCollector(years, [city], path)

    object.download_data()


    ## cleanse data to get processed version
    for year in years:
        object = clean_data.SurfradDataCleaner(city, year, path)
        object.process(path_to_column_names='ModulesProcessing/column_names.pkl')



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

    # df.hist(column=['clearness_index'], bins = 50)
    # print(df['clearness_index'].value_counts())
    # print(df['clearness_index'].max(), df['clearness_index'].min())
    # plt.savefig("histogram plots for clearness index")

    # adjust the outliers for clearness index (no need to do this anymore -- since I am already taking care of 0 clear_ghi)
    # df = preprocess.adjust_outlier_clearness_index(df)
    # print("\n\n after adjusting outliers of clearness index")
    # print(df.tail)


    for season_flag in seasons:
        for lead in lead_times:
            # create dataset with lead
            df_lead = preprocess.create_lead_dataset(df_final, lead, final_features, target_feature)

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
                # extract the X_train, y_train, X_test, y_test
                X_train, y_train, X_test, y_test, index_clearghi, index_ghi, index_zen = preprocess.get_train_test_data(
                    df_train, df_test, final_features, target_feature)
                print("\n\n train and test df shapes ")
                print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

                # filter the training to have only day values
                # X_train, y_train = preprocess.filter_dayvalues_and_zero_clearghi(X_train_all, y_train_all,
                #                                                                       index_zen, index_clearghi)

                print("\n\n after filtering dayvalues")
                print("Final size: ", X_train.shape, y_train.shape)
                print(pd.DataFrame(X_train).describe())
                print(pd.DataFrame(y_train).describe())
                print(pd.DataFrame(X_test).describe())
                print(pd.DataFrame(y_test).describe())


                ## normalize data and shuffle the training dataset
                X_test_before_normalized = X_test.copy()
                X_train, X_test = preprocess.standardize_from_train(X_train, X_test)
                X_train, y_train = preprocess.shuffle(X_train, y_train)

                y_train  = np.reshape(y_train, -1)
                y_test = np.reshape(y_test, -1)

                # call the gridSearch and saving model
                # model = models.rfGridSearch_model(X_train, y_train)
                # for name, importance in zip(final_features[5:], model.best_estimator_.feature_importances_):
                #     print(name, "=", importance)
                # # model = models.lr_model(X_train, y_train)
                #
                # joblib.dump(model.best_estimator_, folder_saving+season_flag + "/rf_model_for_lead_"+str(lead))
                model = joblib.load(folder_saving+season_flag + "/rf_model_for_lead_"+str(lead))
                y_true = y_test
                y_pred = model.predict(X_test)

                print(np.sum(y_true), np.sum(y_pred))
        #
                y_true, y_pred = postprocess.postprocessing_target(y_pred, y_true, X_test_before_normalized, index_ghi, index_clearghi, lead)


                # normal and smart persistence model
                y_np = postprocess.normal_persistence_model(X_test_before_normalized, index_ghi, lead)
                y_sp = postprocess.smart_persistence_model(X_test_before_normalized, y_test, index_clearghi, lead)

                #     # selecting the data tuples which are during day for the test-period
                #     index_daytimes = select_daytimes(X_test, index_zen)

                # extracting the daytimes values and removal of outliers
                #     [true_day, pred_day, np_day, sp_day] = select_pred_daytimes_remove_outliers(y_true, y_pred, y_np, y_sp, index_daytimes)
                true_day, pred_day, np_day, sp_day = postprocess.final_true_pred_sp_np(y_true, y_pred, y_np, y_sp, lead, X_test_before_normalized, index_zen, index_clearghi)
                total_test_samples = len(true_day)
                valid_samples = int(0.5*total_test_samples)
                print("validation data size: ",valid_samples)
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

                postprocess.plot_results(true_day_test, pred_day_test, sp_day_test, lead, season_flag, folder_plots,
                                         model="random_forest_model")


            else:
                print("not enough data for the season: ", season_flag, "and lead: ", lead)


if __name__=='__main__':
    main()



## Winter
# Penn_State_PA at Lead 4 and winter Season
# Performance of our model (rmse, mae, mb, r2):
#
#  66.0 47.0 -2.5 0.7
# Performance of smart persistence model (rmse, mae, mb, r2):
#
#  72.7 48.0 1.7 0.7
# Performance of normal persistence model (rmse, mae, mb, r2):
#
#  94.7 70.4 -0.1 0.5
#
# Skill of our model over smart persistence:  9.3
#
# Skill of our model over normal persistence:  30.4

#Test
#
# Skill of our model over smart persistence:  7.9
#
# Skill of our model over normal persistence:  35.8
# #
#
# Penn_State_PA at Lead 5 and winter Season
# Performance of our model (rmse, mae, mb, r2):
#
#  69.1 50.1 -3.1 0.7
# Performance of smart persistence model (rmse, mae, mb, r2):
#
#  78.0 51.6 2.0 0.6
# Performance of normal persistence model (rmse, mae, mb, r2):
#
#  107.0 80.2 -0.1 0.3
#
# Skill of our model over smart persistence:  11.5
#
# Skill of our model over normal persistence:  35.4
#
# Test
# Skill of our model over smart persistence:  8.9
#
# Skill of our model over normal persistence:  41.7

#
# Penn_State_PA at Lead 6 and winter Season
# Performance of our model (rmse, mae, mb, r2):
#
#  72.9 54.0 -3.7 0.7
# Performance of smart persistence model (rmse, mae, mb, r2):
#
#  81.5 54.8 2.4 0.6
# Performance of normal persistence model (rmse, mae, mb, r2):
#
#  117.6 89.5 -0.2 0.2
#
# Skill of our model over smart persistence:  10.5
#
# Skill of our model over normal persistence:  38.0
#Test
# Skill of our model over smart persistence:  8.9
#
# Skill of our model over normal persistence:  45.6



## Summer
# Penn_State_PA at Lead 4 and summer Season
# Performance of our model (rmse, mae, mb, r2):
#
#  148.9 106.3 -1.0 0.8
# Performance of smart persistence model (rmse, mae, mb, r2):
#
#  174.0 108.7 0.7 0.7
# Performance of normal persistence model (rmse, mae, mb, r2):
#
#  203.9 153.6 -0.2 0.6
#
# Skill of our model over smart persistence:  14.4
#
# Skill of our model over normal persistence:  27.0
#
#Test:
# Skill of our model over smart persistence:  16.8
#
# Skill of our model over normal persistence:  31.1
#


# Penn_State_PA at Lead 5 and summer Season
# Performance of our model (rmse, mae, mb, r2):
#
#  152.7 111.0 -2.1 0.8
# Performance of smart persistence model (rmse, mae, mb, r2):
#
#  177.4 111.6 1.0 0.7
# Performance of normal persistence model (rmse, mae, mb, r2):
#
#  220.0 169.8 -0.4 0.5
#
# Skill of our model over smart persistence:  13.9
#
# Skill of our model over normal persistence:  30.6
#
#Test:
# Skill of our model over smart persistence:  16.0
#
# Skill of our model over normal persistence:  34.4



# Penn_State_PA at Lead 6 and summer Season
# Performance of our model (rmse, mae, mb, r2):
#
#  161.6 119.4 -3.4 0.7
# Performance of smart persistence model (rmse, mae, mb, r2):
#
#  184.9 118.0 1.4 0.6
# Performance of normal persistence model (rmse, mae, mb, r2):
#
#  240.2 189.5 -0.7 0.4
#
# Skill of our model over smart persistence:  12.6
#
# Skill of our model over normal persistence:  32.7

# #Test:
#
# Skill of our model over smart persistence:  16.3
#
# Skill of our model over normal persistence:  38.2