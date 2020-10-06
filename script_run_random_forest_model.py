import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SolarForecasting.ModulesLearning import preprocessing as preprocess
from SolarForecasting.ModulesLearning import postprocessing as postprocess
from SolarForecasting.ModulesLearning import model as models

# All the variables and hyper-parameters
path = "/Users/saumya/Desktop/SolarProject/Data/"

# city
city = 'Penn_State_PA'

# lead time
lead_times = [2] #from [1,2,3,4]

# season
seasons =['fall'] #from ['fall', 'winter', 'spring', 'summer', 'year']

# file locations
processed_file_path = path+'processed/'+city
clearsky_file_path = path+'clear-sky/'+city+'_15mins_original.csv'

# scan all the features (except the flags)
features = ['year','month','day','hour','min','zen','dw_solar','uw_solar','direct_n','diffuse','dw_ir','dw_casetemp','dw_dometemp','uw_ir','uw_casetemp','uw_dometemp','uvb','par','netsolar','netir','totalnet','temp','rh','windspd','winddir','pressure']

# selected features for the study
final_features = ['year','month','day','hour','MinFlag','zen','dw_solar','dw_ir','temp','rh','windspd','winddir','pressure','clear_ghi']
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
lr = 0.0001


def main():

    ## pre-processing steps

    # extract the input data files (SURFAD data)
    combined_csv = preprocess.extract_frame(processed_file_path)
    print("The columns of the initial data file: ", combined_csv.columns)

    # extract the features from the input
    dataset = combined_csv[features]
    print(len(dataset))

    # 15 mins resolution
    dataset['MinFlag'] = dataset['min'].apply(preprocess.generateFlag)
    dataset = dataset.groupby(['year', 'month', 'day', 'hour', 'MinFlag']).mean()
    dataset.reset_index(inplace=True)
    print(len(dataset))

    # read the clear-sky values
    clearsky = pd.read_csv(clearsky_file_path, skiprows=37, delimiter=';')
    print("The columns of the clear sky file: ", clearsky.columns)

    # divide the observation period in form of year, month, day, hour, min (adding them as vaiablesr)
    clearsky[['year', 'month', 'day', 'hour', 'min']] = clearsky['# Observation period'].apply(preprocess.extract_time)
    clearsky['MinFlag'] = clearsky['min'].apply(preprocess.generateFlag)

    # merging the clear sky values with SURFAD input dataset
    df = dataset.merge(clearsky, on=['year', 'month', 'day', 'hour', 'MinFlag'], how='inner')
    # print(df.tail)

    # renaming the clear sky GHI
    df = df.rename(columns={'Clear sky GHI': 'clear_ghi'})

    # selecting only the required columns
    df = df[final_features]


    # adjust the boundary values
    df = preprocess.adjust_boundary_values(df)
    print(df.tail)
    # df: 175073 rows x 14 columns
    # half of them are outliers??!!
    # 87569
    # 85792

    # adding the clearness index as a feature
    df['clearness_index'] = df['dw_solar'] / df['clear_ghi']
    print("clearness_index: ", df['clearness_index'] )

    # get dataset for the study period
    df = preprocess.extract_study_period(df,startmonth, startyear, endmonth, endyear)
    print("\n\n after extracting study period")
    df.reset_index(drop=True, inplace=True)
    print(df.tail)
    # 140077 rows x 15 columns

    # adjust the outliers
    df = preprocess.adjust_outlier_clearness_index(df)
    print("\n\n after adjusting outliers of clearness index")
    print(df.tail)

    for season_flag in seasons:
        for lead in lead_times:
            # create dataset with lead
            df_lead = preprocess.create_lead_dataset(df, lead, final_features, target_feature)
            # (no drop with dropna)
            #     print("\n\n after create_lead_dataset")
            #     print(len(df_lead))

            # get the seasonal data you want
            df, test_startdate, test_enddate = preprocess.get_yearly_or_season_data(df_lead, season_flag)
            print("\n\n after getting seasonal data (test_startdate; test_enddate)", test_startdate, test_enddate)
            print(df.tail)

            # dividing into training and test set
            df_train, df_test = preprocess.train_test_spilt(df, season_flag, testyear)
            print("\n\n after dividing_training_test")
            print("train_set\n",len(df_train))
            print("test_set\n",len(df_test))

            # extract the X_train, y_train, X_test, y_test for training
            X_train_all, y_train_all, X_test_all, y_test_all, index_clearghi, index_ghi, index_zen = preprocess.get_train_test_data(
                df_train, df_test, lead, final_features, target_feature)
            print("\n\n train and test df shapes ")
            print(X_train_all.shape, y_train_all.shape, X_test_all.shape, y_test_all.shape)

            # get the day times
            index_daytimes_train = preprocess.select_daytimes(X_train_all, index_zen)
            index_daytimes_test = preprocess.select_daytimes(X_test_all, index_zen)

            # filter the training and test set to have only day values
            X_train, y_train, X_test, y_test = preprocess.filter_dayvalues(X_train_all, y_train_all, X_test_all, y_test_all,
                                                                                  index_daytimes_train,
                                                                                  index_daytimes_test)
            print("\n\n after filter_dayvalues_remove_outliers")
            print("Final size: ", X_train.shape, y_train.shape, X_test.shape, y_test.shape)

            y_train  = np.reshape(y_train, -1)
            y_test = np.reshape(y_test, -1)

    #         #     # call the gridSearch with finerparameter
            model = models.rf_model(X_train, y_train)

            y_true = y_test
            y_pred = model.predict(X_test)

            # plotting a few analysis results
            plt.figure(figsize=(20,10))
            plt.hist(y_train, density=True, bins=1000)
            plt.xlabel("train set")
            plt.show()

            plt.figure(figsize=(20,10))
            plt.hist(y_test, density=True, bins=1000)
            plt.xlabel("true test set")
            plt.show()

            plt.figure(figsize=(20,10))
            plt.hist(y_pred, density=True, bins=1000)
            plt.xlabel("pred test set")
            plt.show()
    #
            [y_true, y_pred] = postprocess.postprocessing_target(y_pred, y_true, X_test, index_ghi, index_clearghi, lead)
            #     print("after X, y_true, y_pred: ",X_test[:2], y_true[:2], y_pred[:2])

            # normal and smart persistence model
            y_np = postprocess.normal_persistence_model(X_test, index_ghi, lead)
            y_sp = postprocess.smart_persistence_model(X_test, y_test, index_clearghi, lead)

            #     # selecting the data tuples which are during day for the test-period
            #     index_daytimes = select_daytimes(X_test, index_zen)

            # extracting the daytimes values and removal of outliers
            #     [true_day, pred_day, np_day, sp_day] = select_pred_daytimes_remove_outliers(y_true, y_pred, y_np, y_sp, index_daytimes)
            true_day, pred_day1, np_day, sp_day = postprocess.check_and_remove_outliers(y_true, y_pred, y_np, y_sp)
            true_day_valid = true_day[:500]
            true_day_test = true_day[500:]
            pred_day_valid = pred_day1[:500]
            pred_day_test = pred_day1[500:]
            sp_day_valid = sp_day[:500]
            sp_day_test = sp_day[500:]
            np_day_test = np_day[500:]

            #     print(true_day.mean(), pred_day1.mean(), sp_day.mean(), pred_day.mean())

            #     plot_results(true_day_test, pred_day_test, sp_day_test)
            print("\n\n\n" + city + " at Lead " + str(lead) + " and " + season_flag + " Season")
            # calculate the error measures................................................
            rmse_our, mae_our, mb_our, std_our = postprocess.evaluation_metrics(true_day_test, pred_day_test)
            print("Performance of our model (rmse, mae, mb, std): \n\n", round(rmse_our, 1), round(mae_our, 1),
                  round(mb_our, 1), round(std_our, 1))

            rmse_sp, mae_sp, mb_sp, std_sp = postprocess.evaluation_metrics(true_day_test, sp_day_test)
            print("Performance of smart persistence model (rmse, mae, mb): \n\n", round(rmse_sp, 1), round(mae_sp, 1),
                  round(mb_sp, 1), round(std_sp, 1))

            rmse_np, mae_np, mb_np, std_np = postprocess.evaluation_metrics(true_day_test, np_day_test)
            print("Performance of normal persistence model (rmse, mae, mb): \n\n", round(rmse_np, 1), round(mae_np, 1),
                  round(mb_np, 1), round(std_np, 1))

            # calculate the skill score of our model over persistence model
            skill_sp = postprocess.skill_score(rmse_our, rmse_sp)
            print("\nSkill of our model over smart persistence: ", round(skill_sp, 1))

            skill_np = postprocess.skill_score(rmse_our, rmse_np)
            print("\nSkill of our model over normal persistence: ", round(skill_np, 1))


if __name__=='__main__':
    main()


# see if sorting according to dates is essential in the very beginning
# check if the lead is correctly created
