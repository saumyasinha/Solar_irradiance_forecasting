import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import pickle


class SurfradDataCleaner:
    '''
    This class cleans the downloaded .dat files located in raw data folder for a
    particular city and year and stores them as csv files.
    '''
    def __init__(self, city, year, path_to_data, skip_header = 2):
        self.city = city
        self.year = str(year)
        self.raw_dirname = None
        self.path_to_data = path_to_data
        self.processed_dirname = None
        self.skip_header = skip_header
        self.days_with_missing_data = []
        
    def check_for_dir(self):
        if self.city not in next(os.walk(self.path_to_data+'/raw'))[1]:
            return False
        else:
            if self.year not in next(os.walk(self.path_to_data+'/raw/'+self.city))[1]:
                return False
            else:
                self.raw_dirname = self.path_to_data+'/raw/'+self.city+'/'+self.year
                return True
    
    def get_files(self):
        return os.listdir(self.raw_dirname)

    def get_column_names(self,path_to_column_names):
        column_file = open(path_to_column_names, 'rb')
        column = pickle.load(column_file)
        return column
    
    def make_processed_dir(self):
        if 'processed' not in next(os.walk(self.path_to_data))[1]:
            os.mkdir(self.path_to_data+'/processed')
        
        if self.city not in next(os.walk(self.path_to_data+'/processed'))[1]:
            os.mkdir(self.path_to_data+'/processed/'+self.city)
            
        if self.year not in next(os.walk(self.path_to_data+'/processed/'+self.city))[1]:
            self.processed_dirname = self.path_to_data+'/processed/'+self.city+'/'+self.year
            os.mkdir(self.processed_dirname)
        else:
            self.processed_dirname = self.path_to_data+'/processed/'+self.city+'/'+self.year
            print('Directory already exists')


    def get_file_name(self,dataframe):
        file_time = str(int(dataframe.year[0])) + '-' + str(int(dataframe.month[0])) + '-' + str(int(dataframe.day[0]))
        return self.processed_dirname + '/data_' + file_time + '.csv'

    def store_file(self, dataframe):
        file_time = str(int(dataframe.year[0]))+'-'+str(int(dataframe.month[0]))+'-'+str(int(dataframe.day[0]))
        dataframe.to_csv(self.processed_dirname+'/data_'+file_time+'.csv', sep = ',')

    def check_if_enough_data(self, data):
        if data.shape[0] > 480:
            ignore_threshold = 1400
            full_size = 1440
        else:
            ignore_threshold = 460
            full_size = 480

        if data.shape[0] != full_size:
            if data.shape[0] < ignore_threshold:
                self.days_with_missing_data.append(self.get_file_name(data))

    def process(self, path_to_column_names):
        if self.check_for_dir():
            self.make_processed_dir()
            file_paths = self.get_files()
            for path in tqdm(file_paths):
                data = np.loadtxt(self.raw_dirname+'/'+path, skiprows=self.skip_header)
                column_names = self.get_column_names(path_to_column_names)
                print(path)
                print(data.shape, len(column_names))
                if data.ndim==2:
                    data_frame = pd.DataFrame(data=data, columns=column_names)
                    self.check_if_enough_data(data_frame)
                    self.store_file(data_frame)

            return self.days_with_missing_data

        else:
            print('Mentioned city/year does not exist')
            return None


# ### has to be shifted to scripts
# path_to_data = "/Users/saumya/Desktop/SolarProject/Data"
# path_to_missing_days_file = path_to_data+"/missing_days_files.pkl"
# list_of_missing_days_for_all_cities={}
#
# cities = ['Penn_State_PA']
# years = [2005,2006,2007,2008,2009]
# for city in cities:
#     days_with_missing_days_for_this_city=[]
#     for year in years:
#         object = SurfradDataCleaner(city, year, path_to_data)
#         days_with_missing_days_for_this_city_year = object.process(path_to_column_names = 'column_names.pkl')
#
#         if days_with_missing_days_for_this_city_year:
#             days_with_missing_days_for_this_city.extend(days_with_missing_days_for_this_city_year)
#
#     list_of_missing_days_for_all_cities[city] = days_with_missing_days_for_this_city
#
# with open(path_to_missing_days_file, 'wb') as file:
#     pickle.dump(list_of_missing_days_for_all_cities, file)