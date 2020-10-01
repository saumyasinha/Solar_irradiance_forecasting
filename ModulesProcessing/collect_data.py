import ftplib
import os
from tqdm import tqdm
import pickle


class SurfradDataCollector:
    '''
    This class downloads the .dat files from "ftp://aftp.cmdl.noaa.gov/data/radiation/surfrad/" for the given cities and years.
    '''
    def __init__(self, years, cities, path_to_data):
        if not os.path.exists(path_to_data):
            os.makedirs(path_to_data)
        self.path_to_data = path_to_data

        self.years = years
        self.cities = cities
        self.data_already_exists = False
        self.ftp = ftplib.FTP('aftp.cmdl.noaa.gov')
        
    def login_ftp(self):
        self.ftp.login('', '')
    
    # def get_column_headers(self, path_to_column_names):
    #     column_pkl = open(path_to_column_names, 'rb')
    #     columns = pickle.load(column_pkl)
    #     return columns
    
    def change_ftp_dir_year(self, year, city):
        self.ftp.cwd('/data/radiation/surfrad/'+str(city)+'/'+str(year))
    
    def get_file_list(self):
        return self.ftp.nlst()
    
    def make_year_dir(self, year, city_dir):
        download_dir_year = city_dir + '/' + str(year)
        print(download_dir_year)
        if year not in next(os.walk(self.path_to_data+'/raw'))[1]:
            os.mkdir(download_dir_year)
        else:
            self.data_already_exists = True
        return download_dir_year
    
    def make_city_dir(self, city):
        download_dir_city = self.path_to_data+'/raw/' + city
        print(download_dir_city)
        if city not in next(os.walk(self.path_to_data+'/raw'))[1]:
            os.mkdir(download_dir_city)
        else:
            print('this city already exists in DB')
        return download_dir_city
    
    def make_raw_dir(self):
        if 'raw' not in next(os.walk(self.path_to_data))[1]:
            os.mkdir(self.path_to_data+'/raw')
        else:
            pass
        
    def download_file(self, dirname, name):
        file = open(dirname+'/'+name, 'wb')
        self.ftp.retrbinary('RETR '+name, file.write)
        file.close()
        
    def download_data(self):
        self.login_ftp()
        # column_headers = self.get_column_headers(path_to_column_names)
        self.make_raw_dir()
        for city in self.cities:
            city_dir = self.make_city_dir(city)
            for year in self.years:
                download_dir_path = self.make_year_dir(year, city_dir)
                if not self.data_already_exists:
                    self.change_ftp_dir_year(year, city)
                    file_names = self.get_file_list()
                    for name in tqdm(file_names):
                        self.download_file(download_dir_path, name)
        return True

### has to be shifted to scripts
path_to_data = "/Users/saumya/Desktop/SolarProject/Data"
object = SurfradDataCollector([2005,2006,2007,2008,2009], ['Penn_State_PA'], path_to_data)

object.download_data()

