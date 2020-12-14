import pandas as pd
import netCDF4
from scipy import stats


path_project = "/Users/saumya/Desktop/SolarProject/"
path = path_project+"Data/"
path_nwp = path+"ECMWF/"
fp=path_nwp + 'Penn_State.nc'
nc = netCDF4.Dataset(fp)


num_variables = len(nc.variables.values())

for var in nc.variables.values():
    print(var)

for dim in nc.dimensions.values():
    print(dim)

IssueTime = nc.variables["IssueTime"][:]

LeadTime = nc.variables["LeadTime"][:]

Irradiance = nc.variables["Member"][:]

Day = nc.variables["Day"][:]


print(nc.variables["irradiance"][:,10,2,300])




