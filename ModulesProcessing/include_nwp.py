import os
import netCDF4
import numpy as np


path_mars = "/Users/saumya/Desktop/mars/bin/"
path_grib_files = path_mars + "GRIB_files/"
path_grib_ctrl_files = path_mars + "Ctrl_GRIB_files/"
path_raw = path_mars + "Raw/"


def convert_grib_to_nc(path):
    if path == path_grib_files:
        for filename in os.listdir(path):
            if filename.endswith(".grib"):
                file = filename[:-5] + ".nc"
                os.system('grib_to_netcdf -o ' + path_raw+file + ' -T ' + path+filename)

    else:
        for filename in os.listdir(path):
            if filename.endswith(".grib"):
                file = "ctrl_"+filename[:-5] + ".nc"
                os.system('grib_to_netcdf -o ' + path_raw+file + ' -T ' + path+filename)





# def read_nc_files(path, year):
#
#     path_year = path+str(year)+"/"
#     nwp_stacked_array = []
#     dirfiles = os.listdir(path_year)
#     print("before sort",dirfiles)
#     dirfiles.sort()
#     print("after sort", dirfiles)
#     for filename in dirfiles:
#         if filename.endswith(".nc"):
#             if not filename.startswith("ctrl"):
#                 print(filename)
#                 fp = path_year+filename
#                 nc = netCDF4.Dataset(fp)
#
#                 # for var in nc.variables.values():
#                 #     print(var)
#
#                 ssrd = np.array(nc.variables['ssrd'][:])
#                 print("ssrd shape",ssrd.shape)
#                 print(ssrd[:,0,1,0,0,0])
#
#                 ssrd_168 = ssrd[:,:,1,:,:,:]
#                 print("ssrd 168 shape",ssrd_168.shape)
#
#                 fp_ctrl = path_year+"ctrl_"+filename
#                 print(fp_ctrl)
#                 nc_ctrl = netCDF4.Dataset(fp_ctrl)
#
#                 ssrd_ctrl = np.array(nc_ctrl.variables['ssrd'][:])
#                 print("ssrd ctrl shape", ssrd_ctrl.shape)
#                 ssrd_ctrl_168 = ssrd_ctrl[:, :, 1, :, :]
#                 print("ssrd ctrl 168 shape", ssrd_ctrl_168.shape)
#                 ssrd_ctrl_168 = np.expand_dims(ssrd_ctrl_168, axis = 2)
#                 print("ssrd ctrl 168 added dim shape", ssrd_ctrl_168.shape)
#                 ssrd_51 = np.concatenate((ssrd_168,ssrd_ctrl_168),axis = 2)
#                 print("final ssrd 168 shape", ssrd_51.shape)
#                 nwp_stacked_array.append(ssrd_51)
#
#     print(len(nwp_stacked_array))
#     nwp_stacked_array = np.concatenate(nwp_stacked_array)
#     print(nwp_stacked_array.shape)
#     #(366, 2, 51, 72, 193)
#     np.save(path_year+"nwp_"+str(year)+".npy",nwp_stacked_array)



def main():

    convert_grib_to_nc(path_grib_files)
    convert_grib_to_nc(path_grib_ctrl_files)

    # for year in [2016]:
    #     read_nc_files(path_raw, year)



if __name__=='__main__':
    main()



# ssrd _FillValue: -32767