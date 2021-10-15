import numpy as np
import netCDF4
import os

site_names = ["Bondville",
                "Boulder",
                "Desert_Rock",
                "Fort_Peck",
                "Goodwin_Creek",
                "Penn_State",
                "Sioux_Falls"]

site_coord = {"Bondville":(40.05192, 88.37309),
            "Boulder":(40.12498, 105.23680),
            "Desert_Rock":(36.62373, 116.01947),
            "Fort_Peck":(48.30783, 105.10170),
            "Goodwin_Creek":(34.2547, 89.8729),
            "Penn_State":(40.72012, 77.93085),
            "Sioux_Falls":(43.73403, 96.62328)}

path_mars = "/Users/saumya/Desktop/mars/bin/"
path_project = "/Users/saumya/Desktop/SolarProject/"

path_input = path_mars + "Raw/"
path_output = path_project+"Data/NWP/"


def find_nearest(array, value):
    # array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def get_site_array(site,year):

    xp = site_coord[site][0]
    yp = site_coord[site][1]

    print(xp,yp)
    path = path_input + str(year) + "/"

    nwp_stacked_array = []
    for month in ['01','02','03','04','05','06','07','08','09','10','11','12']:

        fp = path+"north_america_"+str(year)+"-"+month+".nc"
        nc = netCDF4.Dataset(fp)

        lon = np.array(nc.variables['longitude'][:])
        lat = np.array(nc.variables['latitude'][:])

        #
        # lon_index = np.where((np.around(lon, 1) == -np.around(yp,1)) | (np.around(lon, 1) == -np.around(yp,1)+0.1))[0][0]
        # lat_index = np.where((np.round(lat, 1) == np.around(xp,1)) | (np.round(lat, 1) == np.around(xp,1)-0.1))[0][0]

        lon_index = find_nearest(lon,-yp)
        lat_index = find_nearest(lat,xp)

        print(lon[lon_index],lat[lat_index])

        ssrd = np.array(nc.variables['ssrd'][:])
        ssrd_168 = ssrd[:, :, 1, :, :, :]

        ssrd_168_lat_long = ssrd_168[:,:,:,lat_index,lon_index]
        # print("ssrd_168_lat_long",ssrd_168_lat_long.shape)

        fp_ctrl = path+"ctrl_north_america_"+str(year)+"-"+month+".nc"
        nc_ctrl = netCDF4.Dataset(fp_ctrl)

        ssrd_ctrl = np.array(nc_ctrl.variables['ssrd'][:])
        ssrd_ctrl_168 = ssrd_ctrl[:, :, 1, :, :]
        ssrd_ctrl_168 = np.expand_dims(ssrd_ctrl_168, axis=2)

        ssrd_ctrl_168_lat_long = ssrd_ctrl_168[:, :, :, lat_index, lon_index]
        # print("ssrd_ctrl_168_lat_long",ssrd_ctrl_168_lat_long.shape)

        ssrd_51 = np.concatenate((ssrd_168_lat_long, ssrd_ctrl_168_lat_long), axis=2)
        print("final ssrd 168 shape", ssrd_51.shape)
        nwp_stacked_array.append(ssrd_51)

    nwp_stacked_array = np.concatenate(nwp_stacked_array)
    print(nwp_stacked_array.shape)
    #(366, 2, 51)

    os.makedirs(
        path_output + site+"/",exist_ok=True)

    np.save(path_output + site+"/"+"nwp_"+str(year)+".npy",nwp_stacked_array)




def main():

    for year in [2017]:
        for site in site_names:
            get_site_array(site,year)


if __name__=='__main__':
    main()
