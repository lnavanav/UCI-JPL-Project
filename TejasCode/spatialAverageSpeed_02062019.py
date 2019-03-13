# Functions

from netCDF4 import Dataset, num2date
import numpy as np
import numpy.ma as ma

def read_data_wind(filename): # note the change of function name (just for proper naming)
    f = Dataset(filename)  # open the first file to check the dimension of U and V
    lon = f.variables['lon'][:]
    lat = f.variables['lat'][:]
    time = f.variables['time'] 
    date = num2date(time[:], time.units)
    u = f.variables['U'][:]
    v = f.variables['V'][:]
    speed = f.variables['speed'][:]
    sample_size = f.variables['sample_size'][:] # note the addition of this line to get the sample_size masked array
    return u, v, speed, lon, lat, date, sample_size # For merra2, comment out the sample_size stuff

from mpl_toolkits.basemap import Basemap, maskoceans

def calc_area_weighted_spatial_average(data, lon, lat, masking = 'mask_off', area_weight=True):
    '''Calculate area weighted average of the values in data
    :param data: two-dimensional masked array
    :type dataset: :class:`numpy.ma.core.MaskedArray`
    :returns: an area weighted mean value
    '''

    if lat.ndim == 1:
        lons, lats = np.meshgrid(lon, lat)
    else:
        lats = lat
    weights = np.cos(lats * np.pi / 180.)
    
    if masking == 'mask_ocean':
        masked_data = maskoceans(lons,lats,data) # masking oceans
        data = masked_data
    if masking == 'mask_land':
        masked_data = maskoceans(lons,lats,data) # masking oceans
        masked_data.mask = ~masked_data.mask # 'inverting' the mask to instead mask land
        data = masked_data
        
    if area_weight:
        spatial_average = ma.average(
            data[:], weights=weights)
    else:
        spatial_average = ma.average(data)

    return spatial_average

# Results Code

misr_u, misr_v, misr_speed, misr_lon, misr_lat, misr_time, misr_sample_size = read_data_wind('MISR_CMV_MAR2000-FEB2018_monthly.nc')

# First, the Spatial Average of the Full Scale Temporally Averaged Data

misr_speed_average_1p5km = ma.mean(misr_speed[:,0,:,:], axis = 0)
misr_speed_average_2p5km = ma.mean(misr_speed[:,1,:,:], axis = 0)
misr_speed_average_4p0km = ma.mean(misr_speed[:,2,:,:], axis = 0)

global_average_speed_1p5km_unmasked = calc_area_weighted_spatial_average(misr_speed_average_1p5km, 
                                                                         misr_lon, misr_lat, 'mask_off')
global_average_speed_2p5km_unmasked = calc_area_weighted_spatial_average(misr_speed_average_2p5km, 
                                                                         misr_lon, misr_lat, 'mask_off')
global_average_speed_4p0km_unmasked = calc_area_weighted_spatial_average(misr_speed_average_4p0km, 
                                                                         misr_lon, misr_lat, 'mask_off')

global_average_speed_1p5km_maskedocean = calc_area_weighted_spatial_average(misr_speed_average_1p5km, 
                                                                         misr_lon, misr_lat, 'mask_ocean')
global_average_speed_2p5km_maskedocean = calc_area_weighted_spatial_average(misr_speed_average_2p5km, 
                                                                         misr_lon, misr_lat, 'mask_ocean')
global_average_speed_4p0km_maskedocean = calc_area_weighted_spatial_average(misr_speed_average_4p0km, 
                                                                         misr_lon, misr_lat, 'mask_ocean')

global_average_speed_1p5km_maskedland = calc_area_weighted_spatial_average(misr_speed_average_1p5km, 
                                                                         misr_lon, misr_lat, 'mask_land')
global_average_speed_2p5km_maskedland = calc_area_weighted_spatial_average(misr_speed_average_2p5km, 
                                                                         misr_lon, misr_lat, 'mask_land')
global_average_speed_4p0km_maskedland = calc_area_weighted_spatial_average(misr_speed_average_4p0km, 
                                                                         misr_lon, misr_lat, 'mask_land')

# Pretty Printing
print("Spatial Average Speed for Full Scale Temporally Averaged Data")
print('-'*len("Spatial Average Speed for Full Scale Temporally Averaged Data"))
print("Full Averages by Elevation")
print(global_average_speed_1p5km_unmasked, global_average_speed_2p5km_unmasked, global_average_speed_4p0km_unmasked, sep='\n')
print("Land Averages by Elevation")
print(global_average_speed_1p5km_maskedocean, global_average_speed_2p5km_maskedocean, global_average_speed_4p0km_maskedocean, sep='\n')
print("Ocean Averages by Elevation")
print(global_average_speed_1p5km_maskedland, global_average_speed_2p5km_maskedland, global_average_speed_4p0km_maskedland, sep='\n')

# Now, we look at seasonal statistics.

misr_months = np.array([i.month for i in misr_time])

misr_t_index_winter = np.where((misr_months >= 12) | (misr_months <= 2))[0]
misr_t_index_spring = np.where(np.logical_and(misr_months >= 3, misr_months <= 5))[0]
misr_t_index_summer = np.where(np.logical_and(misr_months >= 6, misr_months <= 8))[0]
misr_t_index_fall = np.where(np.logical_and(misr_months >= 9, misr_months <= 11))[0]

misr_speed_average_winter_1p5km = ma.mean(misr_speed[misr_t_index_winter,0,:,:], axis = 0)
misr_speed_average_winter_2p5km = ma.mean(misr_speed[misr_t_index_winter,1,:,:], axis = 0)
misr_speed_average_winter_4p0km = ma.mean(misr_speed[misr_t_index_winter,2,:,:], axis = 0)

misr_speed_average_spring_1p5km = ma.mean(misr_speed[misr_t_index_spring,0,:,:], axis = 0)
misr_speed_average_spring_2p5km = ma.mean(misr_speed[misr_t_index_spring,1,:,:], axis = 0)
misr_speed_average_spring_4p0km = ma.mean(misr_speed[misr_t_index_spring,2,:,:], axis = 0)

misr_speed_average_summer_1p5km = ma.mean(misr_speed[misr_t_index_summer,0,:,:], axis = 0)
misr_speed_average_summer_2p5km = ma.mean(misr_speed[misr_t_index_summer,1,:,:], axis = 0)
misr_speed_average_summer_4p0km = ma.mean(misr_speed[misr_t_index_summer,2,:,:], axis = 0)

misr_speed_average_fall_1p5km = ma.mean(misr_speed[misr_t_index_fall,0,:,:], axis = 0)
misr_speed_average_fall_2p5km = ma.mean(misr_speed[misr_t_index_fall,1,:,:], axis = 0)
misr_speed_average_fall_4p0km = ma.mean(misr_speed[misr_t_index_fall,2,:,:], axis = 0)

# Summer Season
global_average_speed_1p5km_summer_unmasked = calc_area_weighted_spatial_average(misr_speed_average_summer_1p5km,
                                                                                misr_lon, misr_lat)
global_average_speed_2p5km_summer_unmasked = calc_area_weighted_spatial_average(misr_speed_average_summer_2p5km,
                                                                           misr_lon, misr_lat)
global_average_speed_4p0km_summer_unmasked = calc_area_weighted_spatial_average(misr_speed_average_summer_4p0km,
                                                                           misr_lon, misr_lat)

global_average_speed_1p5km_summer_maskedocean = calc_area_weighted_spatial_average(misr_speed_average_summer_1p5km,
                                                                       misr_lon, misr_lat, 'mask_ocean')
global_average_speed_2p5km_summer_maskedocean = calc_area_weighted_spatial_average(misr_speed_average_summer_2p5km,
                                                                       misr_lon, misr_lat, 'mask_ocean')
global_average_speed_4p0km_summer_maskedocean = calc_area_weighted_spatial_average(misr_speed_average_summer_4p0km,
                                                                       misr_lon, misr_lat, 'mask_ocean')

global_average_speed_1p5km_summer_maskedland = calc_area_weighted_spatial_average(misr_speed_average_summer_1p5km,
                                                                       misr_lon, misr_lat, 'mask_land')
global_average_speed_2p5km_summer_maskedland = calc_area_weighted_spatial_average(misr_speed_average_summer_2p5km,
                                                                       misr_lon, misr_lat, 'mask_land')
global_average_speed_4p0km_summer_maskedland = calc_area_weighted_spatial_average(misr_speed_average_summer_4p0km,
                                                                       misr_lon, misr_lat, 'mask_land')

# Winter Season
global_average_speed_1p5km_winter_unmasked = calc_area_weighted_spatial_average(misr_speed_average_winter_1p5km,
                                                                           misr_lon, misr_lat)
global_average_speed_2p5km_winter_unmasked = calc_area_weighted_spatial_average(misr_speed_average_winter_2p5km,
                                                                           misr_lon, misr_lat)
global_average_speed_4p0km_winter_unmasked = calc_area_weighted_spatial_average(misr_speed_average_winter_4p0km,
                                                                           misr_lon, misr_lat)

global_average_speed_1p5km_winter_maskedocean = calc_area_weighted_spatial_average(misr_speed_average_winter_1p5km,
                                                                       misr_lon, misr_lat, 'mask_ocean')
global_average_speed_2p5km_winter_maskedocean = calc_area_weighted_spatial_average(misr_speed_average_winter_2p5km,
                                                                       misr_lon, misr_lat, 'mask_ocean')
global_average_speed_4p0km_winter_maskedocean = calc_area_weighted_spatial_average(misr_speed_average_winter_4p0km,
                                                                       misr_lon, misr_lat, 'mask_ocean')

global_average_speed_1p5km_winter_maskedland = calc_area_weighted_spatial_average(misr_speed_average_winter_1p5km,
                                                                       misr_lon, misr_lat, 'mask_land')
global_average_speed_2p5km_winter_maskedland = calc_area_weighted_spatial_average(misr_speed_average_winter_2p5km,
                                                                       misr_lon, misr_lat, 'mask_land')
global_average_speed_4p0km_winter_maskedland = calc_area_weighted_spatial_average(misr_speed_average_winter_4p0km,
                                                                       misr_lon, misr_lat, 'mask_land')

# Fall Season
global_average_speed_1p5km_fall_unmasked = calc_area_weighted_spatial_average(misr_speed_average_fall_1p5km,
                                                                           misr_lon, misr_lat)
global_average_speed_2p5km_fall_unmasked = calc_area_weighted_spatial_average(misr_speed_average_fall_2p5km,
                                                                           misr_lon, misr_lat)
global_average_speed_4p0km_fall_unmasked = calc_area_weighted_spatial_average(misr_speed_average_fall_4p0km,
                                                                           misr_lon, misr_lat)

global_average_speed_1p5km_fall_maskedocean = calc_area_weighted_spatial_average(misr_speed_average_fall_1p5km,
                                                                       misr_lon, misr_lat, 'mask_ocean')
global_average_speed_2p5km_fall_maskedocean = calc_area_weighted_spatial_average(misr_speed_average_fall_2p5km,
                                                                       misr_lon, misr_lat, 'mask_ocean')
global_average_speed_4p0km_fall_maskedocean = calc_area_weighted_spatial_average(misr_speed_average_fall_4p0km,
                                                                       misr_lon, misr_lat, 'mask_ocean')

global_average_speed_1p5km_fall_maskedland = calc_area_weighted_spatial_average(misr_speed_average_fall_1p5km,
                                                                       misr_lon, misr_lat, 'mask_land')
global_average_speed_2p5km_fall_maskedland = calc_area_weighted_spatial_average(misr_speed_average_fall_2p5km,
                                                                       misr_lon, misr_lat, 'mask_land')
global_average_speed_4p0km_fall_maskedland = calc_area_weighted_spatial_average(misr_speed_average_fall_4p0km,
                                                                       misr_lon, misr_lat, 'mask_land')

# Spring Season
global_average_speed_1p5km_spring_unmasked = calc_area_weighted_spatial_average(misr_speed_average_spring_1p5km,
                                                                           misr_lon, misr_lat)
global_average_speed_2p5km_spring_unmasked = calc_area_weighted_spatial_average(misr_speed_average_spring_2p5km,
                                                                           misr_lon, misr_lat)
global_average_speed_4p0km_spring_unmasked = calc_area_weighted_spatial_average(misr_speed_average_spring_4p0km,
                                                                           misr_lon, misr_lat)

global_average_speed_1p5km_spring_maskedocean = calc_area_weighted_spatial_average(misr_speed_average_spring_1p5km,
                                                                       misr_lon, misr_lat, 'mask_ocean')
global_average_speed_2p5km_spring_maskedocean = calc_area_weighted_spatial_average(misr_speed_average_spring_2p5km,
                                                                       misr_lon, misr_lat, 'mask_ocean')
global_average_speed_4p0km_spring_maskedocean = calc_area_weighted_spatial_average(misr_speed_average_spring_4p0km,
                                                                       misr_lon, misr_lat, 'mask_ocean')

global_average_speed_1p5km_spring_maskedland = calc_area_weighted_spatial_average(misr_speed_average_spring_1p5km,
                                                                       misr_lon, misr_lat, 'mask_land')
global_average_speed_2p5km_spring_maskedland = calc_area_weighted_spatial_average(misr_speed_average_spring_2p5km,
                                                                       misr_lon, misr_lat, 'mask_land')
global_average_speed_4p0km_spring_maskedland = calc_area_weighted_spatial_average(misr_speed_average_spring_4p0km,
                                                                       misr_lon, misr_lat, 'mask_land')

# Pretty Printing
print("Spatial Averages for Summer Season Temporally Averaged Data")
print('-'*len("Spatial Averages for Summer Season Temporally Averaged Data"))
print("Full Averages by Elevation")
print(global_average_speed_1p5km_summer_unmasked, global_average_speed_2p5km_summer_unmasked, global_average_speed_4p0km_summer_unmasked, sep='\n')
print("Land Averages by Elevation")
print(global_average_speed_1p5km_summer_maskedocean, global_average_speed_2p5km_summer_maskedocean, global_average_speed_4p0km_summer_maskedocean, sep='\n')
print("Ocean Averages by Elevation")
print(global_average_speed_1p5km_summer_maskedland, global_average_speed_2p5km_summer_maskedland, global_average_speed_4p0km_summer_maskedland, sep='\n')

print("Spatial Averages for Winter Season Temporally Averaged Data")
print('-'*len("Spatial Averages for Winter Season Temporally Averaged Data"))
print("Full Averages by Elevation")
print(global_average_speed_1p5km_winter_unmasked, global_average_speed_2p5km_winter_unmasked, global_average_speed_4p0km_winter_unmasked, sep='\n')
print("Land Averages by Elevation")
print(global_average_speed_1p5km_winter_maskedocean, global_average_speed_2p5km_winter_maskedocean, global_average_speed_4p0km_winter_maskedocean, sep='\n')
print("Ocean Averages by Elevation")
print(global_average_speed_1p5km_winter_maskedland, global_average_speed_2p5km_winter_maskedland, global_average_speed_4p0km_winter_maskedland, sep='\n')

print("Spatial Averages for Fall Season Temporally Averaged Data")
print('-'*len("Spatial Averages for Fall Season Temporally Averaged Data"))
print("Full Averages by Elevation")
print(global_average_speed_1p5km_fall_unmasked, global_average_speed_2p5km_fall_unmasked, global_average_speed_4p0km_fall_unmasked, sep='\n')
print("Land Averages by Elevation")
print(global_average_speed_1p5km_fall_maskedocean, global_average_speed_2p5km_fall_maskedocean, global_average_speed_4p0km_fall_maskedocean, sep='\n')
print("Ocean Averages by Elevation")
print(global_average_speed_1p5km_fall_maskedland, global_average_speed_2p5km_fall_maskedland, global_average_speed_4p0km_fall_maskedland, sep='\n')

print("Spatial Averages for Spring Season Temporally Averaged Data")
print('-'*len("Spatial Averages for Spring Season Temporally Averaged Data"))
print("Full Averages by Elevation")
print(global_average_speed_1p5km_spring_unmasked, global_average_speed_2p5km_spring_unmasked, global_average_speed_4p0km_spring_unmasked, sep='\n')
print("Land Averages by Elevation")
print(global_average_speed_1p5km_spring_maskedocean, global_average_speed_2p5km_spring_maskedocean, global_average_speed_4p0km_spring_maskedocean, sep='\n')
print("Ocean Averages by Elevation")
print(global_average_speed_1p5km_spring_maskedland, global_average_speed_2p5km_spring_maskedland, global_average_speed_4p0km_spring_maskedland, sep='\n')

