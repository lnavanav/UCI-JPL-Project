from netCDF4 import Dataset, num2date
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.basemap import Basemap, maskoceans

def read_merra2_wind(filename):
    f = Dataset(filename)  # open the first file to check the dimension of U and V
    lon = f.variables['lon'][:]
    lat = f.variables['lat'][:]
    time = f.variables['time'] 
    date = num2date(time[:], time.units)
    u = f.variables['U'][:]
    v = f.variables['V'][:]
    speed = f.variables['speed'][:]
    return u, v, speed, lon, lat, date

def plot_global_contour_map(data, lon, lat, levels, figure_file, title='title', cmap=cm.jet): #added title
    lons, lats = np.meshgrid(lon, lat)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    m = Basemap(ax=ax, projection = 'eck4', lon_0=0, llcrnrlat = lat.min(), urcrnrlat = lat.max(), 
        llcrnrlon = lon.min(), urcrnrlon = lon.max(), resolution = 'l', fix_aspect = True)
    x,y = m(lons, lats)
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=.75)
    max = m.contourf(x, y, data, levels = levels, extend='both', cmap=cmap)
    cbar_ax = fig.add_axes([0.92, 0.3,0.01, 0.4])
    cb=plt.colorbar(max,cax=cbar_ax)
    cb.ax.tick_params(labelsize=8)
    ax.set_title(title) #NEW!!! Sets title above plot
    fig.savefig(figure_file,dpi=600,bbox_inches='tight')
    plt.show()
	
def plot_global_contour_map_maskedland(data, lon, lat, levels, figure_file, title='title', cmap=cm.jet): #added title
    lons, lats = np.meshgrid(lon, lat)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    m = Basemap(ax=ax, projection = 'eck4', lon_0=0, llcrnrlat = lat.min(), urcrnrlat = lat.max(), 
        llcrnrlon = lon.min(), urcrnrlon = lon.max(), resolution = 'l', fix_aspect = True)
    x,y = m(lons, lats)
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=.5)
    #MASKING STUFF START
    masked_data = maskoceans(lons,lats,data)
    masked_data.mask = ~masked_data.mask #reverse the mask
    #MASKING STUFF END
    max = m.contourf(x, y, masked_data, levels = levels, extend='both', cmap=cmap)
    cbar_ax = fig.add_axes([0.92, 0.3,0.01, 0.4])
    cb=plt.colorbar(max,cax=cbar_ax)
    cb.ax.tick_params(labelsize=8)
    ax.set_title(title) #NEW!!! Sets title above plot
    fig.savefig(figure_file,dpi=600,bbox_inches='tight')
    plt.show()
    
def plot_global_vector_map(uwind, vwind, lon, lat, figure_file, yskip=10, xskip=20):
    lons, lats = np.meshgrid(lon, lat)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    m = Basemap(ax=ax, projection = 'eck4', lon_0=0, llcrnrlat = lat.min(), urcrnrlat = lat.max(),
        llcrnrlon = lon.min(), urcrnrlon = lon.max(), resolution = 'l', fix_aspect = True)
    x,y = m(lons, lats)
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=.75)
    N = ma.mean(np.sqrt(uwind[::yskip, ::xskip]**2+vwind[::yskip, ::xskip]**2))
    max = m.quiver(x[::yskip, ::xskip], y[::yskip, ::xskip], uwind[::yskip, ::xskip]/N, vwind[::yskip, ::xskip]/N, color='blue', pivot='middle', headwidth=3)
    fig.savefig(figure_file,dpi=600,bbox_inches='tight')
    plt.show()
	
def plot_global_vector_map_maskedland(uwind, vwind, lon, lat, figure_file, title='title', yskip=10, xskip=20):
    lons, lats = np.meshgrid(lon, lat)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    m = Basemap(ax=ax, projection = 'eck4', lon_0=0, llcrnrlat = lat.min(), urcrnrlat = lat.max(),
        llcrnrlon = lon.min(), urcrnrlon = lon.max(), resolution = 'l', fix_aspect = True)
    x,y = m(lons, lats)
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=.5)
	#MASKING STUFF START
    masked_u = maskoceans(lons,lats,uwind)
	masked_v = maskoceans(lons,lats,vwind)
    masked_u.mask = ~masked_u.mask #reverse the mask
	masked_v.mask = ~masked_v.mask #reverse the mask
    #MASKING STUFF END
    N = ma.mean(np.sqrt(masked_u[::yskip, ::xskip]**2+masked_v[::yskip, ::xskip]**2))
    max = m.quiver(x[::yskip, ::xskip], y[::yskip, ::xskip], masked_u[::yskip, ::xskip]/N, masked_v[::yskip, ::xskip]/N, color='blue', pivot='middle', headwidth=3)
	ax.set_title(title) #NEW!!! Sets title above plot
    fig.savefig(figure_file,dpi=600,bbox_inches='tight')
    plt.show()
	
#merra_u, merra_v, merra_speed, lon, lat, merra_time = read_merra2_wind('merra2_wind_MAR2000-FEB2018_18UTC_monthly.nc')
merra_u, merra_v, merra_speed, merra_lon, merra_lat, merra_time = read_merra2_wind('merra2_wind_MAR2000-FEB2018_monthly.nc')
misr_u, misr_v, misr_speed, misr_lon, misr_lat, misr_time = read_merra2_wind('MISR_CMV_MAR2000-FEB2018_monthly.nc')

###Seasonal times###
#MERRA and MISR share a common time so no need to do it twice. Northern hemisphere seasons are used.
months = np.array([i.month for i in merra_time]) 
winter_t_index = np.where((months >=12) | (months <=2))[0] #1 December to 28 February
spring_t_index = np.where((months >=3) & (months <= 5))[0] #1 March to 31 May
summer_t_index = np.where((months >=6) & (months <= 8))[0] #1 June to 31 August
fall_t_index = np.where((months >=9) & (months <=11))[0] #1 September to 30 November 

#OBSOLETE - Saving for documentation at the moment
#speed_diff_1p5km = ma.mean(merra_speed[:, 0, :, :] - misr_speed[:, 0, :, :], axis=0) # - ma.mean(misr_speed[:, 0, :, :], axis=0) #BAD
#speed_diff_1p5km = ma.mean(merra_speed[:, 0, :, :], axis=0) - ma.mean(misr_speed[:, 0, :, :], axis=0) #Same as mask
#speed_diff_2p5km = ma.mean(merra_speed[:, 1, :, :], axis=0) - ma.mean(misr_speed[:, 1, :, :], axis=0)
#speed_diff_4km = ma.mean(merra_speed[:, 2, :, :], axis=0) - ma.mean(misr_speed[:, 2, :, :], axis=0)


#####################################################################################   VARIABLES   #####################################################################################


############# SPEED STUFF (CONTOUR PLOT) ############# 

#Both methods of finding the difference (mean - mean, mean(-)) are the same under this masked scheme, so it's ideal for 
#accurate averaging. I'm almost certain there's a cleaner and better way of doing it though. This might be the same as
#the mean(merra-misr) plotting - graphs look identical.
#This gets the mask of both datasets and applies the MERRA mask to the MISR data and vice versa to produce a uniformly masked dataset on which calculations are done.

#Masks - no need for seasonal masks because we can just take subsets of these for each season
mask_of_merra_1p5km = ma.getmask(merra_speed[:, 0, :, :])
mask_of_misr_1p5km = ma.getmask(misr_speed[:, 0, :, :])
mask_of_merra_2p5km = ma.getmask(merra_speed[:, 1, :, :])
mask_of_misr_2p5km = ma.getmask(misr_speed[:, 1, :, :])
mask_of_merra_4km = ma.getmask(merra_speed[:, 2, :, :])
mask_of_misr_4km = ma.getmask(misr_speed[:, 2, :, :])

##Speeds##

#Winter
merra_speed_winter_masked_by_misr_1p5km = ma.array(merra_speed[winter_t_index, 0, :, :], mask=mask_of_misr_1p5km[winter_t_index])
misr_speed_winter_masked_by_merra_1p5km = ma.array(misr_speed[winter_t_index, 0, :, :], mask=mask_of_merra_1p5km[winter_t_index])
merra_speed_winter_masked_by_misr_2p5km = ma.array(merra_speed[winter_t_index, 1, :, :], mask=mask_of_misr_2p5km[winter_t_index])
misr_speed_winter_masked_by_merra_2p5km = ma.array(misr_speed[winter_t_index, 1, :, :], mask=mask_of_merra_2p5km[winter_t_index])
merra_speed_winter_masked_by_misr_4km = ma.array(merra_speed[winter_t_index, 2, :, :], mask=mask_of_misr_4km[winter_t_index])
misr_speed_winter_masked_by_merra_4km = ma.array(misr_speed[winter_t_index, 2, :, :], mask=mask_of_merra_4km[winter_t_index])
speed_diff_winter_1p5km = ma.mean(merra_speed_winter_masked_by_misr_1p5km - misr_speed_winter_masked_by_merra_1p5km, axis = 0)
speed_diff_winter_2p5km = ma.mean(merra_speed_winter_masked_by_misr_2p5km - misr_speed_winter_masked_by_merra_2p5km, axis = 0)
speed_diff_winter_4km = ma.mean(merra_speed_winter_masked_by_misr_4km - misr_speed_winter_masked_by_merra_4km, axis = 0)

#Spring
merra_speed_spring_masked_by_misr_1p5km = ma.array(merra_speed[spring_t_index, 0, :, :], mask=mask_of_misr_1p5km[spring_t_index])
misr_speed_spring_masked_by_merra_1p5km = ma.array(misr_speed[spring_t_index, 0, :, :], mask=mask_of_merra_1p5km[spring_t_index])
merra_speed_spring_masked_by_misr_2p5km = ma.array(merra_speed[spring_t_index, 1, :, :], mask=mask_of_misr_2p5km[spring_t_index])
misr_speed_spring_masked_by_merra_2p5km = ma.array(misr_speed[spring_t_index, 1, :, :], mask=mask_of_merra_2p5km[spring_t_index])
merra_speed_spring_masked_by_misr_4km = ma.array(merra_speed[spring_t_index, 2, :, :], mask=mask_of_misr_4km[spring_t_index])
misr_speed_spring_masked_by_merra_4km = ma.array(misr_speed[spring_t_index, 2, :, :], mask=mask_of_merra_4km[spring_t_index])
speed_diff_spring_1p5km = ma.mean(merra_speed_spring_masked_by_misr_1p5km - misr_speed_spring_masked_by_merra_1p5km, axis = 0)
speed_diff_spring_2p5km = ma.mean(merra_speed_spring_masked_by_misr_2p5km - misr_speed_spring_masked_by_merra_2p5km, axis = 0)
speed_diff_spring_4km = ma.mean(merra_speed_spring_masked_by_misr_4km - misr_speed_spring_masked_by_merra_4km, axis = 0)

#Summer
merra_speed_summer_masked_by_misr_1p5km = ma.array(merra_speed[summer_t_index, 0, :, :], mask=mask_of_misr_1p5km[summer_t_index])
misr_speed_summer_masked_by_merra_1p5km = ma.array(misr_speed[summer_t_index, 0, :, :], mask=mask_of_merra_1p5km[summer_t_index])
merra_speed_summer_masked_by_misr_2p5km = ma.array(merra_speed[summer_t_index, 1, :, :], mask=mask_of_misr_2p5km[summer_t_index])
misr_speed_summer_masked_by_merra_2p5km = ma.array(misr_speed[summer_t_index, 1, :, :], mask=mask_of_merra_2p5km[summer_t_index])
merra_speed_summer_masked_by_misr_4km = ma.array(merra_speed[summer_t_index, 2, :, :], mask=mask_of_misr_4km[summer_t_index])
misr_speed_summer_masked_by_merra_4km = ma.array(misr_speed[summer_t_index, 2, :, :], mask=mask_of_merra_4km[summer_t_index])
speed_diff_summer_1p5km = ma.mean(merra_speed_summer_masked_by_misr_1p5km - misr_speed_summer_masked_by_merra_1p5km, axis = 0)
speed_diff_summer_2p5km = ma.mean(merra_speed_summer_masked_by_misr_2p5km - misr_speed_summer_masked_by_merra_2p5km, axis = 0)
speed_diff_summer_4km = ma.mean(merra_speed_summer_masked_by_misr_4km - misr_speed_summer_masked_by_merra_4km, axis = 0)

#Fall
merra_speed_fall_masked_by_misr_1p5km = ma.array(merra_speed[fall_t_index, 0, :, :], mask=mask_of_misr_1p5km[fall_t_index])
misr_speed_fall_masked_by_merra_1p5km = ma.array(misr_speed[fall_t_index, 0, :, :], mask=mask_of_merra_1p5km[fall_t_index])
merra_speed_fall_masked_by_misr_2p5km = ma.array(merra_speed[fall_t_index, 1, :, :], mask=mask_of_misr_2p5km[fall_t_index])
misr_speed_fall_masked_by_merra_2p5km = ma.array(misr_speed[fall_t_index, 1, :, :], mask=mask_of_merra_2p5km[fall_t_index])
merra_speed_fall_masked_by_misr_4km = ma.array(merra_speed[fall_t_index, 2, :, :], mask=mask_of_misr_4km[fall_t_index])
misr_speed_fall_masked_by_merra_4km = ma.array(misr_speed[fall_t_index, 2, :, :], mask=mask_of_merra_4km[fall_t_index])
speed_diff_fall_1p5km = ma.mean(merra_speed_fall_masked_by_misr_1p5km - misr_speed_fall_masked_by_merra_1p5km, axis = 0)
speed_diff_fall_2p5km = ma.mean(merra_speed_fall_masked_by_misr_2p5km - misr_speed_fall_masked_by_merra_2p5km, axis = 0)
speed_diff_fall_4km = ma.mean(merra_speed_fall_masked_by_misr_4km - misr_speed_fall_masked_by_merra_4km, axis = 0)

#Overall
merra_speed_overall_masked_by_misr_1p5km = ma.array(merra_speed[:, 0, :, :], mask=mask_of_misr_1p5km)
misr_speed_overall_masked_by_merra_1p5km = ma.array(misr_speed[:, 0, :, :], mask=mask_of_merra_1p5km)
merra_speed_overall_masked_by_misr_2p5km = ma.array(merra_speed[:, 1, :, :], mask=mask_of_misr_2p5km)
misr_speed_overall_masked_by_merra_2p5km = ma.array(misr_speed[:, 1, :, :], mask=mask_of_merra_2p5km)
merra_speed_overall_masked_by_misr_4km = ma.array(merra_speed[:, 2, :, :], mask=mask_of_misr_4km)
misr_speed_overall_masked_by_merra_4km = ma.array(misr_speed[:, 2, :, :], mask=mask_of_merra_4km)
speed_diff_overall_1p5km = ma.mean(merra_speed_overall_masked_by_misr_1p5km - misr_speed_overall_masked_by_merra_1p5km, axis = 0)
speed_diff_overall_2p5km = ma.mean(merra_speed_overall_masked_by_misr_2p5km - misr_speed_overall_masked_by_merra_2p5km, axis = 0)
speed_diff_overall_4km = ma.mean(merra_speed_overall_masked_by_misr_4km - misr_speed_overall_masked_by_merra_4km, axis = 0)


##Shears##
#For this, the masks above come in handy. Differences in shear - because the speed_shear vars are already means, there shouldn't be a need to mask them again

#Winter
speed_shear_merra_winter_2p5km_1p5km = ma.mean(ma.array(merra_speed[winter_t_index, 1, :, :], mask=mask_of_merra_1p5km[winter_t_index]) - ma.array(merra_speed[winter_t_index, 0, :, :], mask=mask_of_merra_2p5km[winter_t_index]), axis = 0)
speed_shear_merra_winter_4km_1p5km = ma.mean(ma.array(merra_speed[winter_t_index, 2, :, :], mask=mask_of_merra_1p5km[winter_t_index]) - ma.array(merra_speed[winter_t_index, 0, :, :], mask=mask_of_merra_4km[winter_t_index]), axis = 0)
speed_shear_merra_winter_4km_2p5km = ma.mean(ma.array(merra_speed[winter_t_index, 2, :, :], mask=mask_of_merra_2p5km[winter_t_index]) - ma.array(merra_speed[winter_t_index, 1, :, :], mask=mask_of_merra_4km[winter_t_index]), axis = 0)
speed_shear_misr_winter_2p5km_1p5km = ma.mean(ma.array(misr_speed[winter_t_index, 1, :, :], mask=mask_of_misr_1p5km[winter_t_index]) - ma.array(misr_speed[winter_t_index, 0, :, :], mask=mask_of_misr_2p5km[winter_t_index]), axis = 0)
speed_shear_misr_winter_4km_1p5km = ma.mean(ma.array(misr_speed[winter_t_index, 2, :, :], mask=mask_of_misr_1p5km[winter_t_index]) - ma.array(misr_speed[winter_t_index, 0, :, :], mask=mask_of_misr_4km[winter_t_index]), axis = 0)
speed_shear_misr_winter_4km_2p5km = ma.mean(ma.array(misr_speed[winter_t_index, 2, :, :], mask=mask_of_misr_2p5km[winter_t_index]) - ma.array(misr_speed[winter_t_index, 1, :, :], mask=mask_of_misr_4km[winter_t_index]), axis = 0)
speed_shear_diff_winter_2p5km_1p5km = speed_shear_merra_winter_2p5km_1p5km - speed_shear_misr_winter_2p5km_1p5km
speed_shear_diff_winter_4km_1p5km = speed_shear_merra_winter_4km_1p5km - speed_shear_misr_winter_4km_1p5km
speed_shear_diff_winter_4km_2p5km = speed_shear_merra_winter_4km_2p5km - speed_shear_misr_winter_4km_2p5km

#Spring
speed_shear_merra_spring_2p5km_1p5km = ma.mean(ma.array(merra_speed[spring_t_index, 1, :, :], mask=mask_of_merra_1p5km[spring_t_index]) - ma.array(merra_speed[spring_t_index, 0, :, :], mask=mask_of_merra_2p5km[spring_t_index]), axis = 0)
speed_shear_merra_spring_4km_1p5km = ma.mean(ma.array(merra_speed[spring_t_index, 2, :, :], mask=mask_of_merra_1p5km[spring_t_index]) - ma.array(merra_speed[spring_t_index, 0, :, :], mask=mask_of_merra_4km[spring_t_index]), axis = 0)
speed_shear_merra_spring_4km_2p5km = ma.mean(ma.array(merra_speed[spring_t_index, 2, :, :], mask=mask_of_merra_2p5km[spring_t_index]) - ma.array(merra_speed[spring_t_index, 1, :, :], mask=mask_of_merra_4km[spring_t_index]), axis = 0)
speed_shear_misr_spring_2p5km_1p5km = ma.mean(ma.array(misr_speed[spring_t_index, 1, :, :], mask=mask_of_misr_1p5km[spring_t_index]) - ma.array(misr_speed[spring_t_index, 0, :, :], mask=mask_of_misr_2p5km[spring_t_index]), axis = 0)
speed_shear_misr_spring_4km_1p5km = ma.mean(ma.array(misr_speed[spring_t_index, 2, :, :], mask=mask_of_misr_1p5km[spring_t_index]) - ma.array(misr_speed[spring_t_index, 0, :, :], mask=mask_of_misr_4km[spring_t_index]), axis = 0)
speed_shear_misr_spring_4km_2p5km = ma.mean(ma.array(misr_speed[spring_t_index, 2, :, :], mask=mask_of_misr_2p5km[spring_t_index]) - ma.array(misr_speed[spring_t_index, 1, :, :], mask=mask_of_misr_4km[spring_t_index]), axis = 0)
speed_shear_diff_spring_2p5km_1p5km = speed_shear_merra_spring_2p5km_1p5km - speed_shear_misr_spring_2p5km_1p5km
speed_shear_diff_spring_4km_1p5km = speed_shear_merra_spring_4km_1p5km - speed_shear_misr_spring_4km_1p5km
speed_shear_diff_spring_4km_2p5km = speed_shear_merra_spring_4km_2p5km - speed_shear_misr_spring_4km_2p5km

#Summer
speed_shear_merra_summer_2p5km_1p5km = ma.mean(ma.array(merra_speed[summer_t_index, 1, :, :], mask=mask_of_merra_1p5km[summer_t_index]) - ma.array(merra_speed[summer_t_index, 0, :, :], mask=mask_of_merra_2p5km[summer_t_index]), axis = 0)
speed_shear_merra_summer_4km_1p5km = ma.mean(ma.array(merra_speed[summer_t_index, 2, :, :], mask=mask_of_merra_1p5km[summer_t_index]) - ma.array(merra_speed[summer_t_index, 0, :, :], mask=mask_of_merra_4km[summer_t_index]), axis = 0)
speed_shear_merra_summer_4km_2p5km = ma.mean(ma.array(merra_speed[summer_t_index, 2, :, :], mask=mask_of_merra_2p5km[summer_t_index]) - ma.array(merra_speed[summer_t_index, 1, :, :], mask=mask_of_merra_4km[summer_t_index]), axis = 0)
speed_shear_misr_summer_2p5km_1p5km = ma.mean(ma.array(misr_speed[summer_t_index, 1, :, :], mask=mask_of_misr_1p5km[summer_t_index]) - ma.array(misr_speed[summer_t_index, 0, :, :], mask=mask_of_misr_2p5km[summer_t_index]), axis = 0)
speed_shear_misr_summer_4km_1p5km = ma.mean(ma.array(misr_speed[summer_t_index, 2, :, :], mask=mask_of_misr_1p5km[summer_t_index]) - ma.array(misr_speed[summer_t_index, 0, :, :], mask=mask_of_misr_4km[summer_t_index]), axis = 0)
speed_shear_misr_summer_4km_2p5km = ma.mean(ma.array(misr_speed[summer_t_index, 2, :, :], mask=mask_of_misr_2p5km[summer_t_index]) - ma.array(misr_speed[summer_t_index, 1, :, :], mask=mask_of_misr_4km[summer_t_index]), axis = 0)
speed_shear_diff_summer_2p5km_1p5km = speed_shear_merra_summer_2p5km_1p5km - speed_shear_misr_summer_2p5km_1p5km
speed_shear_diff_summer_4km_1p5km = speed_shear_merra_summer_4km_1p5km - speed_shear_misr_summer_4km_1p5km
speed_shear_diff_summer_4km_2p5km = speed_shear_merra_summer_4km_2p5km - speed_shear_misr_summer_4km_2p5km

#Fall
speed_shear_merra_fall_2p5km_1p5km = ma.mean(ma.array(merra_speed[fall_t_index, 1, :, :], mask=mask_of_merra_1p5km[fall_t_index]) - ma.array(merra_speed[fall_t_index, 0, :, :], mask=mask_of_merra_2p5km[fall_t_index]), axis = 0)
speed_shear_merra_fall_4km_1p5km = ma.mean(ma.array(merra_speed[fall_t_index, 2, :, :], mask=mask_of_merra_1p5km[fall_t_index]) - ma.array(merra_speed[fall_t_index, 0, :, :], mask=mask_of_merra_4km[fall_t_index]), axis = 0)
speed_shear_merra_fall_4km_2p5km = ma.mean(ma.array(merra_speed[fall_t_index, 2, :, :], mask=mask_of_merra_2p5km[fall_t_index]) - ma.array(merra_speed[fall_t_index, 1, :, :], mask=mask_of_merra_4km[fall_t_index]), axis = 0)
speed_shear_misr_fall_2p5km_1p5km = ma.mean(ma.array(misr_speed[fall_t_index, 1, :, :], mask=mask_of_misr_1p5km[fall_t_index]) - ma.array(misr_speed[fall_t_index, 0, :, :], mask=mask_of_misr_2p5km[fall_t_index]), axis = 0)
speed_shear_misr_fall_4km_1p5km = ma.mean(ma.array(misr_speed[fall_t_index, 2, :, :], mask=mask_of_misr_1p5km[fall_t_index]) - ma.array(misr_speed[fall_t_index, 0, :, :], mask=mask_of_misr_4km[fall_t_index]), axis = 0)
speed_shear_misr_fall_4km_2p5km = ma.mean(ma.array(misr_speed[fall_t_index, 2, :, :], mask=mask_of_misr_2p5km[fall_t_index]) - ma.array(misr_speed[fall_t_index, 1, :, :], mask=mask_of_misr_4km[fall_t_index]), axis = 0)
speed_shear_diff_fall_2p5km_1p5km = speed_shear_merra_fall_2p5km_1p5km - speed_shear_misr_fall_2p5km_1p5km
speed_shear_diff_fall_4km_1p5km = speed_shear_merra_fall_4km_1p5km - speed_shear_misr_fall_4km_1p5km
speed_shear_diff_fall_4km_2p5km = speed_shear_merra_fall_4km_2p5km - speed_shear_misr_fall_4km_2p5km

#Overall
speed_shear_merra_overall_2p5km_1p5km = ma.mean(ma.array(merra_speed[:, 1, :, :], mask=mask_of_merra_1p5km[:]) - ma.array(merra_speed[:, 0, :, :], mask=mask_of_merra_2p5km), axis = 0)
speed_shear_merra_overall_4km_1p5km = ma.mean(ma.array(merra_speed[:, 2, :, :], mask=mask_of_merra_1p5km[:]) - ma.array(merra_speed[:, 0, :, :], mask=mask_of_merra_4km), axis = 0)
speed_shear_merra_overall_4km_2p5km = ma.mean(ma.array(merra_speed[:, 2, :, :], mask=mask_of_merra_2p5km[:]) - ma.array(merra_speed[:, 1, :, :], mask=mask_of_merra_4km), axis = 0)
speed_shear_misr_overall_2p5km_1p5km = ma.mean(ma.array(misr_speed[:, 1, :, :], mask=mask_of_misr_1p5km[:]) - ma.array(misr_speed[:, 0, :, :], mask=mask_of_misr_2p5km), axis = 0)
speed_shear_misr_overall_4km_1p5km = ma.mean(ma.array(misr_speed[:, 2, :, :], mask=mask_of_misr_1p5km[:]) - ma.array(misr_speed[:, 0, :, :], mask=mask_of_misr_4km[:]), axis = 0)
speed_shear_misr_overall_4km_2p5km = ma.mean(ma.array(misr_speed[:, 2, :, :], mask=mask_of_misr_2p5km[:]) - ma.array(misr_speed[:, 1, :, :], mask=mask_of_misr_4km[:]), axis = 0)
speed_shear_diff_overall_2p5km_1p5km = speed_shear_merra_overall_2p5km_1p5km - speed_shear_misr_overall_2p5km_1p5km
speed_shear_diff_overall_4km_1p5km = speed_shear_merra_overall_4km_1p5km - speed_shear_misr_overall_4km_1p5km
speed_shear_diff_overall_4km_2p5km = speed_shear_merra_overall_4km_2p5km - speed_shear_misr_overall_4km_2p5km


############# WIND VECTOR STUFF ############# 

##MERRA wind vectors
u_winter_merra_1p5km = ma.mean(merra_u[winter_t_index, 0, :], axis=0) 
v_winter_merra_1p5km = ma.mean(merra_v[winter_t_index, 0, :], axis=0)
u_winter_merra_2p5km = ma.mean(merra_u[winter_t_index, 1, :], axis=0) 
v_winter_merra_2p5km = ma.mean(merra_v[winter_t_index, 1, :], axis=0)
u_winter_merra_4km = ma.mean(merra_u[winter_t_index, 2, :], axis=0) 
v_winter_merra_4km = ma.mean(merra_v[winter_t_index, 2, :], axis=0)

u_spring_merra_1p5km = ma.mean(merra_u[spring_t_index, 0, :], axis=0) 
v_spring_merra_1p5km = ma.mean(merra_v[spring_t_index, 0, :], axis=0)
u_spring_merra_2p5km = ma.mean(merra_u[spring_t_index, 1, :], axis=0) 
v_spring_merra_2p5km = ma.mean(merra_v[spring_t_index, 1, :], axis=0)
u_spring_merra_4km = ma.mean(merra_u[spring_t_index, 2, :], axis=0) 
v_spring_merra_4km = ma.mean(merra_v[spring_t_index, 2, :], axis=0)

u_summer_merra_1p5km = ma.mean(merra_u[summer_t_index, 0, :], axis=0) 
v_summer_merra_1p5km = ma.mean(merra_v[summer_t_index, 0, :], axis=0) 
u_summer_merra_2p5km = ma.mean(merra_u[summer_t_index, 1, :], axis=0) 
v_summer_merra_2p5km = ma.mean(merra_v[summer_t_index, 1, :], axis=0)
u_summer_merra_4km = ma.mean(merra_u[summer_t_index, 2, :], axis=0) 
v_summer_merra_4km = ma.mean(merra_v[summer_t_index, 2, :], axis=0)

u_fall_merra_1p5km = ma.mean(merra_u[fall_t_index, 0, :], axis=0) 
v_fall_merra_1p5km = ma.mean(merra_v[fall_t_index, 0, :], axis=0)
u_fall_merra_2p5km = ma.mean(merra_u[fall_t_index, 1, :], axis=0) 
v_fall_merra_2p5km = ma.mean(merra_v[fall_t_index, 1, :], axis=0)
u_fall_merra_4km = ma.mean(merra_u[fall_t_index, 2, :], axis=0) 
v_fall_merra_4km = ma.mean(merra_v[fall_t_index, 2, :], axis=0)

u_overall_merra_1p5km = ma.mean(merra_u[:, 0, :], axis=0) 
v_overall_merra_1p5km = ma.mean(merra_v[:, 0, :], axis=0) 
u_overall_merra_2p5km = ma.mean(merra_u[:, 1, :], axis=0) 
v_overall_merra_2p5km = ma.mean(merra_v[:, 1, :], axis=0)
u_overall_merra_4km = ma.mean(merra_u[:, 2, :], axis=0) 
v_overall_merra_4km = ma.mean(merra_v[:, 2, :], axis=0)

##MISR wind vectors
u_winter_misr_1p5km = ma.mean(misr_u[winter_t_index, 0, :], axis=0) 
v_winter_misr_1p5km = ma.mean(misr_v[winter_t_index, 0, :], axis=0)
u_winter_misr_2p5km = ma.mean(misr_u[winter_t_index, 1, :], axis=0) 
v_winter_misr_2p5km = ma.mean(misr_v[winter_t_index, 1, :], axis=0)
u_winter_misr_4km = ma.mean(misr_u[winter_t_index, 2, :], axis=0) 
v_winter_misr_4km = ma.mean(misr_v[winter_t_index, 2, :], axis=0)

u_spring_misr_1p5km = ma.mean(misr_u[spring_t_index, 0, :], axis=0) 
v_spring_misr_1p5km = ma.mean(misr_v[spring_t_index, 0, :], axis=0)
u_spring_misr_2p5km = ma.mean(misr_u[spring_t_index, 1, :], axis=0) 
v_spring_misr_2p5km = ma.mean(misr_v[spring_t_index, 1, :], axis=0)
u_spring_misr_4km = ma.mean(misr_u[spring_t_index, 2, :], axis=0) 
v_spring_misr_4km = ma.mean(misr_v[spring_t_index, 2, :], axis=0)

u_summer_misr_1p5km = ma.mean(misr_u[summer_t_index, 0, :], axis=0) 
v_summer_misr_1p5km = ma.mean(misr_v[summer_t_index, 0, :], axis=0) 
u_summer_misr_2p5km = ma.mean(misr_u[summer_t_index, 1, :], axis=0) 
v_summer_misr_2p5km = ma.mean(misr_v[summer_t_index, 1, :], axis=0)
u_summer_misr_4km = ma.mean(misr_u[summer_t_index, 2, :], axis=0) 
v_summer_misr_4km = ma.mean(misr_v[summer_t_index, 2, :], axis=0)

u_fall_misr_1p5km = ma.mean(misr_u[fall_t_index, 0, :], axis=0) 
v_fall_misr_1p5km = ma.mean(misr_v[fall_t_index, 0, :], axis=0)
u_fall_misr_2p5km = ma.mean(misr_u[fall_t_index, 1, :], axis=0) 
v_fall_misr_2p5km = ma.mean(misr_v[fall_t_index, 1, :], axis=0)
u_fall_misr_4km = ma.mean(misr_u[fall_t_index, 2, :], axis=0) 
v_fall_misr_4km = ma.mean(misr_v[fall_t_index, 2, :], axis=0)

u_overall_misr_1p5km = ma.mean(misr_u[:, 0, :], axis=0) 
v_overall_misr_1p5km = ma.mean(misr_v[:, 0, :], axis=0) 
u_overall_misr_2p5km = ma.mean(misr_u[:, 1, :], axis=0) 
v_overall_misr_2p5km = ma.mean(misr_v[:, 1, :], axis=0)
u_overall_misr_4km = ma.mean(misr_u[:, 2, :], axis=0) 
v_overall_misr_4km = ma.mean(misr_v[:, 2, :], axis=0)

##Differences in the vectors
##NOTE: Not bothering with masks here - if have more time then maybe, but assuming there's enough data that comparing means is OK

u_winter_diff_1p5km = u_winter_merra_1p5km - u_winter_misr_1p5km
v_winter_diff_1p5km = v_winter_merra_1p5km - v_winter_misr_1p5km
u_winter_diff_2p5km = u_winter_merra_2p5km - u_winter_misr_2p5km
v_winter_diff_2p5km = v_winter_merra_2p5km - v_winter_misr_2p5km
u_winter_diff_4km = u_winter_merra_1p5km - u_winter_misr_1p5km
v_winter_diff_4km = v_winter_merra_4km - v_winter_misr_4km

u_spring_diff_1p5km = u_spring_merra_1p5km - u_spring_misr_1p5km
v_spring_diff_1p5km = v_spring_merra_1p5km - v_spring_misr_1p5km
u_spring_diff_2p5km = u_spring_merra_2p5km - u_spring_misr_2p5km
v_spring_diff_2p5km = v_spring_merra_2p5km - v_spring_misr_2p5km
u_spring_diff_4km = u_spring_merra_1p5km - u_spring_misr_1p5km
v_spring_diff_4km = v_spring_merra_4km - v_spring_misr_4km

u_summer_diff_1p5km = u_summer_merra_1p5km - u_summer_misr_1p5km
v_summer_diff_1p5km = v_summer_merra_1p5km - v_summer_misr_1p5km
u_summer_diff_2p5km = u_summer_merra_2p5km - u_summer_misr_2p5km
v_summer_diff_2p5km = v_summer_merra_2p5km - v_summer_misr_2p5km
u_summer_diff_4km = u_summer_merra_1p5km - u_summer_misr_1p5km
v_summer_diff_4km = v_summer_merra_4km - v_summer_misr_4km

u_fall_diff_1p5km = u_fall_merra_1p5km - u_fall_misr_1p5km
v_fall_diff_1p5km = v_fall_merra_1p5km - v_fall_misr_1p5km
u_fall_diff_2p5km = u_fall_merra_2p5km - u_fall_misr_2p5km
v_fall_diff_2p5km = v_fall_merra_2p5km - v_fall_misr_2p5km
u_fall_diff_4km = u_fall_merra_1p5km - u_fall_misr_1p5km
v_fall_diff_4km = v_fall_merra_4km - v_fall_misr_4km

u_overall_diff_1p5km = u_overall_merra_1p5km - u_overall_misr_1p5km
v_overall_diff_1p5km = v_overall_merra_1p5km - v_overall_misr_1p5km
u_overall_diff_2p5km = u_overall_merra_2p5km - u_overall_misr_2p5km
v_overall_diff_2p5km = v_overall_merra_2p5km - v_overall_misr_2p5km
u_overall_diff_4km = u_overall_merra_1p5km - u_overall_misr_1p5km
v_overall_diff_4km = v_overall_merra_4km - v_overall_misr_4km

##Wind vector shears

#MERRA shears
u_shear_merra_winter_2p5km_1p5km = u_winter_merra_2p5km - u_winter_merra_1p5km
v_shear_merra_winter_2p5km_1p5km = v_winter_merra_2p5km - v_winter_merra_1p5km
u_shear_merra_winter_4km_1p5km = u_winter_merra_4km - u_winter_merra_1p5km
v_shear_merra_winter_4km_1p5km = v_winter_merra_4km - v_winter_merra_1p5km
u_shear_merra_winter_4km_2p5km = u_winter_merra_4km - u_winter_merra_2p5km
v_shear_merra_winter_4km_2p5km = v_winter_merra_4km - v_winter_merra_2p5km

u_shear_merra_spring_2p5km_1p5km = u_spring_merra_2p5km - u_spring_merra_1p5km
v_shear_merra_spring_2p5km_1p5km = v_spring_merra_2p5km - v_spring_merra_1p5km
u_shear_merra_spring_4km_1p5km = u_spring_merra_4km - u_spring_merra_1p5km
v_shear_merra_spring_4km_1p5km = v_spring_merra_4km - v_spring_merra_1p5km
u_shear_merra_spring_4km_2p5km = u_spring_merra_4km - u_spring_merra_2p5km
v_shear_merra_spring_4km_2p5km = v_spring_merra_4km - v_spring_merra_2p5km

u_shear_merra_summer_2p5km_1p5km = u_summer_merra_2p5km - u_summer_merra_1p5km
v_shear_merra_summer_2p5km_1p5km = v_summer_merra_2p5km - v_summer_merra_1p5km
u_shear_merra_summer_4km_1p5km = u_summer_merra_4km - u_summer_merra_1p5km
v_shear_merra_summer_4km_1p5km = v_summer_merra_4km - v_summer_merra_1p5km
u_shear_merra_summer_4km_2p5km = u_summer_merra_4km - u_summer_merra_2p5km
v_shear_merra_summer_4km_2p5km = v_summer_merra_4km - v_summer_merra_2p5km

u_shear_merra_fall_2p5km_1p5km = u_fall_merra_2p5km - u_fall_merra_1p5km
v_shear_merra_fall_2p5km_1p5km = v_fall_merra_2p5km - v_fall_merra_1p5km
u_shear_merra_fall_4km_1p5km = u_fall_merra_4km - u_fall_merra_1p5km
v_shear_merra_fall_4km_1p5km = v_fall_merra_4km - v_fall_merra_1p5km
u_shear_merra_fall_4km_2p5km = u_fall_merra_4km - u_fall_merra_2p5km
v_shear_merra_fall_4km_2p5km = v_fall_merra_4km - v_fall_merra_2p5km

u_shear_merra_overall_2p5km_1p5km = u_overall_merra_2p5km - u_overall_merra_1p5km
v_shear_merra_overall_2p5km_1p5km = v_overall_merra_2p5km - v_overall_merra_1p5km
u_shear_merra_overall_4km_1p5km = u_overall_merra_4km - u_overall_merra_1p5km
v_shear_merra_overall_4km_1p5km = v_overall_merra_4km - v_overall_merra_1p5km
u_shear_merra_overall_4km_2p5km = u_overall_merra_4km - u_overall_merra_2p5km
v_shear_merra_overall_4km_2p5km = v_overall_merra_4km - v_overall_merra_2p5km

#MISR shears
u_shear_misr_winter_2p5km_1p5km = u_winter_misr_2p5km - u_winter_misr_1p5km
v_shear_misr_winter_2p5km_1p5km = v_winter_misr_2p5km - v_winter_misr_1p5km
u_shear_misr_winter_4km_1p5km = u_winter_misr_4km - u_winter_misr_1p5km
v_shear_misr_winter_4km_1p5km = v_winter_misr_4km - v_winter_misr_1p5km
u_shear_misr_winter_4km_2p5km = u_winter_misr_4km - u_winter_misr_2p5km
v_shear_misr_winter_4km_2p5km = v_winter_misr_4km - v_winter_misr_2p5km

u_shear_misr_spring_2p5km_1p5km = u_spring_misr_2p5km - u_spring_misr_1p5km
v_shear_misr_spring_2p5km_1p5km = v_spring_misr_2p5km - v_spring_misr_1p5km
u_shear_misr_spring_4km_1p5km = u_spring_misr_4km - u_spring_misr_1p5km
v_shear_misr_spring_4km_1p5km = v_spring_misr_4km - v_spring_misr_1p5km
u_shear_misr_spring_4km_2p5km = u_spring_misr_4km - u_spring_misr_2p5km
v_shear_misr_spring_4km_2p5km = v_spring_misr_4km - v_spring_misr_2p5km

u_shear_misr_summer_2p5km_1p5km = u_summer_misr_2p5km - u_summer_misr_1p5km
v_shear_misr_summer_2p5km_1p5km = v_summer_misr_2p5km - v_summer_misr_1p5km
u_shear_misr_summer_4km_1p5km = u_summer_misr_4km - u_summer_misr_1p5km
v_shear_misr_summer_4km_1p5km = v_summer_misr_4km - v_summer_misr_1p5km
u_shear_misr_summer_4km_2p5km = u_summer_misr_4km - u_summer_misr_2p5km
v_shear_misr_summer_4km_2p5km = v_summer_misr_4km - v_summer_misr_2p5km

u_shear_misr_fall_2p5km_1p5km = u_fall_misr_2p5km - u_fall_misr_1p5km
v_shear_misr_fall_2p5km_1p5km = v_fall_misr_2p5km - v_fall_misr_1p5km
u_shear_misr_fall_4km_1p5km = u_fall_misr_4km - u_fall_misr_1p5km
v_shear_misr_fall_4km_1p5km = v_fall_misr_4km - v_fall_misr_1p5km
u_shear_misr_fall_4km_2p5km = u_fall_misr_4km - u_fall_misr_2p5km
v_shear_misr_fall_4km_2p5km = v_fall_misr_4km - v_fall_misr_2p5km

u_shear_misr_overall_2p5km_1p5km = u_overall_misr_2p5km - u_overall_misr_1p5km
v_shear_misr_overall_2p5km_1p5km = v_overall_misr_2p5km - v_overall_misr_1p5km
u_shear_misr_overall_4km_1p5km = u_overall_misr_4km - u_overall_misr_1p5km
v_shear_misr_overall_4km_1p5km = v_overall_misr_4km - v_overall_misr_1p5km
u_shear_misr_overall_4km_2p5km = u_overall_misr_4km - u_overall_misr_2p5km
v_shear_misr_overall_4km_2p5km = v_overall_misr_4km - v_overall_misr_2p5km

#Difference shears
u_shear_diff_winter_2p5km_1p5km = u_winter_diff_2p5km - u_winter_diff_1p5km
v_shear_diff_winter_2p5km_1p5km = v_winter_diff_2p5km - v_winter_diff_1p5km
u_shear_diff_winter_4km_1p5km = u_winter_diff_4km - u_winter_diff_1p5km
v_shear_diff_winter_4km_1p5km = v_winter_diff_4km - v_winter_diff_1p5km
u_shear_diff_winter_4km_2p5km = u_winter_diff_4km - u_winter_diff_2p5km
v_shear_diff_winter_4km_2p5km = v_winter_diff_4km - v_winter_diff_2p5km

u_shear_diff_spring_2p5km_1p5km = u_spring_diff_2p5km - u_spring_diff_1p5km
v_shear_diff_spring_2p5km_1p5km = v_spring_diff_2p5km - v_spring_diff_1p5km
u_shear_diff_spring_4km_1p5km = u_spring_diff_4km - u_spring_diff_1p5km
v_shear_diff_spring_4km_1p5km = v_spring_diff_4km - v_spring_diff_1p5km
u_shear_diff_spring_4km_2p5km = u_spring_diff_4km - u_spring_diff_2p5km
v_shear_diff_spring_4km_2p5km = v_spring_diff_4km - v_spring_diff_2p5km

u_shear_diff_summer_2p5km_1p5km = u_summer_diff_2p5km - u_summer_diff_1p5km
v_shear_diff_summer_2p5km_1p5km = v_summer_diff_2p5km - v_summer_diff_1p5km
u_shear_diff_summer_4km_1p5km = u_summer_diff_4km - u_summer_diff_1p5km
v_shear_diff_summer_4km_1p5km = v_summer_diff_4km - v_summer_diff_1p5km
u_shear_diff_summer_4km_2p5km = u_summer_diff_4km - u_summer_diff_2p5km
v_shear_diff_summer_4km_2p5km = v_summer_diff_4km - v_summer_diff_2p5km

u_shear_diff_fall_2p5km_1p5km = u_fall_diff_2p5km - u_fall_diff_1p5km
v_shear_diff_fall_2p5km_1p5km = v_fall_diff_2p5km - v_fall_diff_1p5km
u_shear_diff_fall_4km_1p5km = u_fall_diff_4km - u_fall_diff_1p5km
v_shear_diff_fall_4km_1p5km = v_fall_diff_4km - v_fall_diff_1p5km
u_shear_diff_fall_4km_2p5km = u_fall_diff_4km - u_fall_diff_2p5km
v_shear_diff_fall_4km_2p5km = v_fall_diff_4km - v_fall_diff_2p5km

u_shear_diff_overall_2p5km_1p5km = u_overall_diff_2p5km - u_overall_diff_1p5km
v_shear_diff_overall_2p5km_1p5km = v_overall_diff_2p5km - v_overall_diff_1p5km
u_shear_diff_overall_4km_1p5km = u_overall_diff_4km - u_overall_diff_1p5km
v_shear_diff_overall_4km_1p5km = v_overall_diff_4km - v_overall_diff_1p5km
u_shear_diff_overall_4km_2p5km = u_overall_diff_4km - u_overall_diff_2p5km
v_shear_diff_overall_4km_2p5km = v_overall_diff_4km - v_overall_diff_2p5km


#####################################################################################   PLOTS   #####################################################################################


############# Contour Plots ############# 

###Plot wind speed differences, land masked out###

#Winter
plot_global_contour_map_maskedland(speed_diff_winter_1p5km, merra_lon, merra_lat, np.arange(21)*0.5-5, 'MERRA-MISR_winter_1p5km', 'MERRA2 - MISR (mean), winter, 1.5 km', cmap=cm.RdBu_r)
plot_global_contour_map_maskedland(speed_diff_winter_2p5km, merra_lon, merra_lat, np.arange(21)*0.5-5, 'MERRA-MISR_winter_2p5km', 'MERRA2 - MISR (mean), winter, 2.5 km', cmap=cm.RdBu_r)
plot_global_contour_map_maskedland(speed_diff_winter_4km, merra_lon, merra_lat, np.arange(21)*0.5-5, 'MERRA-MISR_winter_4km', 'MERRA2 - MISR (mean), winter, 4 km', cmap=cm.RdBu_r)

#Spring
plot_global_contour_map_maskedland(speed_diff_spring_1p5km, merra_lon, merra_lat, np.arange(21)*0.5-5, 'MERRA-MISR_spring_1p5km', 'MERRA2 - MISR (mean), spring, 1.5 km', cmap=cm.RdBu_r)
plot_global_contour_map_maskedland(speed_diff_spring_2p5km, merra_lon, merra_lat, np.arange(21)*0.5-5, 'MERRA-MISR_spring_2p5km', 'MERRA2 - MISR (mean), spring, 2.5 km', cmap=cm.RdBu_r)
plot_global_contour_map_maskedland(speed_diff_spring_4km, merra_lon, merra_lat, np.arange(21)*0.5-5, 'MERRA-MISR_spring_4km', 'MERRA2 - MISR (mean), spring, 4 km', cmap=cm.RdBu_r)

#Summer
plot_global_contour_map_maskedland(speed_diff_summer_1p5km, merra_lon, merra_lat, np.arange(21)*0.5-5, 'MERRA-MISR_summer_1p5km', 'MERRA2 - MISR (mean), summer, 1.5 km', cmap=cm.RdBu_r)
plot_global_contour_map_maskedland(speed_diff_summer_2p5km, merra_lon, merra_lat, np.arange(21)*0.5-5, 'MERRA-MISR_summer_2p5km', 'MERRA2 - MISR (mean), summer, 2.5 km', cmap=cm.RdBu_r)
plot_global_contour_map_maskedland(speed_diff_summer_4km, merra_lon, merra_lat, np.arange(21)*0.5-5, 'MERRA-MISR_summer_4km', 'MERRA2 - MISR (mean), summer, 4 km', cmap=cm.RdBu_r)

#Fall
plot_global_contour_map_maskedland(speed_diff_fall_1p5km, merra_lon, merra_lat, np.arange(21)*0.5-5, 'MERRA-MISR_fall_1p5km', 'MERRA2 - MISR (mean), fall, 1.5 km', cmap=cm.RdBu_r)
plot_global_contour_map_maskedland(speed_diff_fall_2p5km, merra_lon, merra_lat, np.arange(21)*0.5-5, 'MERRA-MISR_fall_2p5km', 'MERRA2 - MISR (mean), fall, 2.5 km', cmap=cm.RdBu_r)
plot_global_contour_map_maskedland(speed_diff_fall_4km, merra_lon, merra_lat, np.arange(21)*0.5-5, 'MERRA-MISR_fall_4km', 'MERRA2 - MISR (mean), fall, 4 km', cmap=cm.RdBu_r)

#Overall
plot_global_contour_map_maskedland(speed_diff_overall_1p5km, merra_lon, merra_lat, np.arange(21)*0.5-5, 'MERRA-MISR_overall_1p5km', 'MERRA2 - MISR (mean), overall, 1.5 km', cmap=cm.RdBu_r)
plot_global_contour_map_maskedland(speed_diff_overall_2p5km, merra_lon, merra_lat, np.arange(21)*0.5-5, 'MERRA-MISR_overall_2p5km', 'MERRA2 - MISR (mean), overall, 2.5 km', cmap=cm.RdBu_r)
plot_global_contour_map_maskedland(speed_diff_overall_4km, merra_lon, merra_lat, np.arange(21)*0.5-5, 'MERRA-MISR_overall_4km', 'MERRA2 - MISR (mean), overall, 4 km', cmap=cm.RdBu_r)


###Plot wind shears, land masked out###

##Winter
#MERRA
plot_global_contour_map_maskedland(speed_shear_merra_winter_2p5km_1p5km, merra_lon, merra_lat, np.arange(21)*0.5-5, 'MERRA_shear_winter_2p5km_1p5km', "Shear of MERRA winds, winter, 2.5km - 1.5km", cmap=cm.RdBu_r)
plot_global_contour_map_maskedland(speed_shear_merra_winter_4km_1p5km, merra_lon, merra_lat, np.arange(21)*0.5-5, 'MERRA_shear_winter_4km_1p5km', "Shear of MERRA winds, winter, 4km - 1.5km", cmap=cm.RdBu_r)
plot_global_contour_map_maskedland(speed_shear_merra_winter_4km_2p5km, merra_lon, merra_lat, np.arange(21)*0.5-5, 'MERRA_shear_winter_4km_2p5km', "Shear of MERRA winds, winter, 4km - 2.5km", cmap=cm.RdBu_r)
#MISR
plot_global_contour_map_maskedland(speed_shear_misr_winter_2p5km_1p5km, misr_lon, misr_lat, np.arange(21)*0.5-5, 'MISR_shear_winter_2p5km_1p5km', "Shear of MISR winds, winter, 2.5km - 1.5km", cmap=cm.RdBu_r)
plot_global_contour_map_maskedland(speed_shear_misr_winter_4km_1p5km, misr_lon, misr_lat, np.arange(21)*0.5-5, 'MISR_shear_winter_4km_1p5km', "Shear of MISR winds, winter, 4km - 1.5km", cmap=cm.RdBu_r)
plot_global_contour_map_maskedland(speed_shear_misr_winter_4km_2p5km, misr_lon, misr_lat, np.arange(21)*0.5-5, 'MISR_shear_winter_4km_2p5km', "Shear of MISR winds, winter, 4km - 2.5km", cmap=cm.RdBu_r)
#MERRA-MISR difference
plot_global_contour_map_maskedland(speed_shear_diff_winter_2p5km_1p5km, merra_lon, merra_lat, np.arange(21)*0.5-5, 'diff_shear_winter_2p5km_1p5km', "MERRA shear - MISR shear, winter, 2.5km - 1.5km", cmap=cm.RdBu_r)
plot_global_contour_map_maskedland(speed_shear_diff_winter_4km_1p5km, merra_lon, merra_lat, np.arange(21)*0.5-5, 'diff_shear_winter_4km_1p5km', "MERRA shear - MISR shear, winter, 4km - 1.5km", cmap=cm.RdBu_r)
plot_global_contour_map_maskedland(speed_shear_diff_winter_4km_2p5km, merra_lon, merra_lat, np.arange(21)*0.5-5, 'diff_shear_winter_4km_2p5km', "MERRA shear - MISR shear, winter, 4km - 2.5km", cmap=cm.RdBu_r)

##Spring
#MERRA
plot_global_contour_map_maskedland(speed_shear_merra_spring_2p5km_1p5km, merra_lon, merra_lat, np.arange(21)*0.5-5, 'MERRA_shear_spring_2p5km_1p5km', "Shear of MERRA winds, spring, 2.5km - 1.5km", cmap=cm.RdBu_r)
plot_global_contour_map_maskedland(speed_shear_merra_spring_4km_1p5km, merra_lon, merra_lat, np.arange(21)*0.5-5, 'MERRA_shear_spring_4km_1p5km', "Shear of MERRA winds, spring, 4km - 1.5km", cmap=cm.RdBu_r)
plot_global_contour_map_maskedland(speed_shear_merra_spring_4km_2p5km, merra_lon, merra_lat, np.arange(21)*0.5-5, 'MERRA_shear_spring_4km_2p5km', "Shear of MERRA winds, spring, 4km - 2.5km", cmap=cm.RdBu_r)
#MISR
plot_global_contour_map_maskedland(speed_shear_misr_spring_2p5km_1p5km, misr_lon, misr_lat, np.arange(21)*0.5-5, 'MISR_shear_spring_2p5km_1p5km', "Shear of MISR winds, spring, 2.5km - 1.5km", cmap=cm.RdBu_r)
plot_global_contour_map_maskedland(speed_shear_misr_spring_4km_1p5km, misr_lon, misr_lat, np.arange(21)*0.5-5, 'MISR_shear_spring_4km_1p5km', "Shear of MISR winds, spring, 4km - 1.5km", cmap=cm.RdBu_r)
plot_global_contour_map_maskedland(speed_shear_misr_spring_4km_2p5km, misr_lon, misr_lat, np.arange(21)*0.5-5, 'MISR_shear_spring_4km_2p5km', "Shear of MISR winds, spring, 4km - 2.5km", cmap=cm.RdBu_r)
#MERRA-MISR difference
plot_global_contour_map_maskedland(speed_shear_diff_spring_2p5km_1p5km, merra_lon, merra_lat, np.arange(21)*0.5-5, 'diff_shear_spring_2p5km_1p5km', "MERRA shear - MISR shear, spring, 2.5km - 1.5km", cmap=cm.RdBu_r)
plot_global_contour_map_maskedland(speed_shear_diff_spring_4km_1p5km, merra_lon, merra_lat, np.arange(21)*0.5-5, 'diff_shear_spring_4km_1p5km', "MERRA shear - MISR shear, spring, 4km - 1.5km", cmap=cm.RdBu_r)
plot_global_contour_map_maskedland(speed_shear_diff_spring_4km_2p5km, merra_lon, merra_lat, np.arange(21)*0.5-5, 'diff_shear_spring_4km_2p5km', "MERRA shear - MISR shear, spring, 4km - 2.5km", cmap=cm.RdBu_r)

##Summer
#MERRA
plot_global_contour_map_maskedland(speed_shear_merra_summer_2p5km_1p5km, merra_lon, merra_lat, np.arange(21)*0.5-5, 'MERRA_shear_summer_2p5km_1p5km', "Shear of MERRA winds, summer, 2.5km - 1.5km", cmap=cm.RdBu_r)
plot_global_contour_map_maskedland(speed_shear_merra_summer_4km_1p5km, merra_lon, merra_lat, np.arange(21)*0.5-5, 'MERRA_shear_summer_4km_1p5km', "Shear of MERRA winds, summer, 4km - 1.5km", cmap=cm.RdBu_r)
plot_global_contour_map_maskedland(speed_shear_merra_summer_4km_2p5km, merra_lon, merra_lat, np.arange(21)*0.5-5, 'MERRA_shear_summer_4km_2p5km', "Shear of MERRA winds, summer, 4km - 2.5km", cmap=cm.RdBu_r)
#MISR
plot_global_contour_map_maskedland(speed_shear_misr_summer_2p5km_1p5km, misr_lon, misr_lat, np.arange(21)*0.5-5, 'MISR_shear_summer_2p5km_1p5km', "Shear of MISR winds, summer, 2.5km - 1.5km", cmap=cm.RdBu_r)
plot_global_contour_map_maskedland(speed_shear_misr_summer_4km_1p5km, misr_lon, misr_lat, np.arange(21)*0.5-5, 'MISR_shear_summer_4km_1p5km', "Shear of MISR winds, summer, 4km - 1.5km", cmap=cm.RdBu_r)
plot_global_contour_map_maskedland(speed_shear_misr_summer_4km_2p5km, misr_lon, misr_lat, np.arange(21)*0.5-5, 'MISR_shear_summer_4km_2p5km', "Shear of MISR winds, summer, 4km - 2.5km", cmap=cm.RdBu_r)
#MERRA-MISR difference
plot_global_contour_map_maskedland(speed_shear_diff_summer_2p5km_1p5km, merra_lon, merra_lat, np.arange(21)*0.5-5, 'diff_shear_summer_2p5km_1p5km', "MERRA shear - MISR shear, summer, 2.5km - 1.5km", cmap=cm.RdBu_r)
plot_global_contour_map_maskedland(speed_shear_diff_summer_4km_1p5km, merra_lon, merra_lat, np.arange(21)*0.5-5, 'diff_shear_summer_4km_1p5km', "MERRA shear - MISR shear, summer, 4km - 1.5km", cmap=cm.RdBu_r)
plot_global_contour_map_maskedland(speed_shear_diff_summer_4km_2p5km, merra_lon, merra_lat, np.arange(21)*0.5-5, 'diff_shear_summer_4km_2p5km', "MERRA shear - MISR shear, summer, 4km - 2.5km", cmap=cm.RdBu_r)

##Fall
#MERRA
plot_global_contour_map_maskedland(speed_shear_merra_fall_2p5km_1p5km, merra_lon, merra_lat, np.arange(21)*0.5-5, 'MERRA_shear_fall_2p5km_1p5km', "Shear of MERRA winds, fall, 2.5km - 1.5km", cmap=cm.RdBu_r)
plot_global_contour_map_maskedland(speed_shear_merra_fall_4km_1p5km, merra_lon, merra_lat, np.arange(21)*0.5-5, 'MERRA_shear_fall_4km_1p5km', "Shear of MERRA winds, fall, 4km - 1.5km", cmap=cm.RdBu_r)
plot_global_contour_map_maskedland(speed_shear_merra_fall_4km_2p5km, merra_lon, merra_lat, np.arange(21)*0.5-5, 'MERRA_shear_fall_4km_2p5km', "Shear of MERRA winds, fall, 4km - 2.5km", cmap=cm.RdBu_r)
#MISR
plot_global_contour_map_maskedland(speed_shear_misr_fall_2p5km_1p5km, misr_lon, misr_lat, np.arange(21)*0.5-5, 'MISR_shear_fall_2p5km_1p5km', "Shear of MISR winds, fall, 2.5km - 1.5km", cmap=cm.RdBu_r)
plot_global_contour_map_maskedland(speed_shear_misr_fall_4km_1p5km, misr_lon, misr_lat, np.arange(21)*0.5-5, 'MISR_shear_fall_4km_1p5km', "Shear of MISR winds, fall, 4km - 1.5km", cmap=cm.RdBu_r)
plot_global_contour_map_maskedland(speed_shear_misr_fall_4km_2p5km, misr_lon, misr_lat, np.arange(21)*0.5-5, 'MISR_shear_fall_4km_2p5km', "Shear of MISR winds, fall, 4km - 2.5km", cmap=cm.RdBu_r)
#MERRA-MISR difference
plot_global_contour_map_maskedland(speed_shear_diff_fall_2p5km_1p5km, merra_lon, merra_lat, np.arange(21)*0.5-5, 'diff_shear_fall_2p5km_1p5km', "MERRA shear - MISR shear, fall, 2.5km - 1.5km", cmap=cm.RdBu_r)
plot_global_contour_map_maskedland(speed_shear_diff_fall_4km_1p5km, merra_lon, merra_lat, np.arange(21)*0.5-5, 'diff_shear_fall_4km_1p5km', "MERRA shear - MISR shear, fall, 4km - 1.5km", cmap=cm.RdBu_r)
plot_global_contour_map_maskedland(speed_shear_diff_fall_4km_2p5km, merra_lon, merra_lat, np.arange(21)*0.5-5, 'diff_shear_fall_4km_2p5km', "MERRA shear - MISR shear, fall, 4km - 2.5km", cmap=cm.RdBu_r)

##Overall
#MERRA
plot_global_contour_map_maskedland(speed_shear_merra_overall_2p5km_1p5km, merra_lon, merra_lat, np.arange(21)*0.5-5, 'MERRA_shear_overall_2p5km_1p5km', "Shear of MERRA winds, overall, 2.5km - 1.5km", cmap=cm.RdBu_r)
plot_global_contour_map_maskedland(speed_shear_merra_overall_4km_1p5km, merra_lon, merra_lat, np.arange(21)*0.5-5, 'MERRA_shear_overall_4km_1p5km', "Shear of MERRA winds, overall, 4km - 1.5km", cmap=cm.RdBu_r)
plot_global_contour_map_maskedland(speed_shear_merra_overall_4km_2p5km, merra_lon, merra_lat, np.arange(21)*0.5-5, 'MERRA_shear_overall_4km_2p5km', "Shear of MERRA winds, overall, 4km - 2.5km", cmap=cm.RdBu_r)
#MISR
plot_global_contour_map_maskedland(speed_shear_misr_overall_2p5km_1p5km, misr_lon, misr_lat, np.arange(21)*0.5-5, 'MISR_shear_overall_2p5km_1p5km', "Shear of MISR winds, overall, 2.5km - 1.5km", cmap=cm.RdBu_r)
plot_global_contour_map_maskedland(speed_shear_misr_overall_4km_1p5km, misr_lon, misr_lat, np.arange(21)*0.5-5, 'MISR_shear_overall_4km_1p5km', "Shear of MISR winds, overall, 4km - 1.5km", cmap=cm.RdBu_r)
plot_global_contour_map_maskedland(speed_shear_misr_overall_4km_2p5km, misr_lon, misr_lat, np.arange(21)*0.5-5, 'MISR_shear_overall_4km_2p5km', "Shear of MISR winds, overall, 4km - 2.5km", cmap=cm.RdBu_r)
#MERRA-MISR difference
plot_global_contour_map_maskedland(speed_shear_diff_overall_2p5km_1p5km, merra_lon, merra_lat, np.arange(21)*0.5-5, 'diff_shear_overall_2p5km_1p5km', "MERRA shear - MISR shear, overall, 2.5km - 1.5km", cmap=cm.RdBu_r)
plot_global_contour_map_maskedland(speed_shear_diff_overall_4km_1p5km, merra_lon, merra_lat, np.arange(21)*0.5-5, 'diff_shear_overall_4km_1p5km', "MERRA shear - MISR shear, overall, 4km - 1.5km", cmap=cm.RdBu_r)
plot_global_contour_map_maskedland(speed_shear_diff_overall_4km_2p5km, merra_lon, merra_lat, np.arange(21)*0.5-5, 'diff_shear_overall_4km_2p5km', "MERRA shear - MISR shear, overall, 4km - 2.5km", cmap=cm.RdBu_r)


############# Vector Plots ############# 

#Use zeros in place of u vector to get only v vector. Replace with u vector to get complete wind vector
#Occasionally will get "RuntimeWarning: invalid value encountered in sqrt" which corresponds with negative values in sqrt function... weird. Ignore for now

##Differences between MERRA and MISR

#Winter
plot_global_vector_map_maskedland(u_winter_diff_1p5km, v_winter_diff_1p5km, merra_lon, merra_lat, figure_file='MERRA-MISR winter wind vector difference 1p5km', title='MERRA-MISR winter wind vector difference 1.5 km')
plot_global_vector_map_maskedland(np.zeros(np.shape(u_winter_diff_1p5km)), v_winter_diff_1p5km, merra_lon, merra_lat, figure_file='MERRA-MISR winter wind vector v difference 1p5km', title='MERRA-MISR winter wind vector v difference, 1.5 km')
plot_global_vector_map_maskedland(u_winter_diff_2p5km, v_winter_diff_2p5km, merra_lon, merra_lat, figure_file='MERRA-MISR winter wind vector difference 2p5km', title='MERRA-MISR winter wind vector difference, 2.5 km')
plot_global_vector_map_maskedland(np.zeros(np.shape(u_winter_diff_2p5km)), v_winter_diff_2p5km, merra_lon, merra_lat, figure_file='MERRA-MISR winter wind vector v difference 2p5km', title='MERRA-MISR winter wind vector v difference, 2.5 km')
plot_global_vector_map_maskedland(u_winter_diff_4km, v_winter_diff_4km, merra_lon, merra_lat, figure_file='MERRA-MISR winter wind vector difference 4km', title='MERRA-MISR winter wind vector difference, 4 km')
plot_global_vector_map_maskedland(np.zeros(np.shape(u_winter_diff_4km)), v_winter_diff_4km, merra_lon, merra_lat, figure_file='MERRA-MISR winter wind vector v difference 4km', title='MERRA-MISR winter wind vector v difference, 4 km')

#Spring
plot_global_vector_map_maskedland(u_spring_diff_1p5km, v_spring_diff_1p5km, merra_lon, merra_lat, figure_file='MERRA-MISR spring wind vector difference 1p5km', title='MERRA-MISR spring wind vector difference 1.5 km')
plot_global_vector_map_maskedland(np.zeros(np.shape(u_spring_diff_1p5km)), v_spring_diff_1p5km, merra_lon, merra_lat, figure_file='MERRA-MISR spring wind vector v difference 1p5km', title='MERRA-MISR spring wind vector v difference, 1.5 km')
plot_global_vector_map_maskedland(u_spring_diff_2p5km, v_spring_diff_2p5km, merra_lon, merra_lat, figure_file='MERRA-MISR spring wind vector difference 2p5km', title='MERRA-MISR spring wind vector difference, 2.5 km')
plot_global_vector_map_maskedland(np.zeros(np.shape(u_spring_diff_2p5km)), v_spring_diff_2p5km, merra_lon, merra_lat, figure_file='MERRA-MISR spring wind vector v difference 2p5km', title='MERRA-MISR spring wind vector v difference, 2.5 km')
plot_global_vector_map_maskedland(u_spring_diff_4km, v_spring_diff_4km, merra_lon, merra_lat, figure_file='MERRA-MISR spring wind vector difference 4km', title='MERRA-MISR spring wind vector difference, 4 km')
plot_global_vector_map_maskedland(np.zeros(np.shape(u_spring_diff_4km)), v_spring_diff_4km, merra_lon, merra_lat, figure_file='MERRA-MISR spring wind vector v difference 4km', title='MERRA-MISR spring wind vector v difference, 4 km')

#Summer
plot_global_vector_map_maskedland(u_summer_diff_1p5km, v_summer_diff_1p5km, merra_lon, merra_lat, figure_file='MERRA-MISR summer wind vector difference 1p5km', title='MERRA-MISR summer wind vector difference 1.5 km')
plot_global_vector_map_maskedland(np.zeros(np.shape(u_summer_diff_1p5km)), v_summer_diff_1p5km, merra_lon, merra_lat, figure_file='MERRA-MISR summer wind vector v difference 1p5km', title='MERRA-MISR summer wind vector v difference, 1.5 km')
plot_global_vector_map_maskedland(u_summer_diff_2p5km, v_summer_diff_2p5km, merra_lon, merra_lat, figure_file='MERRA-MISR summer wind vector difference 2p5km', title='MERRA-MISR summer wind vector difference, 2.5 km')
plot_global_vector_map_maskedland(np.zeros(np.shape(u_summer_diff_2p5km)), v_summer_diff_2p5km, merra_lon, merra_lat, figure_file='MERRA-MISR summer wind vector v difference 2p5km', title='MERRA-MISR summer wind vector v difference, 2.5 km')
plot_global_vector_map_maskedland(u_summer_diff_4km, v_summer_diff_4km, merra_lon, merra_lat, figure_file='MERRA-MISR summer wind vector difference 4km', title='MERRA-MISR summer wind vector difference, 4 km')
plot_global_vector_map_maskedland(np.zeros(np.shape(u_summer_diff_4km)), v_summer_diff_4km, merra_lon, merra_lat, figure_file='MERRA-MISR summer wind vector v difference 4km', title='MERRA-MISR summer wind vector v difference, 4 km')

#Fall
plot_global_vector_map_maskedland(u_fall_diff_1p5km, v_fall_diff_1p5km, merra_lon, merra_lat, figure_file='MERRA-MISR fall wind vector difference 1p5km', title='MERRA-MISR fall wind vector difference 1.5 km')
plot_global_vector_map_maskedland(np.zeros(np.shape(u_fall_diff_1p5km)), v_fall_diff_1p5km, merra_lon, merra_lat, figure_file='MERRA-MISR fall wind vector v difference 1p5km', title='MERRA-MISR fall wind vector v difference, 1.5 km')
plot_global_vector_map_maskedland(u_fall_diff_2p5km, v_fall_diff_2p5km, merra_lon, merra_lat, figure_file='MERRA-MISR fall wind vector difference 2p5km', title='MERRA-MISR fall wind vector difference, 2.5 km')
plot_global_vector_map_maskedland(np.zeros(np.shape(u_fall_diff_2p5km)), v_fall_diff_2p5km, merra_lon, merra_lat, figure_file='MERRA-MISR fall wind vector v difference 2p5km', title='MERRA-MISR fall wind vector v difference, 2.5 km')
plot_global_vector_map_maskedland(u_fall_diff_4km, v_fall_diff_4km, merra_lon, merra_lat, figure_file='MERRA-MISR fall wind vector difference 4km', title='MERRA-MISR fall wind vector difference, 4 km')
plot_global_vector_map_maskedland(np.zeros(np.shape(u_fall_diff_4km)), v_fall_diff_4km, merra_lon, merra_lat, figure_file='MERRA-MISR fall wind vector v difference 4km', title='MERRA-MISR fall wind vector v difference, 4 km')

#Overall
plot_global_vector_map_maskedland(u_overall_diff_1p5km, v_overall_diff_1p5km, merra_lon, merra_lat, figure_file='MERRA-MISR overall wind vector difference 1p5km', title='MERRA-MISR overall wind vector difference 1.5 km')
plot_global_vector_map_maskedland(np.zeros(np.shape(u_overall_diff_1p5km)), v_overall_diff_1p5km, merra_lon, merra_lat, figure_file='MERRA-MISR overall wind vector v difference 1p5km', title='MERRA-MISR overall wind vector v difference, 1.5 km')
plot_global_vector_map_maskedland(u_overall_diff_2p5km, v_overall_diff_2p5km, merra_lon, merra_lat, figure_file='MERRA-MISR overall wind vector difference 2p5km', title='MERRA-MISR overall wind vector difference, 2.5 km')
plot_global_vector_map_maskedland(np.zeros(np.shape(u_overall_diff_2p5km)), v_overall_diff_2p5km, merra_lon, merra_lat, figure_file='MERRA-MISR overall wind vector v difference 2p5km', title='MERRA-MISR overall wind vector v difference, 2.5 km')
plot_global_vector_map_maskedland(u_overall_diff_4km, v_overall_diff_4km, merra_lon, merra_lat, figure_file='MERRA-MISR overall wind vector difference 4km', title='MERRA-MISR overall wind vector difference, 4 km')
plot_global_vector_map_maskedland(np.zeros(np.shape(u_overall_diff_4km)), v_overall_diff_4km, merra_lon, merra_lat, figure_file='MERRA-MISR overall wind vector v difference 4km', title='MERRA-MISR overall wind vector v difference, 4 km')


##Shears

#Winter
plot_global_vector_map_maskedland(u_shear_diff_winter_2p5km_1p5km, v_shear_diff_winter_2p5km_1p5km, merra_lon, merra_lat, figure_file='MERRA-MISR winter wind vector difference shear 2p5km 1p5km', title='MERRA-MISR winter wind vector difference shear 2.5km-1.5km')
plot_global_vector_map_maskedland(np.zeros(np.shape(u_shear_diff_winter_2p5km_1p5km)), v_shear_diff_winter_2p5km_1p5km, merra_lon, merra_lat, figure_file='MERRA-MISR winter wind vector v difference shear 2p5km 1p5km', title='MERRA-MISR winter wind vector v difference shear 2.5km-1.5km')
plot_global_vector_map_maskedland(u_shear_diff_winter_4km_1p5km, v_shear_diff_winter_4km_1p5km, merra_lon, merra_lat, figure_file='MERRA-MISR winter wind vector difference shear 4km 1p5km', title='MERRA-MISR winter wind vector difference shear 4km-1.5km')
plot_global_vector_map_maskedland(np.zeros(np.shape(u_shear_diff_winter_4km_1p5km)), v_shear_diff_winter_4km_1p5km, merra_lon, merra_lat, figure_file='MERRA-MISR winter wind vector v difference shear 4km 1p5km', title='MERRA-MISR winter wind vector v difference shear 4km-1.5km')
plot_global_vector_map_maskedland(u_shear_diff_winter_4km_2p5km, v_shear_diff_winter_4km_2p5km, merra_lon, merra_lat, figure_file='MERRA-MISR winter wind vector difference shear 4km 2p5km', title='MERRA-MISR winter wind vector difference shear 4km-2.5km')
plot_global_vector_map_maskedland(np.zeros(np.shape(u_shear_diff_winter_4km_2p5km)), v_shear_diff_winter_4km_2p5km, merra_lon, merra_lat, figure_file='MERRA-MISR winter wind vector v difference shear 4km 2p5km', title='MERRA-MISR winter wind vector v difference shear 4km-2.5km')

#Spring
plot_global_vector_map_maskedland(u_shear_diff_spring_2p5km_1p5km, v_shear_diff_spring_2p5km_1p5km, merra_lon, merra_lat, figure_file='MERRA-MISR spring wind vector difference shear 2p5km 1p5km', title='MERRA-MISR spring wind vector difference shear 2.5km-1.5km')
plot_global_vector_map_maskedland(np.zeros(np.shape(u_shear_diff_spring_2p5km_1p5km)), v_shear_diff_spring_2p5km_1p5km, merra_lon, merra_lat, figure_file='MERRA-MISR spring wind vector v difference shear 2p5km 1p5km', title='MERRA-MISR spring wind vector v difference shear 2.5km-1.5km')
plot_global_vector_map_maskedland(u_shear_diff_spring_4km_1p5km, v_shear_diff_spring_4km_1p5km, merra_lon, merra_lat, figure_file='MERRA-MISR spring wind vector difference shear 4km 1p5km', title='MERRA-MISR spring wind vector difference shear 4km-1.5km')
plot_global_vector_map_maskedland(np.zeros(np.shape(u_shear_diff_spring_4km_1p5km)), v_shear_diff_spring_4km_1p5km, merra_lon, merra_lat, figure_file='MERRA-MISR spring wind vector v difference shear 4km 1p5km', title='MERRA-MISR spring wind vector v difference shear 4km-1.5km')
plot_global_vector_map_maskedland(u_shear_diff_spring_4km_2p5km, v_shear_diff_spring_4km_2p5km, merra_lon, merra_lat, figure_file='MERRA-MISR spring wind vector difference shear 4km 2p5km', title='MERRA-MISR spring wind vector difference shear 4km-2.5km')
plot_global_vector_map_maskedland(np.zeros(np.shape(u_shear_diff_spring_4km_2p5km)), v_shear_diff_spring_4km_2p5km, merra_lon, merra_lat, figure_file='MERRA-MISR spring wind vector v difference shear 4km 2p5km', title='MERRA-MISR spring wind vector v difference shear 4km-2.5km')

#Summer
plot_global_vector_map_maskedland(u_shear_diff_summer_2p5km_1p5km, v_shear_diff_summer_2p5km_1p5km, merra_lon, merra_lat, figure_file='MERRA-MISR summer wind vector difference shear 2p5km 1p5km', title='MERRA-MISR summer wind vector difference shear 2.5km-1.5km')
plot_global_vector_map_maskedland(np.zeros(np.shape(u_shear_diff_summer_2p5km_1p5km)), v_shear_diff_summer_2p5km_1p5km, merra_lon, merra_lat, figure_file='MERRA-MISR summer wind vector v difference shear 2p5km 1p5km', title='MERRA-MISR summer wind vector v difference shear 2.5km-1.5km')
plot_global_vector_map_maskedland(u_shear_diff_summer_4km_1p5km, v_shear_diff_summer_4km_1p5km, merra_lon, merra_lat, figure_file='MERRA-MISR summer wind vector difference shear 4km 1p5km', title='MERRA-MISR summer wind vector difference shear 4km-1.5km')
plot_global_vector_map_maskedland(np.zeros(np.shape(u_shear_diff_summer_4km_1p5km)), v_shear_diff_summer_4km_1p5km, merra_lon, merra_lat, figure_file='MERRA-MISR summer wind vector v difference shear 4km 1p5km', title='MERRA-MISR summer wind vector v difference shear 4km-1.5km')
plot_global_vector_map_maskedland(u_shear_diff_summer_4km_2p5km, v_shear_diff_summer_4km_2p5km, merra_lon, merra_lat, figure_file='MERRA-MISR summer wind vector difference shear 4km 2p5km', title='MERRA-MISR summer wind vector difference shear 4km-2.5km')
plot_global_vector_map_maskedland(np.zeros(np.shape(u_shear_diff_summer_4km_2p5km)), v_shear_diff_summer_4km_2p5km, merra_lon, merra_lat, figure_file='MERRA-MISR summer wind vector v difference shear 4km 2p5km', title='MERRA-MISR summer wind vector v difference shear 4km-2.5km')

#Fall
plot_global_vector_map_maskedland(u_shear_diff_fall_2p5km_1p5km, v_shear_diff_fall_2p5km_1p5km, merra_lon, merra_lat, figure_file='MERRA-MISR fall wind vector difference shear 2p5km 1p5km', title='MERRA-MISR fall wind vector difference shear 2.5km-1.5km')
plot_global_vector_map_maskedland(np.zeros(np.shape(u_shear_diff_fall_2p5km_1p5km)), v_shear_diff_fall_2p5km_1p5km, merra_lon, merra_lat, figure_file='MERRA-MISR fall wind vector v difference shear 2p5km 1p5km', title='MERRA-MISR fall wind vector v difference shear 2.5km-1.5km')
plot_global_vector_map_maskedland(u_shear_diff_fall_4km_1p5km, v_shear_diff_fall_4km_1p5km, merra_lon, merra_lat, figure_file='MERRA-MISR fall wind vector difference shear 4km 1p5km', title='MERRA-MISR fall wind vector difference shear 4km-1.5km')
plot_global_vector_map_maskedland(np.zeros(np.shape(u_shear_diff_fall_4km_1p5km)), v_shear_diff_fall_4km_1p5km, merra_lon, merra_lat, figure_file='MERRA-MISR fall wind vector v difference shear 4km 1p5km', title='MERRA-MISR fall wind vector v difference shear 4km-1.5km')
plot_global_vector_map_maskedland(u_shear_diff_fall_4km_2p5km, v_shear_diff_fall_4km_2p5km, merra_lon, merra_lat, figure_file='MERRA-MISR fall wind vector difference shear 4km 2p5km', title='MERRA-MISR fall wind vector difference shear 4km-2.5km')
plot_global_vector_map_maskedland(np.zeros(np.shape(u_shear_diff_fall_4km_2p5km)), v_shear_diff_fall_4km_2p5km, merra_lon, merra_lat, figure_file='MERRA-MISR fall wind vector v difference shear 4km 2p5km', title='MERRA-MISR fall wind vector v difference shear 4km-2.5km')

#Overall
plot_global_vector_map_maskedland(u_shear_diff_overall_2p5km_1p5km, v_shear_diff_overall_2p5km_1p5km, merra_lon, merra_lat, figure_file='MERRA-MISR overall wind vector difference shear 2p5km 1p5km', title='MERRA-MISR overall wind vector difference shear 2.5km-1.5km')
plot_global_vector_map_maskedland(np.zeros(np.shape(u_shear_diff_overall_2p5km_1p5km)), v_shear_diff_overall_2p5km_1p5km, merra_lon, merra_lat, figure_file='MERRA-MISR overall wind vector v difference shear 2p5km 1p5km', title='MERRA-MISR overall wind vector v difference shear 2.5km-1.5km')
plot_global_vector_map_maskedland(u_shear_diff_overall_4km_1p5km, v_shear_diff_overall_4km_1p5km, merra_lon, merra_lat, figure_file='MERRA-MISR overall wind vector difference shear 4km 1p5km', title='MERRA-MISR overall wind vector difference shear 4km-1.5km') 
plot_global_vector_map_maskedland(np.zeros(np.shape(u_shear_diff_overall_4km_1p5km)), v_shear_diff_overall_4km_1p5km, merra_lon, merra_lat, figure_file='MERRA-MISR overall wind vector v difference shear 4km 1p5km', title='MERRA-MISR overall wind vector v difference shear 4km-1.5km')
plot_global_vector_map_maskedland(u_shear_diff_overall_4km_2p5km, v_shear_diff_overall_4km_2p5km, merra_lon, merra_lat, figure_file='MERRA-MISR overall wind vector difference shear 4km 2p5km', title='MERRA-MISR overall wind vector difference shear 4km-2.5km')
plot_global_vector_map_maskedland(np.zeros(np.shape(u_shear_diff_overall_4km_2p5km)), v_shear_diff_overall_4km_2p5km, merra_lon, merra_lat, figure_file='MERRA-MISR overall wind vector v difference shear 4km 2p5km', title='MERRA-MISR overall wind vector v difference shear 4km-2.5km')




