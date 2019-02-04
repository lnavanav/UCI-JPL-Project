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

############# WIND VECTOR STUFF ############# 

#MERRA and MISR share a common time so no need to do it twice
months = np.array([i.month for i in merra_time]) 
winter_t_index = np.where((months >=12) | (months <=2))[0] #1 December to 28 February
spring_t_index = np.where((months >=3) & (months <= 5))[0] #1 March to 31 May
summer_t_index = np.where((months >=6) & (months <= 8))[0] #1 June to 31 August
fall_t_index = np.where((months >=9) & (months <=11))[0] #1 September to 30 November 

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

#####Plot wind vector stuff#####
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






