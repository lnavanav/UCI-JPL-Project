from netCDF4 import Dataset, num2date
import numpy as np
import numpy.ma as ma
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, maskoceans


########################################## FUNCTIONS ##########################################

def read_AR_winds(filename):
	f = Dataset(filename)
	lon = f.variables['lon'][:]
	lat = f.variables['lat'][:]
	time = f.variables['time']
	dates = num2date(time[:], time.units)
	AR_shape = f.variables['AR_shape'][:]
	MERRA2_speed = f.variables['MERRA2_speed'][:]
	MERRA2_U = f.variables['MERRA2_U'][:]
	MERRA2_V = f.variables['MERRA2_V'][:]
	MISR_CTH = f.variables['MISR_CTH'][:]
	MISR_CTH_sample_size = f.variables['MISR_CTH_sample_size'][:]
	MISR_speed = f.variables['MISR_speed'][:]
	MISR_speed_at_CTH = f.variables['MISR_speed_at_CTH'][:]
	MISR_U = f.variables['MISR_U'][:]
	MISR_V = f.variables['MISR_V'][:]
	MISR_U_at_CTH = f.variables['MISR_U_at_CTH'][:]
	MISR_V_at_CTH = f.variables['MISR_V_at_CTH'][:]
	return lon, lat, dates, AR_shape, MERRA2_speed, MERRA2_U, MERRA2_V, MISR_CTH, MISR_CTH_sample_size, MISR_speed, MISR_speed_at_CTH, MISR_U, MISR_V, MISR_U_at_CTH, MISR_V_at_CTH

#Plots wind speed contours over our west coast region of interest
def plot_local_contour_map(data, lon, lat, levels, figure_file, title='title', cmap=cm.jet): #added title
    lons, lats = np.meshgrid(lon, lat)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    m = Basemap(ax=ax, projection = 'cyl', llcrnrlat = lat.min(), urcrnrlat = lat.max(), 
        llcrnrlon = lon.min(), urcrnrlon = lon.max(), resolution = 'h', fix_aspect = True)
    x,y = m(lons, lats)
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=.75)
    m.drawstates(linewidth=.5)
    max = m.contourf(x, y, data, levels = levels, extend='both', cmap=cmap)
    cbar_ax = fig.add_axes([0.85, 0.3, 0.01, 0.4]) #[0.15, 0.25, 0.3, 0.01])
    cbar_ax.set_xlabel('m/s')
    cb=plt.colorbar(max, cax=cbar_ax)  #orientation = 'horizontal',
    cb.ax.tick_params(labelsize=8)
    ax.set_title(title)
    fig.savefig(figure_file,dpi=600,bbox_inches='tight')
    plt.show()
	
#Plots wind vectors over our west coast region of interest
def plot_local_vector_map(uwind, vwind, lon, lat, figure_file='filename', title='title', yskip=2, xskip=3):
    lons, lats = np.meshgrid(lon, lat)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    m = Basemap(ax=ax, projection = 'cyl', llcrnrlat = lat.min(), urcrnrlat = lat.max(),
        llcrnrlon = lon.min(), urcrnrlon = lon.max(), resolution = 'h', fix_aspect = True)
    x,y = m(lons, lats)
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=.75)
    m.drawstates(linewidth=.5)
    N = ma.mean(np.sqrt(uwind[::yskip, ::xskip]**2+vwind[::yskip, ::xskip]**2)) #Obsolete?
    max = m.quiver(x[::yskip, ::xskip], y[::yskip, ::xskip], uwind[::yskip, ::xskip]/N, vwind[::yskip, ::xskip]/N, color='blue', pivot='middle', headwidth=3)
    ax.set_title(title)
    fig.savefig(figure_file,dpi=600,bbox_inches='tight')
    plt.show()

	


########################################## VARIABLES ##########################################

##Get all data
lon, lat, dates, AR_shape = read_AR_winds('West_Coast_AR_and_Wind_18UTC_daily_NOV2000-APR2017.nc')[0:4]
MERRA2_speed, MERRA2_U, MERRA2_V = read_AR_winds('West_Coast_AR_and_Wind_18UTC_daily_NOV2000-APR2017.nc')[4:7]
MISR_CTH, MISR_CTH_sample_size, MISR_speed, MISR_speed_at_CTH, MISR_U, MISR_V, MISR_U_at_CTH, MISR_V_at_CTH = read_AR_winds('West_Coast_AR_and_Wind_18UTC_daily_NOV2000-APR2017.nc')[7:]


##Time indicies
#Currently doing months but can be sliced up into whatever we want

months = np.array([i.month for i in dates]) 
november_t_indices = np.where(months == 11) #blocks of 30 days each
december_t_indices = np.where(months == 12) #blocks of 31 days each
january_t_indices = np.where(months == 1)   #blocks of 30 days each
february_t_indices = np.where(months == 2)  #blocks of 28 days each (feb. 29 has been removed) 
march_t_indices = np.where(months == 3)     #blocks of 31 days each
april_t_indices = np.where(months == 4)     #blocks of 30 days each


##Speeds
#Need to add more - just testing for now. 

merra_overall_speed_1p5km = ma.mean(MERRA2_speed[:, 0, :, :], axis=0)
misr_overall_speed_1p5km = ma.mean(MISR_speed[:, 0, :, :], axis=0)
merra_overall_u_1p5km = ma.mean(MERRA2_U[:, 0, :, :], axis=0)
merra_overall_v_1p5km = ma.mean(MERRA2_V[:, 0, :, :], axis=0)
misr_overall_u_1p5km = ma.mean(MISR_U[:, 0, :, :], axis=0)
misr_overall_v_1p5km = ma.mean(MISR_V[:, 0, :, :], axis=0)

#TIME INDEXED STUFF SEEMS BUGGED - WILL LOOK INTO IT LATER - problem to do with averaging over index lists with jump discontinuities? idk
# merra_november_speed_1p5km = ma.mean(MERRA2_speed[november_t_indices, 0, :, :], axis=0)
# merra_november_speed_2p5km = ma.mean(MERRA2_speed[november_t_indices, 1, :, :], axis=0)
# merra_november_speed_4km = ma.mean(MERRA2_speed[november_t_indices, 2, :, :], axis=0)
# merra_november_u_1p5km = ma.mean(MERRA2_U[november_t_indices, 0, :, :], axis=0)
# merra_november_v_1p5km = ma.mean(MERRA2_V[november_t_indices, 0, :, :], axis=0)

# misr_november_speed_1p5km = ma.mean(MISR_speed[november_t_indices, 0, :, :], axis=0)
# misr_november_speed_2p5km = ma.mean(MISR_speed[november_t_indices, 1, :, :], axis=0)
# misr_november_speed_4km = ma.mean(MISR_speed[november_t_indices, 2, :, :], axis=0)
# misr_november_u_1p5km = ma.mean(MISR_U[november_t_indices, 0, :, :], axis=0)
# misr_november_v_1p5km = ma.mean(MISR_V[november_t_indices, 0, :, :], axis=0)

# november_speed_diff_1p5km = merra_november_speed_1p5km - misr_november_speed_1p5km
# november_speed_diff_2p5km = merra_november_speed_2p5km - misr_november_speed_2p5km
# november_speed_diff_4km = merra_november_speed_4km - misr_november_speed_4km




########################################## PLOTS ##########################################

plot_local_contour_map(merra_overall_speed_1p5km-misr_overall_speed_1p5km, lon, lat, np.arange(21)*0.5-5, figure_file='merra-misr overall speeds 1p5km', title='MERRA - MISR overall wind speed, 1.5 km', cmap=cm.RdBu_r)

plot_local_vector_map(merra_overall_u_1p5km-misr_overall_u_1p5km, merra_overall_v_1p5km-misr_overall_v_1p5km, lon, lat, figure_file='merra-misr vectors 1p5km', title='merra-misr vectors, 1.5 km')






