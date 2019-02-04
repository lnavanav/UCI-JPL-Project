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
	is_land = f.variables['islnd'][:]
	is_coast = f.variables['iscst'][:]
	AR_shape = f.variables['AR_shape'][:]
	AR_lfloc = f.variables['AR_lfloc'][:]
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
	return lon, lat, dates, is_land, is_coast, AR_shape, AR_lfloc, MERRA2_speed, MERRA2_U, MERRA2_V, MISR_CTH, MISR_CTH_sample_size, MISR_speed, MISR_speed_at_CTH, MISR_U, MISR_V, MISR_U_at_CTH, MISR_V_at_CTH

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
	
#Plots wind speed contours over our west coast region of interest. Masks the land
def plot_local_contour_map_maskedland(data, lon, lat, plot_mask, levels, figure_file, title='title', cmap=cm.jet): #added title
    lons, lats = np.meshgrid(lon, lat)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    m = Basemap(ax=ax, projection = 'cyl', llcrnrlat = lat.min(), urcrnrlat = lat.max(), 
        llcrnrlon = lon.min(), urcrnrlon = lon.max(), resolution = 'h', fix_aspect = True)
    x,y = m(lons, lats)
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=.75)
    m.drawstates(linewidth=.5)
	#MASKING STUFF START
    masked_data = ma.array(data, mask=plot_mask)
    #MASKING STUFF END
    max = m.contourf(x, y, masked_data, levels = levels, extend='both', cmap=cmap)
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
	
#Plots wind vectors over our west coast region of interest. Masked land	
def plot_local_vector_map_maskedland(uwind, vwind, lon, lat, plot_mask, figure_file='filename', title='title', yskip=2, xskip=3):
    lons, lats = np.meshgrid(lon, lat)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    m = Basemap(ax=ax, projection = 'cyl', llcrnrlat = lat.min(), urcrnrlat = lat.max(),
        llcrnrlon = lon.min(), urcrnrlon = lon.max(), resolution = 'h', fix_aspect = True)
    x,y = m(lons, lats)
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=.75)
    m.drawstates(linewidth=.5)
	#MASKING STUFF START
    masked_u = ma.array(uwind, mask=plot_mask)
	masked_v = ma.array(vwind, mask=plot_mask)
    #MASKING STUFF END
    N = ma.mean(np.sqrt(masked_u[::yskip, ::xskip]**2+masked_v[::yskip, ::xskip]**2)) #Obsolete?
    max = m.quiver(x[::yskip, ::xskip], y[::yskip, ::xskip], masked_u[::yskip, ::xskip]/N, masked_v[::yskip, ::xskip]/N, color='blue', pivot='middle', headwidth=3)
    ax.set_title(title)
    fig.savefig(figure_file,dpi=600,bbox_inches='tight')
    plt.show()	
	
#Make the mask for plotting stuff - excludes land but includes the coast. HORRIBLY WRITTEN BUT WORKS OK	
def build_land_mask(is_land, is_coast):
	temp_mask = ~is_land.mask
	lat, lon = np.shape(is_land)	#is_land and is_coast overlap, so ~is_land and is_coast don't.
	for i in range(lat):			#need to build a mask that allows us to plot over the ocean and the coasts but NOT the land.
		for j in range(lon):		#because these are arrays of NaN and 1.0, no need to use mask here - call it in the plotting function.
			if is_coast[i,j] == 1:
				temp_mask[i,j] = False
	return temp_mask
	
def get_lfloc_dates(lfloc):
	t_list = []
	for t in lfloc[0]:
		if ma.mean(lfloc[t,:]) == 1: #there was landfall on that day - don't care about location
			t_list.append(t)
	return t_list


########################################## VARIABLES ##########################################

##Get all data
lon, lat, dates, is_land, is_coast, AR_shape, AR_lfloc = read_AR_winds('West_Coast_AR_and_Wind_18UTC_daily_NOV2000-APR2017.nc')[0:7]
MERRA2_speed, MERRA2_U, MERRA2_V = read_AR_winds('West_Coast_AR_and_Wind_18UTC_daily_NOV2000-APR2017.nc')[7:10]
MISR_CTH, MISR_CTH_sample_size, MISR_speed, MISR_speed_at_CTH, MISR_U, MISR_V, MISR_U_at_CTH, MISR_V_at_CTH = read_AR_winds('West_Coast_AR_and_Wind_18UTC_daily_NOV2000-APR2017.nc')[10:]

land_mask_for_plotting = build_land_mask(is_land, is_coast)
	
##Time indicies
months = np.array([i.month for i in dates]) 
november_t_indices = np.where(months == 11) #blocks of 30 days each
december_t_indices = np.where(months == 12) #blocks of 31 days each
january_t_indices = np.where(months == 1)   #blocks of 30 days each
february_t_indices = np.where(months == 2)  #blocks of 28 days each (feb. 29 has been removed) 
march_t_indices = np.where(months == 3)     #blocks of 31 days each
april_t_indices = np.where(months == 4)     #blocks of 30 days each

##Speeds and vectors
#CHANGES FOR THE DIFFERENCE: now taking the mean of the subtracted vectors rather than difference of the means. There shouldn't be a need to define specific masks - python should automatically mask the arrays as they're subtracted,
#because subtraction is a binary operation for masked arrays (http://folk.uio.no/inf3330/scripting/doc/python/NumPy/Numeric/numpy-22.html)

#Overall
merra_overall_speed_1p5km = ma.mean(MERRA2_speed[:, 0, :, :], axis=0)
merra_overall_speed_2p5km = ma.mean(MERRA2_speed[:, 1, :, :], axis=0)
merra_overall_speed_4km = ma.mean(MERRA2_speed[:, 2, :, :], axis=0)
merra_overall_u_1p5km = ma.mean(MERRA2_U[:, 0, :, :], axis=0)
merra_overall_v_1p5km = ma.mean(MERRA2_V[:, 0, :, :], axis=0)
merra_overall_u_2p5km = ma.mean(MERRA2_U[:, 1, :, :], axis=0)
merra_overall_v_2p5km = ma.mean(MERRA2_V[:, 1, :, :], axis=0)
merra_overall_u_4km = ma.mean(MERRA2_U[:, 2, :, :], axis=0)
merra_overall_v_4km = ma.mean(MERRA2_V[:, 2, :, :], axis=0)
misr_overall_speed_1p5km = ma.mean(MISR_speed[:, 0, :, :], axis=0)
misr_overall_speed_2p5km = ma.mean(MISR_speed[:, 1, :, :], axis=0)
misr_overall_speed_4km = ma.mean(MISR_speed[:, 2, :, :], axis=0)
misr_overall_u_1p5km = ma.mean(MISR_U[:, 0, :, :], axis=0)
misr_overall_v_1p5km = ma.mean(MISR_V[:, 0, :, :], axis=0)
misr_overall_u_2p5km = ma.mean(MISR_U[:, 1, :, :], axis=0)
misr_overall_v_2p5km = ma.mean(MISR_V[:, 1, :, :], axis=0)
misr_overall_u_4km = ma.mean(MISR_U[:, 2, :, :], axis=0)
misr_overall_v_4km = ma.mean(MISR_V[:, 2, :, :], axis=0)
overall_speed_diff_1p5km = ma.mean(MERRA2_speed[:, 0, :, :] - MISR_speed[:, 0, :, :], axis=0)
overall_speed_diff_2p5km = ma.mean(MERRA2_speed[:, 1, :, :] - MISR_speed[:, 1, :, :], axis=0)
overall_speed_diff_4km = ma.mean(MERRA2_speed[:, 2, :, :] - MISR_speed[:, 2, :, :], axis=0)
overall_u_diff_1p5km = ma.mean(MERRA2_U[:, 0, :, :] - MISR_U[:, 0, :, :], axis=0)
overall_v_diff_1p5km = ma.mean(MERRA2_V[:, 0, :, :] - MISR_V[:, 0, :, :], axis=0)
overall_u_diff_2p5km = ma.mean(MERRA2_U[:, 1, :, :] - MISR_U[:, 1, :, :], axis=0)
overall_v_diff_2p5km = ma.mean(MERRA2_V[:, 1, :, :] - MISR_V[:, 1, :, :], axis=0)
overall_u_diff_4km = ma.mean(MERRA2_U[:, 2, :, :] - MISR_U[:, 2, :, :], axis=0)
overall_v_diff_4km = ma.mean(MERRA2_V[:, 1, :, :] - MISR_V[:, 1, :, :], axis=0)

#November
merra_november_speed_1p5km = ma.mean(MERRA2_speed[november_t_indices[0], 0, :, :], axis=0)
merra_november_speed_2p5km = ma.mean(MERRA2_speed[november_t_indices[0], 1, :, :], axis=0)
merra_november_speed_4km = ma.mean(MERRA2_speed[november_t_indices[0], 2, :, :], axis=0)
merra_november_u_1p5km = ma.mean(MERRA2_U[november_t_indices[0], 0, :, :], axis=0)
merra_november_v_1p5km = ma.mean(MERRA2_V[november_t_indices[0], 0, :, :], axis=0)
merra_november_u_2p5km = ma.mean(MERRA2_U[november_t_indices[0], 1, :, :], axis=0)
merra_november_v_2p5km = ma.mean(MERRA2_V[november_t_indices[0], 1, :, :], axis=0)
merra_november_u_4km = ma.mean(MERRA2_U[november_t_indices[0], 2, :, :], axis=0)
merra_november_v_4km = ma.mean(MERRA2_V[november_t_indices[0], 2, :, :], axis=0)
misr_november_speed_1p5km = ma.mean(MISR_speed[november_t_indices[0], 0, :, :], axis=0)
misr_november_speed_2p5km = ma.mean(MISR_speed[november_t_indices[0], 1, :, :], axis=0)
misr_november_speed_4km = ma.mean(MISR_speed[november_t_indices[0], 2, :, :], axis=0)
misr_november_u_1p5km = ma.mean(MISR_U[november_t_indices[0], 0, :, :], axis=0)
misr_november_v_1p5km = ma.mean(MISR_V[november_t_indices[0], 0, :, :], axis=0)
misr_november_u_2p5km = ma.mean(MISR_U[november_t_indices[0], 1, :, :], axis=0)
misr_november_v_2p5km = ma.mean(MISR_V[november_t_indices[0], 1, :, :], axis=0)
misr_november_u_4km = ma.mean(MISR_U[november_t_indices[0], 2, :, :], axis=0)
misr_november_v_4km = ma.mean(MISR_V[november_t_indices[0], 2, :, :], axis=0)
november_speed_diff_1p5km = ma.mean(MERRA2_speed[november_t_indices[0], 0, :, :] - MISR_speed[november_t_indices[0], 0, :, :], axis=0)
november_speed_diff_2p5km = ma.mean(MERRA2_speed[november_t_indices[0], 1, :, :] - MISR_speed[november_t_indices[0], 1, :, :], axis=0)
november_speed_diff_4km = ma.mean(MERRA2_speed[november_t_indices[0], 2, :, :] - MISR_speed[november_t_indices[0], 2, :, :], axis=0)
november_u_diff_1p5km = ma.mean(MERRA2_U[november_t_indices[0], 0, :, :] - MISR_U[november_t_indices[0], 0, :, :], axis=0)
november_v_diff_1p5km = ma.mean(MERRA2_V[november_t_indices[0], 0, :, :] - MISR_V[november_t_indices[0], 0, :, :], axis=0)
november_u_diff_2p5km = ma.mean(MERRA2_U[november_t_indices[0], 1, :, :] - MISR_U[november_t_indices[0], 1, :, :], axis=0)
november_v_diff_2p5km = ma.mean(MERRA2_V[november_t_indices[0], 1, :, :] - MISR_V[november_t_indices[0], 1, :, :], axis=0)
november_u_diff_4km = ma.mean(MERRA2_U[november_t_indices[0], 2, :, :] - MISR_U[november_t_indices[0], 2, :, :], axis=0)
november_v_diff_4km = ma.mean(MERRA2_V[november_t_indices[0], 1, :, :] - MISR_V[november_t_indices[0], 1, :, :], axis=0)

#December
merra_december_speed_1p5km = ma.mean(MERRA2_speed[december_t_indices[0], 0, :, :], axis=0)
merra_december_speed_2p5km = ma.mean(MERRA2_speed[december_t_indices[0], 1, :, :], axis=0)
merra_december_speed_4km = ma.mean(MERRA2_speed[december_t_indices[0], 2, :, :], axis=0)
merra_december_u_1p5km = ma.mean(MERRA2_U[december_t_indices[0], 0, :, :], axis=0)
merra_december_v_1p5km = ma.mean(MERRA2_V[december_t_indices[0], 0, :, :], axis=0)
merra_december_u_2p5km = ma.mean(MERRA2_U[december_t_indices[0], 1, :, :], axis=0)
merra_december_v_2p5km = ma.mean(MERRA2_V[december_t_indices[0], 1, :, :], axis=0)
merra_december_u_4km = ma.mean(MERRA2_U[december_t_indices[0], 2, :, :], axis=0)
merra_december_v_4km = ma.mean(MERRA2_V[december_t_indices[0], 2, :, :], axis=0)
misr_december_speed_1p5km = ma.mean(MISR_speed[december_t_indices[0], 0, :, :], axis=0)
misr_december_speed_2p5km = ma.mean(MISR_speed[december_t_indices[0], 1, :, :], axis=0)
misr_december_speed_4km = ma.mean(MISR_speed[december_t_indices[0], 2, :, :], axis=0)
misr_december_u_1p5km = ma.mean(MISR_U[december_t_indices[0], 0, :, :], axis=0)
misr_december_v_1p5km = ma.mean(MISR_V[december_t_indices[0], 0, :, :], axis=0)
misr_december_u_2p5km = ma.mean(MISR_U[december_t_indices[0], 1, :, :], axis=0)
misr_december_v_2p5km = ma.mean(MISR_V[december_t_indices[0], 1, :, :], axis=0)
misr_december_u_4km = ma.mean(MISR_U[december_t_indices[0], 2, :, :], axis=0)
misr_december_v_4km = ma.mean(MISR_V[december_t_indices[0], 2, :, :], axis=0)
december_speed_diff_1p5km = ma.mean(MERRA2_speed[december_t_indices[0], 0, :, :] - MISR_speed[december_t_indices[0], 0, :, :], axis=0)
december_speed_diff_2p5km = ma.mean(MERRA2_speed[december_t_indices[0], 1, :, :] - MISR_speed[december_t_indices[0], 1, :, :], axis=0)
december_speed_diff_4km = ma.mean(MERRA2_speed[december_t_indices[0], 2, :, :] - MISR_speed[december_t_indices[0], 2, :, :], axis=0)
december_u_diff_1p5km = ma.mean(MERRA2_U[december_t_indices[0], 0, :, :] - MISR_U[december_t_indices[0], 0, :, :], axis=0)
december_v_diff_1p5km = ma.mean(MERRA2_V[december_t_indices[0], 0, :, :] - MISR_V[december_t_indices[0], 0, :, :], axis=0)
december_u_diff_2p5km = ma.mean(MERRA2_U[december_t_indices[0], 1, :, :] - MISR_U[december_t_indices[0], 1, :, :], axis=0)
december_v_diff_2p5km = ma.mean(MERRA2_V[december_t_indices[0], 1, :, :] - MISR_V[december_t_indices[0], 1, :, :], axis=0)
december_u_diff_4km = ma.mean(MERRA2_U[december_t_indices[0], 2, :, :] - MISR_U[december_t_indices[0], 2, :, :], axis=0)
december_v_diff_4km = ma.mean(MERRA2_V[december_t_indices[0], 1, :, :] - MISR_V[december_t_indices[0], 1, :, :], axis=0)

#January
merra_january_speed_1p5km = ma.mean(MERRA2_speed[january_t_indices[0], 0, :, :], axis=0)
merra_january_speed_2p5km = ma.mean(MERRA2_speed[january_t_indices[0], 1, :, :], axis=0)
merra_january_speed_4km = ma.mean(MERRA2_speed[january_t_indices[0], 2, :, :], axis=0)
merra_january_u_1p5km = ma.mean(MERRA2_U[january_t_indices[0], 0, :, :], axis=0)
merra_january_v_1p5km = ma.mean(MERRA2_V[january_t_indices[0], 0, :, :], axis=0)
merra_january_u_2p5km = ma.mean(MERRA2_U[january_t_indices[0], 1, :, :], axis=0)
merra_january_v_2p5km = ma.mean(MERRA2_V[january_t_indices[0], 1, :, :], axis=0)
merra_january_u_4km = ma.mean(MERRA2_U[january_t_indices[0], 2, :, :], axis=0)
merra_january_v_4km = ma.mean(MERRA2_V[january_t_indices[0], 2, :, :], axis=0)
misr_january_speed_1p5km = ma.mean(MISR_speed[january_t_indices[0], 0, :, :], axis=0)
misr_january_speed_2p5km = ma.mean(MISR_speed[january_t_indices[0], 1, :, :], axis=0)
misr_january_speed_4km = ma.mean(MISR_speed[january_t_indices[0], 2, :, :], axis=0)
misr_january_u_1p5km = ma.mean(MISR_U[january_t_indices[0], 0, :, :], axis=0)
misr_january_v_1p5km = ma.mean(MISR_V[january_t_indices[0], 0, :, :], axis=0)
misr_january_u_2p5km = ma.mean(MISR_U[january_t_indices[0], 1, :, :], axis=0)
misr_january_v_2p5km = ma.mean(MISR_V[january_t_indices[0], 1, :, :], axis=0)
misr_january_u_4km = ma.mean(MISR_U[january_t_indices[0], 2, :, :], axis=0)
misr_january_v_4km = ma.mean(MISR_V[january_t_indices[0], 2, :, :], axis=0)
january_speed_diff_1p5km = ma.mean(MERRA2_speed[january_t_indices[0], 0, :, :] - MISR_speed[january_t_indices[0], 0, :, :], axis=0)
january_speed_diff_2p5km = ma.mean(MERRA2_speed[january_t_indices[0], 1, :, :] - MISR_speed[january_t_indices[0], 1, :, :], axis=0)
january_speed_diff_4km = ma.mean(MERRA2_speed[january_t_indices[0], 2, :, :] - MISR_speed[january_t_indices[0], 2, :, :], axis=0)
january_u_diff_1p5km = ma.mean(MERRA2_U[january_t_indices[0], 0, :, :] - MISR_U[january_t_indices[0], 0, :, :], axis=0)
january_v_diff_1p5km = ma.mean(MERRA2_V[january_t_indices[0], 0, :, :] - MISR_V[january_t_indices[0], 0, :, :], axis=0)
january_u_diff_2p5km = ma.mean(MERRA2_U[january_t_indices[0], 1, :, :] - MISR_U[january_t_indices[0], 1, :, :], axis=0)
january_v_diff_2p5km = ma.mean(MERRA2_V[january_t_indices[0], 1, :, :] - MISR_V[january_t_indices[0], 1, :, :], axis=0)
january_u_diff_4km = ma.mean(MERRA2_U[january_t_indices[0], 2, :, :] - MISR_U[january_t_indices[0], 2, :, :], axis=0)
january_v_diff_4km = ma.mean(MERRA2_V[january_t_indices[0], 1, :, :] - MISR_V[january_t_indices[0], 1, :, :], axis=0)

#February
merra_february_speed_1p5km = ma.mean(MERRA2_speed[february_t_indices[0], 0, :, :], axis=0)
merra_february_speed_2p5km = ma.mean(MERRA2_speed[february_t_indices[0], 1, :, :], axis=0)
merra_february_speed_4km = ma.mean(MERRA2_speed[february_t_indices[0], 2, :, :], axis=0)
merra_february_u_1p5km = ma.mean(MERRA2_U[february_t_indices[0], 0, :, :], axis=0)
merra_february_v_1p5km = ma.mean(MERRA2_V[february_t_indices[0], 0, :, :], axis=0)
merra_february_u_2p5km = ma.mean(MERRA2_U[february_t_indices[0], 1, :, :], axis=0)
merra_february_v_2p5km = ma.mean(MERRA2_V[february_t_indices[0], 1, :, :], axis=0)
merra_february_u_4km = ma.mean(MERRA2_U[february_t_indices[0], 2, :, :], axis=0)
merra_february_v_4km = ma.mean(MERRA2_V[february_t_indices[0], 2, :, :], axis=0)
misr_february_speed_1p5km = ma.mean(MISR_speed[february_t_indices[0], 0, :, :], axis=0)
misr_february_speed_2p5km = ma.mean(MISR_speed[february_t_indices[0], 1, :, :], axis=0)
misr_february_speed_4km = ma.mean(MISR_speed[february_t_indices[0], 2, :, :], axis=0)
misr_february_u_1p5km = ma.mean(MISR_U[february_t_indices[0], 0, :, :], axis=0)
misr_february_v_1p5km = ma.mean(MISR_V[february_t_indices[0], 0, :, :], axis=0)
misr_february_u_2p5km = ma.mean(MISR_U[february_t_indices[0], 1, :, :], axis=0)
misr_february_v_2p5km = ma.mean(MISR_V[february_t_indices[0], 1, :, :], axis=0)
misr_february_u_4km = ma.mean(MISR_U[february_t_indices[0], 2, :, :], axis=0)
misr_february_v_4km = ma.mean(MISR_V[february_t_indices[0], 2, :, :], axis=0)
february_speed_diff_1p5km = ma.mean(MERRA2_speed[february_t_indices[0], 0, :, :] - MISR_speed[february_t_indices[0], 0, :, :], axis=0)
february_speed_diff_2p5km = ma.mean(MERRA2_speed[february_t_indices[0], 1, :, :] - MISR_speed[february_t_indices[0], 1, :, :], axis=0)
february_speed_diff_4km = ma.mean(MERRA2_speed[february_t_indices[0], 2, :, :] - MISR_speed[february_t_indices[0], 2, :, :], axis=0)
february_u_diff_1p5km = ma.mean(MERRA2_U[february_t_indices[0], 0, :, :] - MISR_U[february_t_indices[0], 0, :, :], axis=0)
february_v_diff_1p5km = ma.mean(MERRA2_V[february_t_indices[0], 0, :, :] - MISR_V[february_t_indices[0], 0, :, :], axis=0)
february_u_diff_2p5km = ma.mean(MERRA2_U[february_t_indices[0], 1, :, :] - MISR_U[february_t_indices[0], 1, :, :], axis=0)
february_v_diff_2p5km = ma.mean(MERRA2_V[february_t_indices[0], 1, :, :] - MISR_V[february_t_indices[0], 1, :, :], axis=0)
february_u_diff_4km = ma.mean(MERRA2_U[february_t_indices[0], 2, :, :] - MISR_U[february_t_indices[0], 2, :, :], axis=0)
february_v_diff_4km = ma.mean(MERRA2_V[february_t_indices[0], 1, :, :] - MISR_V[february_t_indices[0], 1, :, :], axis=0)

#March
merra_march_speed_1p5km = ma.mean(MERRA2_speed[march_t_indices[0], 0, :, :], axis=0)
merra_march_speed_2p5km = ma.mean(MERRA2_speed[march_t_indices[0], 1, :, :], axis=0)
merra_march_speed_4km = ma.mean(MERRA2_speed[march_t_indices[0], 2, :, :], axis=0)
merra_march_u_1p5km = ma.mean(MERRA2_U[march_t_indices[0], 0, :, :], axis=0)
merra_march_v_1p5km = ma.mean(MERRA2_V[march_t_indices[0], 0, :, :], axis=0)
merra_march_u_2p5km = ma.mean(MERRA2_U[march_t_indices[0], 1, :, :], axis=0)
merra_march_v_2p5km = ma.mean(MERRA2_V[march_t_indices[0], 1, :, :], axis=0)
merra_march_u_4km = ma.mean(MERRA2_U[march_t_indices[0], 2, :, :], axis=0)
merra_march_v_4km = ma.mean(MERRA2_V[march_t_indices[0], 2, :, :], axis=0)
misr_march_speed_1p5km = ma.mean(MISR_speed[march_t_indices[0], 0, :, :], axis=0)
misr_march_speed_2p5km = ma.mean(MISR_speed[march_t_indices[0], 1, :, :], axis=0)
misr_march_speed_4km = ma.mean(MISR_speed[march_t_indices[0], 2, :, :], axis=0)
misr_march_u_1p5km = ma.mean(MISR_U[march_t_indices[0], 0, :, :], axis=0)
misr_march_v_1p5km = ma.mean(MISR_V[march_t_indices[0], 0, :, :], axis=0)
misr_march_u_2p5km = ma.mean(MISR_U[march_t_indices[0], 1, :, :], axis=0)
misr_march_v_2p5km = ma.mean(MISR_V[march_t_indices[0], 1, :, :], axis=0)
misr_march_u_4km = ma.mean(MISR_U[march_t_indices[0], 2, :, :], axis=0)
misr_march_v_4km = ma.mean(MISR_V[march_t_indices[0], 2, :, :], axis=0)
march_speed_diff_1p5km = ma.mean(MERRA2_speed[march_t_indices[0], 0, :, :] - MISR_speed[march_t_indices[0], 0, :, :], axis=0)
march_speed_diff_2p5km = ma.mean(MERRA2_speed[march_t_indices[0], 1, :, :] - MISR_speed[march_t_indices[0], 1, :, :], axis=0)
march_speed_diff_4km = ma.mean(MERRA2_speed[march_t_indices[0], 2, :, :] - MISR_speed[march_t_indices[0], 2, :, :], axis=0)
march_u_diff_1p5km = ma.mean(MERRA2_U[march_t_indices[0], 0, :, :] - MISR_U[march_t_indices[0], 0, :, :], axis=0)
march_v_diff_1p5km = ma.mean(MERRA2_V[march_t_indices[0], 0, :, :] - MISR_V[march_t_indices[0], 0, :, :], axis=0)
march_u_diff_2p5km = ma.mean(MERRA2_U[march_t_indices[0], 1, :, :] - MISR_U[march_t_indices[0], 1, :, :], axis=0)
march_v_diff_2p5km = ma.mean(MERRA2_V[march_t_indices[0], 1, :, :] - MISR_V[march_t_indices[0], 1, :, :], axis=0)
march_u_diff_4km = ma.mean(MERRA2_U[march_t_indices[0], 2, :, :] - MISR_U[march_t_indices[0], 2, :, :], axis=0)
march_v_diff_4km = ma.mean(MERRA2_V[march_t_indices[0], 1, :, :] - MISR_V[march_t_indices[0], 1, :, :], axis=0)

#April
merra_april_speed_1p5km = ma.mean(MERRA2_speed[april_t_indices[0], 0, :, :], axis=0)
merra_april_speed_2p5km = ma.mean(MERRA2_speed[april_t_indices[0], 1, :, :], axis=0)
merra_april_speed_4km = ma.mean(MERRA2_speed[april_t_indices[0], 2, :, :], axis=0)
merra_april_u_1p5km = ma.mean(MERRA2_U[april_t_indices[0], 0, :, :], axis=0)
merra_april_v_1p5km = ma.mean(MERRA2_V[april_t_indices[0], 0, :, :], axis=0)
merra_april_u_2p5km = ma.mean(MERRA2_U[april_t_indices[0], 1, :, :], axis=0)
merra_april_v_2p5km = ma.mean(MERRA2_V[april_t_indices[0], 1, :, :], axis=0)
merra_april_u_4km = ma.mean(MERRA2_U[april_t_indices[0], 2, :, :], axis=0)
merra_april_v_4km = ma.mean(MERRA2_V[april_t_indices[0], 2, :, :], axis=0)
misr_april_speed_1p5km = ma.mean(MISR_speed[april_t_indices[0], 0, :, :], axis=0)
misr_april_speed_2p5km = ma.mean(MISR_speed[april_t_indices[0], 1, :, :], axis=0)
misr_april_speed_4km = ma.mean(MISR_speed[april_t_indices[0], 2, :, :], axis=0)
misr_april_u_1p5km = ma.mean(MISR_U[april_t_indices[0], 0, :, :], axis=0)
misr_april_v_1p5km = ma.mean(MISR_V[april_t_indices[0], 0, :, :], axis=0)
misr_april_u_2p5km = ma.mean(MISR_U[april_t_indices[0], 1, :, :], axis=0)
misr_april_v_2p5km = ma.mean(MISR_V[april_t_indices[0], 1, :, :], axis=0)
misr_april_u_4km = ma.mean(MISR_U[april_t_indices[0], 2, :, :], axis=0)
misr_april_v_4km = ma.mean(MISR_V[april_t_indices[0], 2, :, :], axis=0)
april_speed_diff_1p5km = ma.mean(MERRA2_speed[april_t_indices[0], 0, :, :] - MISR_speed[april_t_indices[0], 0, :, :], axis=0)
april_speed_diff_2p5km = ma.mean(MERRA2_speed[april_t_indices[0], 1, :, :] - MISR_speed[april_t_indices[0], 1, :, :], axis=0)
april_speed_diff_4km = ma.mean(MERRA2_speed[april_t_indices[0], 2, :, :] - MISR_speed[april_t_indices[0], 2, :, :], axis=0)
april_u_diff_1p5km = ma.mean(MERRA2_U[april_t_indices[0], 0, :, :] - MISR_U[april_t_indices[0], 0, :, :], axis=0)
april_v_diff_1p5km = ma.mean(MERRA2_V[april_t_indices[0], 0, :, :] - MISR_V[april_t_indices[0], 0, :, :], axis=0)
april_u_diff_2p5km = ma.mean(MERRA2_U[april_t_indices[0], 1, :, :] - MISR_U[april_t_indices[0], 1, :, :], axis=0)
april_v_diff_2p5km = ma.mean(MERRA2_V[april_t_indices[0], 1, :, :] - MISR_V[april_t_indices[0], 1, :, :], axis=0)
april_u_diff_4km = ma.mean(MERRA2_U[april_t_indices[0], 2, :, :] - MISR_U[april_t_indices[0], 2, :, :], axis=0)
april_v_diff_4km = ma.mean(MERRA2_V[april_t_indices[0], 1, :, :] - MISR_V[april_t_indices[0], 1, :, :], axis=0)

##Shears

#DO THIS LATER
#DO THIS LATER
#DO THIS LATER
#DO THIS LATER
#DO THIS LATER
#DO THIS LATER
#DO THIS LATER
#DO THIS LATER
#DO THIS LATER


########################################## PLOTS ##########################################

####Contour plots####
###Speed Differences

# #Overall
# plot_local_contour_map_maskedland(overall_speed_diff_1p5km, lon, lat, land_mask_for_plotting, np.arange(21)*0.5-5, figure_file='merra-misr overall speed difference 1p5km', title='MERRA - MISR wind speed difference, overall, 1.5 km', cmap=cm.RdBu_r)
# plot_local_contour_map_maskedland(overall_speed_diff_2p5km, lon, lat, land_mask_for_plotting, np.arange(21)*0.5-5, figure_file='merra-misr overall speed difference 2p5km', title='MERRA - MISR wind speed difference, overall, 2.5 km', cmap=cm.RdBu_r)
# plot_local_contour_map_maskedland(overall_speed_diff_4km, lon, lat, land_mask_for_plotting, np.arange(21)*0.5-5, figure_file='merra-misr overall speed difference 4km', title='MERRA - MISR wind speed difference, overall, 4 km', cmap=cm.RdBu_r)

#November
plot_local_contour_map_maskedland(november_speed_diff_1p5km, lon, lat, land_mask_for_plotting, np.arange(21)*0.5-5, figure_file='merra-misr november speed difference 1p5km', title='MERRA - MISR wind speed difference, November, 1.5 km', cmap=cm.RdBu_r)
plot_local_contour_map_maskedland(november_speed_diff_2p5km, lon, lat, land_mask_for_plotting, np.arange(21)*0.5-5, figure_file='merra-misr november speed difference 2p5km', title='MERRA - MISR wind speed difference, November, 2.5 km', cmap=cm.RdBu_r)
plot_local_contour_map_maskedland(november_speed_diff_4km, lon, lat, land_mask_for_plotting, np.arange(21)*0.5-5, figure_file='merra-misr november speed difference 4km', title='MERRA - MISR wind speed difference, November, 4 km', cmap=cm.RdBu_r)

#December
plot_local_contour_map_maskedland(december_speed_diff_1p5km, lon, lat, land_mask_for_plotting, np.arange(21)*0.5-5, figure_file='merra-misr december speed difference 1p5km', title='MERRA - MISR wind speed difference, December, 1.5 km', cmap=cm.RdBu_r)
plot_local_contour_map_maskedland(december_speed_diff_2p5km, lon, lat, land_mask_for_plotting, np.arange(21)*0.5-5, figure_file='merra-misr december speed difference 2p5km', title='MERRA - MISR wind speed difference, December, 2.5 km', cmap=cm.RdBu_r)
plot_local_contour_map_maskedland(december_speed_diff_4km, lon, lat, land_mask_for_plotting, np.arange(21)*0.5-5, figure_file='merra-misr december speed difference 4km', title='MERRA - MISR wind speed difference, December, 4 km', cmap=cm.RdBu_r)

#January
plot_local_contour_map_maskedland(january_speed_diff_1p5km, lon, lat, land_mask_for_plotting, np.arange(21)*0.5-5, figure_file='merra-misr january speed difference 1p5km', title='MERRA - MISR wind speed difference, January, 1.5 km', cmap=cm.RdBu_r)
plot_local_contour_map_maskedland(january_speed_diff_2p5km, lon, lat, land_mask_for_plotting, np.arange(21)*0.5-5, figure_file='merra-misr january speed difference 2p5km', title='MERRA - MISR wind speed difference, January, 2.5 km', cmap=cm.RdBu_r)
plot_local_contour_map_maskedland(january_speed_diff_4km, lon, lat, land_mask_for_plotting, np.arange(21)*0.5-5, figure_file='merra-misr january speed difference 4km', title='MERRA - MISR wind speed difference, January, 4 km', cmap=cm.RdBu_r)

#February
plot_local_contour_map_maskedland(february_speed_diff_1p5km, lon, lat, land_mask_for_plotting, np.arange(21)*0.5-5, figure_file='merra-misr february speed difference 1p5km', title='MERRA - MISR wind speed difference, February, 1.5 km', cmap=cm.RdBu_r)
plot_local_contour_map_maskedland(february_speed_diff_2p5km, lon, lat, land_mask_for_plotting, np.arange(21)*0.5-5, figure_file='merra-misr february speed difference 2p5km', title='MERRA - MISR wind speed difference, February, 2.5 km', cmap=cm.RdBu_r)
plot_local_contour_map_maskedland(february_speed_diff_4km, lon, lat, land_mask_for_plotting, np.arange(21)*0.5-5, figure_file='merra-misr february speed difference 4km', title='MERRA - MISR wind speed difference, February, 4 km', cmap=cm.RdBu_r)

#March
plot_local_contour_map_maskedland(march_speed_diff_1p5km, lon, lat, land_mask_for_plotting, np.arange(21)*0.5-5, figure_file='merra-misr march speed difference 1p5km', title='MERRA - MISR wind speed difference, March, 1.5 km', cmap=cm.RdBu_r)
plot_local_contour_map_maskedland(march_speed_diff_2p5km, lon, lat, land_mask_for_plotting, np.arange(21)*0.5-5, figure_file='merra-misr march speed difference 2p5km', title='MERRA - MISR wind speed difference, March, 2.5 km', cmap=cm.RdBu_r)
plot_local_contour_map_maskedland(march_speed_diff_4km, lon, lat, land_mask_for_plotting, np.arange(21)*0.5-5, figure_file='merra-misr march speed difference 4km', title='MERRA - MISR wind speed difference, March, 4 km', cmap=cm.RdBu_r)

#April
plot_local_contour_map_maskedland(april_speed_diff_1p5km, lon, lat, land_mask_for_plotting, np.arange(21)*0.5-5, figure_file='merra-misr april speed difference 1p5km', title='MERRA - MISR wind speed difference, April, 1.5 km', cmap=cm.RdBu_r)
plot_local_contour_map_maskedland(april_speed_diff_2p5km, lon, lat, land_mask_for_plotting, np.arange(21)*0.5-5, figure_file='merra-misr april speed difference 2p5km', title='MERRA - MISR wind speed difference, April, 2.5 km', cmap=cm.RdBu_r)
plot_local_contour_map_maskedland(april_speed_diff_4km, lon, lat, land_mask_for_plotting, np.arange(21)*0.5-5, figure_file='merra-misr april speed difference 4km', title='MERRA - MISR wind speed difference, April, 4 km', cmap=cm.RdBu_r)


####Vector plots####
#Note: not doing shears for this - can do later if there's any interest but tbh I don't think many of these plots will be that useful

###Speed differences

# ##Overall
# #MERRA
# plot_local_vector_map_maskedland(merra_overall_u_1p5km, merra_overall_v_1p5km, lon, lat, land_mask_for_plotting, figure_file='merra vectors overall 1p5km', title='MERRA wind vectors, Overall, 1.5 km')
# plot_local_vector_map_maskedland(merra_overall_u_2p5km, merra_overall_v_2p5km, lon, lat, land_mask_for_plotting, figure_file='merra vectors overall 2p5km', title='MERRA wind vectors, Overall, 2.5 km')
# plot_local_vector_map_maskedland(merra_overall_u_4km, merra_overall_v_4km, lon, lat, land_mask_for_plotting, figure_file='merra vectors overall 4km', title='MERRA wind vectors, Overall, 4 km')
# #MISR
# plot_local_vector_map_maskedland(misr_overall_u_1p5km, misr_overall_v_1p5km, lon, lat, land_mask_for_plotting, figure_file='misr vectors overall 1p5km', title='MISR wind vectors, Overall, 1.5 km')
# plot_local_vector_map_maskedland(misr_overall_u_2p5km, misr_overall_v_2p5km, lon, lat, land_mask_for_plotting, figure_file='misr vectors overall 2p5km', title='MISR wind vectors, Overall, 2.5 km')
# plot_local_vector_map_maskedland(misr_overall_u_4km, misr_overall_v_4km, lon, lat, land_mask_for_plotting, figure_file='misr vectors overall 4km', title='MISR wind vectors, Overall, 4 km')
# #Differences
# plot_local_vector_map_maskedland(overall_u_diff_1p5km, overall_v_diff_1p5km, lon, lat, land_mask_for_plotting, figure_file='merra-misr vectors overall 1p5km', title='MERRA - MISR wind vectors, Overall, 1.5 km')
# plot_local_vector_map_maskedland(overall_u_diff_2p5km, overall_v_diff_2p5km, lon, lat, land_mask_for_plotting, figure_file='merra-misr vectors overall 2p5km', title='MERRA - MISR wind vectors, Overall, 2.5 km')
# plot_local_vector_map_maskedland(overall_u_diff_4km, overall_v_diff_4km, lon, lat, land_mask_for_plotting, figure_file='merra-misr vectors overall 4km', title='MERRA - MISR wind vectors, Overall, 4 km')

##November
#MERRA
plot_local_vector_map_maskedland(merra_november_u_1p5km, merra_november_v_1p5km, lon, lat, land_mask_for_plotting, figure_file='merra vectors november 1p5km', title='MERRA wind vectors, November, 1.5 km')
plot_local_vector_map_maskedland(merra_november_u_2p5km, merra_november_v_2p5km, lon, lat, land_mask_for_plotting, figure_file='merra vectors november 2p5km', title='MERRA wind vectors, November, 2.5 km')
plot_local_vector_map_maskedland(merra_november_u_4km, merra_november_v_4km, lon, lat, land_mask_for_plotting, figure_file='merra vectors november 4km', title='MERRA wind vectors, November, 4 km')
#MISR
plot_local_vector_map_maskedland(misr_november_u_1p5km, misr_november_v_1p5km, lon, lat, land_mask_for_plotting, figure_file='misr vectors november 1p5km', title='MISR wind vectors, November, 1.5 km')
plot_local_vector_map_maskedland(misr_november_u_2p5km, misr_november_v_2p5km, lon, lat, land_mask_for_plotting, figure_file='misr vectors november 2p5km', title='MISR wind vectors, November, 2.5 km')
plot_local_vector_map_maskedland(misr_november_u_4km, misr_november_v_4km, lon, lat, land_mask_for_plotting, figure_file='misr vectors november 4km', title='MISR wind vectors, November, 4 km')
#Differences
plot_local_vector_map_maskedland(november_u_diff_1p5km, november_v_diff_1p5km, lon, lat, land_mask_for_plotting, figure_file='merra-misr vectors november 1p5km', title='MERRA - MISR wind vectors, November, 1.5 km')
plot_local_vector_map_maskedland(november_u_diff_2p5km, november_v_diff_2p5km, lon, lat, land_mask_for_plotting, figure_file='merra-misr vectors november 2p5km', title='MERRA - MISR wind vectors, November, 2.5 km')
plot_local_vector_map_maskedland(november_u_diff_4km, november_v_diff_4km, lon, lat, land_mask_for_plotting, figure_file='merra-misr vectors november 4km', title='MERRA - MISR wind vectors, November, 4 km')

##December
#MERRA
plot_local_vector_map_maskedland(merra_december_u_1p5km, merra_december_v_1p5km, lon, lat, land_mask_for_plotting, figure_file='merra vectors december 1p5km', title='MERRA wind vectors, December, 1.5 km')
plot_local_vector_map_maskedland(merra_december_u_2p5km, merra_december_v_2p5km, lon, lat, land_mask_for_plotting, figure_file='merra vectors december 2p5km', title='MERRA wind vectors, December, 2.5 km')
plot_local_vector_map_maskedland(merra_december_u_4km, merra_december_v_4km, lon, lat, land_mask_for_plotting, figure_file='merra vectors december 4km', title='MERRA wind vectors, December, 4 km')
#MISR
plot_local_vector_map_maskedland(misr_december_u_1p5km, misr_december_v_1p5km, lon, lat, land_mask_for_plotting, figure_file='misr vectors december 1p5km', title='MISR wind vectors, December, 1.5 km')
plot_local_vector_map_maskedland(misr_december_u_2p5km, misr_december_v_2p5km, lon, lat, land_mask_for_plotting, figure_file='misr vectors december 2p5km', title='MISR wind vectors, December, 2.5 km')
plot_local_vector_map_maskedland(misr_december_u_4km, misr_december_v_4km, lon, lat, land_mask_for_plotting, figure_file='misr vectors december 4km', title='MISR wind vectors, December, 4 km')
#Differences
plot_local_vector_map_maskedland(december_u_diff_1p5km, december_v_diff_1p5km, lon, lat, land_mask_for_plotting, figure_file='merra-misr vectors december 1p5km', title='MERRA - MISR wind vectors, December, 1.5 km')
plot_local_vector_map_maskedland(december_u_diff_2p5km, december_v_diff_2p5km, lon, lat, land_mask_for_plotting, figure_file='merra-misr vectors december 2p5km', title='MERRA - MISR wind vectors, December, 2.5 km')
plot_local_vector_map_maskedland(december_u_diff_4km, december_v_diff_4km, lon, lat, land_mask_for_plotting, figure_file='merra-misr vectors december 4km', title='MERRA - MISR wind vectors, December, 4 km')

##January
#MERRA
plot_local_vector_map_maskedland(merra_january_u_1p5km, merra_january_v_1p5km, lon, lat, land_mask_for_plotting, figure_file='merra vectors january 1p5km', title='MERRA wind vectors, January, 1.5 km')
plot_local_vector_map_maskedland(merra_january_u_2p5km, merra_january_v_2p5km, lon, lat, land_mask_for_plotting, figure_file='merra vectors january 2p5km', title='MERRA wind vectors, January, 2.5 km')
plot_local_vector_map_maskedland(merra_january_u_4km, merra_january_v_4km, lon, lat, land_mask_for_plotting, figure_file='merra vectors january 4km', title='MERRA wind vectors, January, 4 km')
#MISR
plot_local_vector_map_maskedland(misr_january_u_1p5km, misr_january_v_1p5km, lon, lat, land_mask_for_plotting, figure_file='misr vectors january 1p5km', title='MISR wind vectors, January, 1.5 km')
plot_local_vector_map_maskedland(misr_january_u_2p5km, misr_january_v_2p5km, lon, lat, land_mask_for_plotting, figure_file='misr vectors january 2p5km', title='MISR wind vectors, January, 2.5 km')
plot_local_vector_map_maskedland(misr_january_u_4km, misr_january_v_4km, lon, lat, land_mask_for_plotting, figure_file='misr vectors january 4km', title='MISR wind vectors, January, 4 km')
#Differences
plot_local_vector_map_maskedland(january_u_diff_1p5km, january_v_diff_1p5km, lon, lat, land_mask_for_plotting, figure_file='merra-misr vectors january 1p5km', title='MERRA - MISR wind vectors, January, 1.5 km')
plot_local_vector_map_maskedland(january_u_diff_2p5km, january_v_diff_2p5km, lon, lat, land_mask_for_plotting, figure_file='merra-misr vectors january 2p5km', title='MERRA - MISR wind vectors, January, 2.5 km')
plot_local_vector_map_maskedland(january_u_diff_4km, january_v_diff_4km, lon, lat, land_mask_for_plotting, figure_file='merra-misr vectors january 4km', title='MERRA - MISR wind vectors, January, 4 km')

##February
#MERRA
plot_local_vector_map_maskedland(merra_february_u_1p5km, merra_february_v_1p5km, lon, lat, land_mask_for_plotting, figure_file='merra vectors february 1p5km', title='MERRA wind vectors, February, 1.5 km')
plot_local_vector_map_maskedland(merra_february_u_2p5km, merra_february_v_2p5km, lon, lat, land_mask_for_plotting, figure_file='merra vectors february 2p5km', title='MERRA wind vectors, February, 2.5 km')
plot_local_vector_map_maskedland(merra_february_u_4km, merra_february_v_4km, lon, lat, land_mask_for_plotting, figure_file='merra vectors february 4km', title='MERRA wind vectors, February, 4 km')
#MISR
plot_local_vector_map_maskedland(misr_february_u_1p5km, misr_february_v_1p5km, lon, lat, land_mask_for_plotting, figure_file='misr vectors february 1p5km', title='MISR wind vectors, February, 1.5 km')
plot_local_vector_map_maskedland(misr_february_u_2p5km, misr_february_v_2p5km, lon, lat, land_mask_for_plotting, figure_file='misr vectors february 2p5km', title='MISR wind vectors, February, 2.5 km')
plot_local_vector_map_maskedland(misr_february_u_4km, misr_february_v_4km, lon, lat, land_mask_for_plotting, figure_file='misr vectors february 4km', title='MISR wind vectors, February, 4 km')
#Differences
plot_local_vector_map_maskedland(february_u_diff_1p5km, february_v_diff_1p5km, lon, lat, land_mask_for_plotting, figure_file='merra-misr vectors february 1p5km', title='MERRA - MISR wind vectors, February, 1.5 km')
plot_local_vector_map_maskedland(february_u_diff_2p5km, february_v_diff_2p5km, lon, lat, land_mask_for_plotting, figure_file='merra-misr vectors february 2p5km', title='MERRA - MISR wind vectors, February, 2.5 km')
plot_local_vector_map_maskedland(february_u_diff_4km, february_v_diff_4km, lon, lat, land_mask_for_plotting, figure_file='merra-misr vectors february 4km', title='MERRA - MISR wind vectors, February, 4 km')

##March
#MERRA
plot_local_vector_map_maskedland(merra_march_u_1p5km, merra_march_v_1p5km, lon, lat, land_mask_for_plotting, figure_file='merra vectors march 1p5km', title='MERRA wind vectors, March, 1.5 km')
plot_local_vector_map_maskedland(merra_march_u_2p5km, merra_march_v_2p5km, lon, lat, land_mask_for_plotting, figure_file='merra vectors march 2p5km', title='MERRA wind vectors, March, 2.5 km')
plot_local_vector_map_maskedland(merra_march_u_4km, merra_march_v_4km, lon, lat, land_mask_for_plotting, figure_file='merra vectors march 4km', title='MERRA wind vectors, March, 4 km')
#MISR
plot_local_vector_map_maskedland(misr_march_u_1p5km, misr_march_v_1p5km, lon, lat, land_mask_for_plotting, figure_file='misr vectors march 1p5km', title='MISR wind vectors, March, 1.5 km')
plot_local_vector_map_maskedland(misr_march_u_2p5km, misr_march_v_2p5km, lon, lat, land_mask_for_plotting, figure_file='misr vectors march 2p5km', title='MISR wind vectors, March, 2.5 km')
plot_local_vector_map_maskedland(misr_march_u_4km, misr_march_v_4km, lon, lat, land_mask_for_plotting, figure_file='misr vectors march 4km', title='MISR wind vectors, March, 4 km')
#Differences
plot_local_vector_map_maskedland(march_u_diff_1p5km, march_v_diff_1p5km, lon, lat, land_mask_for_plotting, figure_file='merra-misr vectors march 1p5km', title='MERRA - MISR wind vectors, March, 1.5 km')
plot_local_vector_map_maskedland(march_u_diff_2p5km, march_v_diff_2p5km, lon, lat, land_mask_for_plotting, figure_file='merra-misr vectors march 2p5km', title='MERRA - MISR wind vectors, March, 2.5 km')
plot_local_vector_map_maskedland(march_u_diff_4km, march_v_diff_4km, lon, lat, land_mask_for_plotting, figure_file='merra-misr vectors march 4km', title='MERRA - MISR wind vectors, March, 4 km')

##April
#MERRA
plot_local_vector_map_maskedland(merra_april_u_1p5km, merra_april_v_1p5km, lon, lat, land_mask_for_plotting, figure_file='merra vectors april 1p5km', title='MERRA wind vectors, April, 1.5 km')
plot_local_vector_map_maskedland(merra_april_u_2p5km, merra_april_v_2p5km, lon, lat, land_mask_for_plotting, figure_file='merra vectors april 2p5km', title='MERRA wind vectors, April, 2.5 km')
plot_local_vector_map_maskedland(merra_april_u_4km, merra_april_v_4km, lon, lat, land_mask_for_plotting, figure_file='merra vectors april 4km', title='MERRA wind vectors, April, 4 km')
#MISR
plot_local_vector_map_maskedland(misr_april_u_1p5km, misr_april_v_1p5km, lon, lat, land_mask_for_plotting, figure_file='misr vectors april 1p5km', title='MISR wind vectors, April, 1.5 km')
plot_local_vector_map_maskedland(misr_april_u_2p5km, misr_april_v_2p5km, lon, lat, land_mask_for_plotting, figure_file='misr vectors april 2p5km', title='MISR wind vectors, April, 2.5 km')
plot_local_vector_map_maskedland(misr_april_u_4km, misr_april_v_4km, lon, lat, land_mask_for_plotting, figure_file='misr vectors april 4km', title='MISR wind vectors, April, 4 km')
#Differences
plot_local_vector_map_maskedland(april_u_diff_1p5km, april_v_diff_1p5km, lon, lat, land_mask_for_plotting, figure_file='merra-misr vectors april 1p5km', title='MERRA - MISR wind vectors, April, 1.5 km')
plot_local_vector_map_maskedland(april_u_diff_2p5km, april_v_diff_2p5km, lon, lat, land_mask_for_plotting, figure_file='merra-misr vectors april 2p5km', title='MERRA - MISR wind vectors, April, 2.5 km')
plot_local_vector_map_maskedland(april_u_diff_4km, april_v_diff_4km, lon, lat, land_mask_for_plotting, figure_file='merra-misr vectors april 4km', title='MERRA - MISR wind vectors, April, 4 km')
