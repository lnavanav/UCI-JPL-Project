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
	
#Make the mask for plotting stuff - excludes land but includes the coast	
def build_land_mask(is_land, is_coast):
	temp_mask = ~is_land.mask
	lat, lon = np.shape(is_land)	#is_land and is_coast overlap, so ~is_land and is_coast don't.
	for i in range(lat):			#need to build a mask that allows us to plot over the ocean and the coasts but NOT the land.
		for j in range(lon):		
			if is_coast[i,j] == 1:
				temp_mask[i,j] = False	#NOTE: "false" allows more of the ocean/coast region to be plotted. If "true" then it's just is_land.mask.
	return temp_mask
	
#Get the landfall dates. Not the most efficient way, I think, but at least it works for now	
def get_lfloc_dates(lfloc):
	t_list = []
	for t in range(len(lfloc)):
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

#Find all dates of landfall - returns list of dates
lfloc_dates = get_lfloc_dates(AR_lfloc) #should be 1,022 days of landfall


#####Speeds and vectors
#CHANGES FOR THE DIFFERENCE: now taking the mean of the subtracted vectors rather than difference of the means. There shouldn't be a need to define specific masks - python should automatically mask the arrays as they're subtracted,
#because subtraction is a binary operation for masked arrays (http://folk.uio.no/inf3330/scripting/doc/python/NumPy/Numeric/numpy-22.html)

#Unsure if masking in this was will cause issues. Attempting to combine masks is a horrible headache and seems unnecessary based on own testing of results.
#ISSUES - MISR data inside an AR is INCREDIBLY sparse

##All times, but only in an AR
#These are NOT time-averaged means, but rather 3077x61x57 arrays. To plot, MUST take mean along axis=0, or specify a time to plot.
merra_speed_in_AR_alltimes_1p5km = ma.array(MERRA2_speed[:, 0, :, :], mask = AR_shape.mask)
merra_speed_in_AR_alltimes_2p5km = ma.array(MERRA2_speed[:, 1, :, :], mask = AR_shape.mask)
merra_speed_in_AR_alltimes_4km = ma.array(MERRA2_speed[:, 2, :, :], mask = AR_shape.mask)
merra_u_in_AR_alltimes_1p5km = ma.array(MERRA2_U[:, 0, :, :], mask=AR_shape.mask)
merra_v_in_AR_alltimes_1p5km = ma.array(MERRA2_V[:, 0, :, :], mask=AR_shape.mask)
merra_u_in_AR_alltimes_2p5km = ma.array(MERRA2_U[:, 1, :, :], mask=AR_shape.mask)
merra_v_in_AR_alltimes_2p5km = ma.array(MERRA2_V[:, 1, :, :], mask=AR_shape.mask)
merra_u_in_AR_alltimes_4km = ma.array(MERRA2_U[:, 2, :, :], mask=AR_shape.mask)
merra_v_in_AR_alltimes_4km = ma.array(MERRA2_V[:, 2, :, :], mask=AR_shape.mask)
misr_speed_in_AR_alltimes_1p5km = ma.array(MISR_speed[:, 0, :, :], mask=AR_shape.mask)
misr_speed_in_AR_alltimes_2p5km = ma.array(MISR_speed[:, 1, :, :], mask=AR_shape.mask)
misr_speed_in_AR_alltimes_4km = ma.array(MISR_speed[:, 2, :, :], mask=AR_shape.mask)
misr_u_in_AR_alltimes_1p5km = ma.array(MISR_U[:, 0, :, :], mask=AR_shape.mask)
misr_v_in_AR_alltimes_1p5km = ma.array(MISR_V[:, 0, :, :], mask=AR_shape.mask)
misr_u_in_AR_alltimes_2p5km = ma.array(MISR_U[:, 1, :, :], mask=AR_shape.mask)
misr_v_in_AR_alltimes_2p5km = ma.array(MISR_V[:, 1, :, :], mask=AR_shape.mask)
misr_u_in_AR_alltimes_4km = ma.array(MISR_U[:, 2, :, :], mask=AR_shape.mask)
misr_v_in_AR_alltimes_4km = ma.array(MISR_V[:, 2, :, :], mask=AR_shape.mask)
speed_in_AR_alltimes_diff_1p5km = ma.array(MERRA2_speed[:, 0, :, :] - MISR_speed[:, 0, :, :], mask=AR_shape.mask)
speed_in_AR_alltimes_diff_2p5km = ma.array(MERRA2_speed[:, 1, :, :] - MISR_speed[:, 1, :, :], mask=AR_shape.mask)
speed_in_AR_alltimes_diff_4km = ma.array(MERRA2_speed[:, 2, :, :] - MISR_speed[:, 2, :, :], mask=AR_shape.mask)
u_in_AR_alltimes_diff_1p5km = ma.array(MERRA2_U[:, 0, :, :] - MISR_U[:, 0, :, :], mask=AR_shape.mask)
v_in_AR_alltimes_diff_1p5km = ma.array(MERRA2_V[:, 0, :, :] - MISR_V[:, 0, :, :], mask=AR_shape.mask)
u_in_AR_alltimes_diff_2p5km = ma.array(MERRA2_U[:, 1, :, :] - MISR_U[:, 1, :, :], mask=AR_shape.mask)
v_in_AR_alltimes_diff_2p5km = ma.array(MERRA2_V[:, 1, :, :] - MISR_V[:, 1, :, :], mask=AR_shape.mask)
u_in_AR_alltimes_diff_4km = ma.array(MERRA2_U[:, 2, :, :] - MISR_U[:, 2, :, :], mask=AR_shape.mask)
v_in_AR_alltimes_diff_4km = ma.array(MERRA2_V[:, 1, :, :] - MISR_V[:, 1, :, :], mask=AR_shape.mask)



##Landfall dates only, but only in an AR
#These are NOT time-averaged means, but rather 3077x61x57 arrays. To plot, MUST take mean along axis=0, or specify a time to plot.
merra_speed_in_AR_landfalltimes_1p5km = ma.array(MERRA2_speed[lfloc_dates, 0, :, :], mask = AR_shape.mask[lfloc_dates])
merra_speed_in_AR_landfalltimes_2p5km = ma.array(MERRA2_speed[lfloc_dates, 1, :, :], mask = AR_shape.mask[lfloc_dates])
merra_speed_in_AR_landfalltimes_4km = ma.array(MERRA2_speed[lfloc_dates, 2, :, :], mask = AR_shape.mask[lfloc_dates])
merra_u_in_AR_landfalltimes_1p5km = ma.array(MERRA2_U[lfloc_dates, 0, :, :], mask=AR_shape.mask[lfloc_dates])
merra_v_in_AR_landfalltimes_1p5km = ma.array(MERRA2_V[lfloc_dates, 0, :, :], mask=AR_shape.mask[lfloc_dates])
merra_u_in_AR_landfalltimes_2p5km = ma.array(MERRA2_U[lfloc_dates, 1, :, :], mask=AR_shape.mask[lfloc_dates])
merra_v_in_AR_landfalltimes_2p5km = ma.array(MERRA2_V[lfloc_dates, 1, :, :], mask=AR_shape.mask[lfloc_dates])
merra_u_in_AR_landfalltimes_4km = ma.array(MERRA2_U[lfloc_dates, 2, :, :], mask=AR_shape.mask[lfloc_dates])
merra_v_in_AR_landfalltimes_4km = ma.array(MERRA2_V[lfloc_dates, 2, :, :], mask=AR_shape.mask[lfloc_dates])
misr_speed_in_AR_landfalltimes_1p5km = ma.array(MISR_speed[lfloc_dates, 0, :, :], mask=AR_shape.mask[lfloc_dates])
misr_speed_in_AR_landfalltimes_2p5km = ma.array(MISR_speed[lfloc_dates, 1, :, :], mask=AR_shape.mask[lfloc_dates])
misr_speed_in_AR_landfalltimes_4km = ma.array(MISR_speed[lfloc_dates, 2, :, :], mask=AR_shape.mask[lfloc_dates])
misr_u_in_AR_landfalltimes_1p5km = ma.array(MISR_U[lfloc_dates, 0, :, :], mask=AR_shape.mask[lfloc_dates])
misr_v_in_AR_landfalltimes_1p5km = ma.array(MISR_V[lfloc_dates, 0, :, :], mask=AR_shape.mask[lfloc_dates])
misr_u_in_AR_landfalltimes_2p5km = ma.array(MISR_U[lfloc_dates, 1, :, :], mask=AR_shape.mask[lfloc_dates])
misr_v_in_AR_landfalltimes_2p5km = ma.array(MISR_V[lfloc_dates, 1, :, :], mask=AR_shape.mask[lfloc_dates])
misr_u_in_AR_landfalltimes_4km = ma.array(MISR_U[lfloc_dates, 2, :, :], mask=AR_shape.mask[lfloc_dates])
misr_v_in_AR_landfalltimes_4km = ma.array(MISR_V[lfloc_dates, 2, :, :], mask=AR_shape.mask[lfloc_dates])
speed_in_AR_landfalltimes_diff_1p5km = ma.array(MERRA2_speed[lfloc_dates, 0, :, :] - MISR_speed[lfloc_dates, 0, :, :], mask=AR_shape.mask[lfloc_dates])
speed_in_AR_landfalltimes_diff_2p5km = ma.array(MERRA2_speed[lfloc_dates, 1, :, :] - MISR_speed[lfloc_dates, 1, :, :], mask=AR_shape.mask[lfloc_dates])
speed_in_AR_landfalltimes_diff_4km = ma.array(MERRA2_speed[lfloc_dates, 2, :, :] - MISR_speed[lfloc_dates, 2, :, :], mask=AR_shape.mask[lfloc_dates])
u_in_AR_landfalltimes_diff_1p5km = ma.array(MERRA2_U[lfloc_dates, 0, :, :] - MISR_U[lfloc_dates, 0, :, :], mask=AR_shape.mask[lfloc_dates])
v_in_AR_landfalltimes_diff_1p5km = ma.array(MERRA2_V[lfloc_dates, 0, :, :] - MISR_V[lfloc_dates, 0, :, :], mask=AR_shape.mask[lfloc_dates])
u_in_AR_landfalltimes_diff_2p5km = ma.array(MERRA2_U[lfloc_dates, 1, :, :] - MISR_U[lfloc_dates, 1, :, :], mask=AR_shape.mask[lfloc_dates])
v_in_AR_landfalltimes_diff_2p5km = ma.array(MERRA2_V[lfloc_dates, 1, :, :] - MISR_V[lfloc_dates, 1, :, :], mask=AR_shape.mask[lfloc_dates])
u_in_AR_landfalltimes_diff_4km = ma.array(MERRA2_U[lfloc_dates, 2, :, :] - MISR_U[lfloc_dates, 2, :, :], mask=AR_shape.mask[lfloc_dates])
v_in_AR_landfalltimes_diff_4km = ma.array(MERRA2_V[lfloc_dates, 1, :, :] - MISR_V[lfloc_dates, 1, :, :], mask=AR_shape.mask[lfloc_dates])


########################################## PLOTS ##########################################

####Contour plots####
###Speed Differences

plot_local_contour_map_maskedland(ma.mean(speed_in_AR_alltimes_diff_1p5km, axis=0), lon, lat, land_mask_for_plotting, np.arange(21)*0.5-5, figure_file='merra-misr speed difference in AR all times 1p5km', title='MERRA-MISR wind speed diff in AR, all times, 1.5 km', cmap=cm.RdBu_r)
plot_local_contour_map_maskedland(ma.mean(speed_in_AR_landfalltimes_diff_1p5km, axis=0), lon, lat, land_mask_for_plotting, np.arange(21)*0.5-5, figure_file='merra-misr speed difference in AR landfall times 1p5km', title='MERRA-MISR wind speed diff in AR, landfall times, 1.5 km', cmap=cm.RdBu_r)

#NEEDS MORE HERE
